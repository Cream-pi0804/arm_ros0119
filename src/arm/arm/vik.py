#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂【速度级增量逆运动学 (Velocity / Incremental IK)】控制节点。

整体思路：
    外部（视觉/手柄）告诉机械臂“相对当前位置再走 dx, dy, dz”，
    本节点用雅可比矩阵把这个空间位移换算成 3 个电机的角度增量，
    转成脉冲下发给 STM32，再阻塞等待下位机反馈“已到位”后返回服务结果。

文件结构：
    RobotArmVelocityIK  —— 纯数学引擎（正运动学 / 雅可比 / 阻尼最小二乘求逆），不含任何 ROS 代码
    IncrementalIKNode   —— ROS 2 节点，负责通信、单位换算与“下发-等待”的握手流程
    main                —— 入口，使用多线程执行器避免服务阻塞导致的死锁
"""

import rclpy
from rclpy.node import Node
# 引入多线程和重入回调组，这是解决 ROS 2 Service 和 Subscription 互相死锁的终极方案
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading  # 提供 Event 事件，用于服务线程与话题回调线程之间的同步
import time       # 预留的计时模块（当前流程未直接使用）
import numpy as np  # 矩阵 / 向量运算的核心库

# 导入自定义的 ROS 2 消息和服务
from geometry_msgs.msg import Point          # 标准几何点消息（此文件暂未直接使用，预留）
from arm_interfaces.msg import Jointangle, Pointnow  # Jointangle: 下发的电机脉冲; Pointnow: 下位机回传的当前状态
from arm_interfaces.srv import Targetpoint   # 服务接口：请求携带目标位移，响应携带执行结果

class RobotArmVelocityIK:
    """
    机械臂运动学核心计算类：
    负责正运动学（FK）、雅可比矩阵（Jacobian）的推导，以及速度级逆运动学（IK）的求解。
    本类只做纯粹的数学计算，不涉及任何 ROS 通信。
    """
    def __init__(self):
        # 记录机械臂当前的绝对关节角度（弧度制）
        # 【重要】实际应用中，节点启动时最好能先读取下位机反馈的真实角度覆盖这里
        self.current_q = np.array([0.0, 0.0, 0.0])
        
        # 关节的硬件运动限位（弧度制），防止算法算出超出机械结构的极限角度
        # 格式：[(最小值, 最大值), ...]
        self.bounds = [
            (-np.pi/2, np.pi/2),  # 关节 1：水平旋转（-90度 到 90度）
            (0, np.pi/2),         # 关节 2：大臂俯仰（0度 到 90度）
            (0, 3*np.pi/4)        # 关节 3：小臂俯仰（0度 到 135度）
        ]

    def _get_transform(self, alpha, a, d, theta):
        """
        标准 DH (Denavit-Hartenberg) 齐次变换矩阵公式。
        用于计算相邻两个连杆坐标系之间的平移和旋转关系。

        参数（标准 DH 四参数）：
            alpha: 连杆扭转角（绕 X 轴）
            a    : 连杆长度（沿 X 轴）
            d    : 连杆偏距（沿 Z 轴）
            theta: 关节转角（绕 Z 轴）
        返回：
            4x4 齐次变换矩阵（左上 3x3 是旋转，右上 3x1 是平移）
        """
        return np.array([
            # 第 1 行：X 方向的旋转分量 与 沿 X 轴的平移 a
            [np.cos(theta), -np.sin(theta), 0, a],
            # 第 2 行：考虑 alpha 扭转后的 Y 方向分量 与 -d*sin(alpha) 平移
            [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
            # 第 3 行：Z 方向分量 与 d*cos(alpha) 平移
            [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), d*np.cos(alpha)],
            # 第 4 行：齐次坐标固定的 [0, 0, 0, 1]
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, qs):
        """
        正运动学函数：已知三个电机的角度，算出机械臂末端在空间中的 (X, Y, Z) 坐标。
        参数 qs: 当前的关节角度列表 [q1, q2, q3]
        返回: 包含末端 x, y, z 坐标的 numpy 数组
        """
        q1, q2, q3 = qs
        
        # 提取自你论文或设计的机械参数 (alpha, a, d, theta)，从基座到末端逐级排列
        # 每一行是一个连杆的 DH 参数；带 q 的项随关节角变化，常数项是机械结构固定的安装偏置
        dh_params = [
            (0, 0.135, 0.0913, q1),                    # 连杆1：底座旋转，theta 直接由 q1 驱动
            (-np.pi/2, 0.135, 0, -q2),                 # 连杆2：大臂，alpha=-90° 把坐标系翻到俯仰平面
            (0, 0.08, 0, -q2 + 3.6585*np.pi/180),      # 连杆3：与 q2 联动并叠加 3.6585° 的结构补偿角
            (0, 0.43881, 0, q3 - 3.0054),              # 连杆4：小臂，q3 减去 3.0054 rad 的零位偏置
            (-np.pi/2, 0.12286, 0.02214, 0)            # 连杆5：末端固定法兰，theta=0 只做几何偏移
        ]
        
        # t_total 用于累乘所有的 DH 矩阵，从基座一路推导到末端
        t_total = np.eye(4)
        for alpha, a, d, theta in dh_params:
            t_total = t_total @ self._get_transform(alpha, a, d, theta)
        
        # 矩阵的第0~2行，第3列就是我们要的位置坐标 (X, Y, Z)
        return t_total[0:3, 3]

    def get_position_jacobian(self, qs):
        """
        核心数学函数：利用【数值微小扰动法】求解雅可比矩阵 (Jacobian Matrix)。
        雅可比矩阵的作用：它像一个齿轮比，描述了“每个电机转动1度，末端会在X/Y/Z上移动多少毫米”。
        """
        J_v = np.zeros((3, 3)) # 初始化 3x3 矩阵，因为是 3 个电机控制 3 个空间坐标(X,Y,Z)
        delta = 1e-5  # 微小扰动量（给电机悄悄加上一个极小的角度，比如 0.00001 弧度）
        
        # 算出现有角度下的末端位置
        pos_0 = self.forward_kinematics(qs)
        
        # 遍历 3 个电机，分别给它们“捣乱”
        for i in range(3):
            q_temp = np.copy(qs)
            q_temp[i] += delta  # 给第 i 个电机加上微小偏转
            pos_1 = self.forward_kinematics(q_temp) # 看看末端移动到了哪里
            
            # 偏导数定义：位置变化量 / 角度变化量。把这个结果填入雅可比矩阵的第 i 列
            J_v[:, i] = (pos_1 - pos_0) / delta
            
        return J_v

    def solve_incremental_move(self, dx, dy, dz):
        """
        核心控制函数：将用户想要的空间位移 (dx, dy, dz) 转化为电机的角度增量 (dq1, dq2, dq3)。
        """
        # 将期望的 X,Y,Z 增量打包成向量
        delta_X = np.array([dx, dy, dz])
        
        # 1. 拿到当前姿态下的“齿轮比”（雅可比矩阵）
        J = self.get_position_jacobian(self.current_q)
        
        # 2. 求解矩阵的伪逆：使用【阻尼最小二乘法 (Damped Least Squares, DLS)】
        # 为什么要加阻尼 (lambda_sq)？
        # 当机械臂完全伸直时，数学上会出现“奇异点”，矩阵的逆会变得无穷大。
        # 这会导致你让它往前走 1 毫米，电机却疯转 100 圈。加一个极小的阻尼项能强制压制这种暴走。
        lambda_sq = 0.005 ** 2 
        # J_dls = J^T * (J * J^T + λ^2 * I)^-1  这是机器人学中最经典的抗奇异求逆公式
        J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(3))
        
        # 3. 映射：电机增量 = 伪逆雅可比矩阵 × 空间增量
        delta_q = J_dls @ delta_X
        
        # 4. 把算出来的增量加到当前角度上
        new_q = self.current_q + delta_q
        
        # 5. 安全第一：将算出的新角度进行硬件限位裁剪（低于下限取下限，高于上限取上限）
        for i in range(3):
            new_q[i] = np.clip(new_q[i], self.bounds[i][0], self.bounds[i][1])
            
        # 更新记录并返回
        self.current_q = new_q
        return self.current_q


class IncrementalIKNode(Node):
    """
    ROS 2 通信与调度节点：
    负责接收外部的服务请求、调用数学类进行计算、将弧度转为电机脉冲并发布，最后等待下位机到达。
    """
    def __init__(self):
        super().__init__('incremental_ik_node')
        
        # 创建重入回调组：这允许 Node 在“挂起等待”某个信号时，仍然能腾出其他线程去接收话题消息
        self.callback_group = ReentrantCallbackGroup()
        
        # 实例化刚才写的数学引擎
        self.ik_engine = RobotArmVelocityIK()
        
        # 线程同步事件，像一个红绿灯。没到位是红灯(clear)，到位了变绿灯(set)
        self.move_done_event = threading.Event()
        
        # 声明发布者：把算好的电机脉冲发给 STM32
        self.joint_pub = self.create_publisher(Jointangle, '/arm/Jointangle', 10)
        
        # 声明订阅者：听 STM32 反馈当前状态（有没有到位）
        self.point_sub = self.create_subscription(
            Pointnow, 
            '/arm/Pointnow', 
            self.point_now_callback, 
            10,
            callback_group=self.callback_group # 务必绑定回调组！否则等服务时听不到消息
        )

        # 声明服务端：接收来自外部（比如手柄、视觉）的 (dx, dy, dz) 增量指令
        self.srv = self.create_service(
            Targetpoint, 
            'calculate_incremental_ik', 
            self.handle_incremental_service,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('🟢 纯位置增量控制 IK 节点已启动！')

    def point_now_callback(self, msg):
        """
        话题回调函数：只要下位机不断发消息过来，就会进这个函数。
        """
        # 如果 STM32 说它走到了（is_reached == 1）
        if msg.is_reached == 1:
            # 并且事件还没被点亮（红灯状态）
            if not self.move_done_event.is_set():
                self.move_done_event.set() # 设为绿灯，放行正在死等的服务线程

    def handle_incremental_service(self, request, response):
        """
        服务回调函数：处理一次微调步进的完整流程。
        """
        # 提取用户想走的相对距离（米）
        dx, dy, dz = request.target_x, request.target_y, request.target_z
        self.get_logger().info(f'📦 收到位移增量: X: {dx:.4f}m, Y: {dy:.4f}m, Z: {dz:.4f}m')
        
        # 1. 呼叫数学引擎，传入位移增量，得到电机的绝对目标弧度
        target_qs = self.ik_engine.solve_incremental_move(dx, dy, dz)

        # 2. 弧度 -> 脉冲的物理转换
        joint_msg = Jointangle()
        
        # 提取你原本代码里的减速比/分辨率常数：每“1 度”对应多少个电机脉冲
        # 数值 = 该电机走完某段角度所需的总脉冲 / 对应的角度，由实测标定得到
        PULSE_PER_DEGREE1 = 600 / 40.0    # 关节1：转 40° 需 600 脉冲
        PULSE_PER_DEGREE2 = 10000 / 60.0  # 关节2：转 60° 需 10000 脉冲
        PULSE_PER_DEGREE3 = 2600 / 180.0  # 关节3：转 180° 需 2600 脉冲
        
        # np.degrees() 把弧度变角度，乘以比例常数，再 round() 四舍五入，最后转为 int 整数
        # 里面加入了软件安全限位 max/min，做双重保险
        # 关节1：弧度→角度→脉冲，并裁剪到 [-600, 600]；外层负号修正电机安装方向（与算法正方向相反）
        val1 = int(round(float(np.degrees(target_qs[0])) * PULSE_PER_DEGREE1))
        joint_msg.motor_1 = - max(-600, min(val1, 600)) # 注意这里的负号可能代表电机安装方向反了

        # 关节2：额外乘 2 是因为该轴存在 1:2 的机械联动/传动放大关系，裁剪到 [0, 10000]
        val2 = int(round(float(np.degrees(target_qs[1])) * 2 * PULSE_PER_DEGREE2))
        joint_msg.motor_2 = max(0, min(val2, 10000))

        # 关节3：常规换算后裁剪到 [0, 2600]
        val3 = int(round(float(np.degrees(target_qs[2])) * PULSE_PER_DEGREE3))
        joint_msg.motor_3 = max(0, min(val3, 2600))

        # 3. 把“红绿灯”切回红灯，准备下发指令
        self.move_done_event.clear()
        
        # 4. 发布给 STM32
        self.joint_pub.publish(joint_msg)
        self.get_logger().info(f'🚀 下发电机指令: M1: {joint_msg.motor_1}, M2: {joint_msg.motor_2}, M3: {joint_msg.motor_3}')
        
        # 5. 阻塞死等！线程在这里卡住，直到 STM32 发来 is_reached == 1，或者超时 2 秒
        # 因为我们走的是微小增量，动作应该很快完成，所以超时设为 2.0 秒足以
        timeout_sec = 2.0 
        is_completed = self.move_done_event.wait(timeout=timeout_sec)
        
        # 6. 根据等待结果，填写服务的回应报告给呼叫方
        if is_completed:
            response.result = Targetpoint.Response.SUCCESS 
            response.message = "步进增量完成"
        else:
            response.result = Targetpoint.Response.FAIL
            response.message = "下发成功，但等待反馈超时"
            self.get_logger().warn('⚠️ 等待电机反馈超时，但脉冲已下发。')

        return response

def main(args=None):
    # 启动 ROS 2 底层通信系统
    rclpy.init(args=args)
    
    # 实例化我们的控制节点
    node = IncrementalIKNode()
    
    # 【必须使用】多线程执行器！
    # 如果用默认的单线程执行器，当 handle_incremental_service 卡在 wait() 等待下位机时，
    # 整个 Node 会被卡死，永远无法触发 point_now_callback 去接收消息，就变成真·死锁了。
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin() # 开始循环监听
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node() # 销毁节点，释放资源
        rclpy.shutdown()    # 关闭 ROS 2

if __name__ == '__main__':
    main()