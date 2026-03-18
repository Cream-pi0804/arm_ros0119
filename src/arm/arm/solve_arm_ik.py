import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
import time

# 消息和服务定义
from geometry_msgs.msg import Point
from arm_interfaces.msg import Jointangle
from arm_interfaces.msg import Pointnow
from arm_interfaces.srv import Targetpoint 

import numpy as np
from scipy.optimize import minimize

class RobotArmIK:
    """[保留 RobotArmIK 类逻辑，保持不变]"""
    def __init__(self):
        self.bounds = [(-np.pi/2, np.pi/2), (0, np.pi/2), (0, np.pi)]
        self.target_pos = np.array([0, 0, 0])
        self.last_motor_1 = 0
        self.last_motor_2 = 0
        self.last_motor_3 = 0
    def _get_transform(self, alpha, a, d, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
            [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), d*np.cos(alpha)],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, qs):
        q1, q2, q3 = qs
        dh_params = [
            (0, 0.135, 0.0913, q1),
            (-np.pi/2, 0.135, 0, -q2),
            (0, 0.08, 0, -q2 + 3.6585*np.pi/180),
            (0, 0.43881, 0, q3 - 3.0054),
            (-np.pi/2, 0.12286, 0.02214, 0)
        ]
        t_total = np.eye(4)
        for alpha, a, d, theta in dh_params:
            t_total = t_total @ self._get_transform(alpha, a, d, theta)
        return t_total[0:3, 3]

    def solve_ik(self, target_xyz):
        self.target_pos = np.array(target_xyz)
        res = minimize(
            lambda qs: np.linalg.norm(self.forward_kinematics(qs) - self.target_pos),
            [self.last_motor_1, self.last_motor_2, self.last_motor_3],
            bounds=self.bounds,
            method='SLSQP',
            tol=1e-10
        )
        if res.success:
            self.last_motor_1, self.last_motor_2, self.last_motor_3 = res.x
            return res.x
        return None

class IKServiceNode(Node):
    def __init__(self):
        super().__init__('ik_service_node')
        
        # --- 核心修改 1: 创建重入回调组 ---
        # 允许 Service 等待时，Subscription 依然能接收消息
        self.callback_group = ReentrantCallbackGroup()
        
        # 初始化 IK 引擎
        self.ik_engine = RobotArmIK()
        
        # --- 核心修改 2: 线程同步事件 ---
        self.move_done_event = threading.Event()
        
        # 创建发布者
        self.joint_pub = self.create_publisher(Jointangle, '/arm/Jointangle', 10)
        
        # --- 核心修改 3: 订阅 Pointnow (绑定回调组) ---
        self.point_sub = self.create_subscription(
            Pointnow,
            '/arm/Pointnow',
            self.point_now_callback,
            10,
            callback_group=self.callback_group # 关键！
        )

        # 创建服务 (绑定回调组)
        self.srv = self.create_service(
            Targetpoint, 
            'calculate_ik', 
            self.handle_ik_service,
            callback_group=self.callback_group # 关键！
        )
        
        self.last_target = [0.4, 0.3, 0.6]
        self.get_logger().info('IK ROS2 服务节点(闭环控制版)已启动...')

    def point_now_callback(self, msg):
        """实时监听电机反馈"""
        # 只要接收到 is_reached 为 1，就触发事件
        if msg.is_reached == 1:
            if not self.move_done_event.is_set():
                self.move_done_event.set() # 唤醒正在等待的服务线程

    def handle_ik_service(self, request, response):
        """服务回调函数：计算 -> 发布 -> 等待到位 -> 返回"""
        self.last_target = [request.target_x, request.target_y, request.target_z]
        self.get_logger().info(f'1. 收到请求，目标: {self.last_target}，开始求解IK...')
        
        # 1. 计算 IK
        solution = self.ik_engine.solve_ik(self.last_target)

        if solution is not None:
            # 2. 发布关节角度
            joint_msg = Jointangle()
         # 定义一个比例常数，方便后续修改（4069 应该是编码器分辨率）
            PULSE_PER_DEGREE = 4069 / 360.0
            # 转换并取整
            # 关节1（范围 -600 ~ 600）
            val1 = int(round(float(np.degrees(solution[0])) * PULSE_PER_DEGREE))
            joint_msg.motor_1 = max(-600, min(val1, 600))

            # 关节2（范围 0 ~ 10000）
            val2 = int(round(float(np.degrees(solution[1])) *2* PULSE_PER_DEGREE))
            joint_msg.motor_2 = max(0, min(val2, 10000))

            # 关节3（范围 0 ~ 2600）
            val3 = int(round(float(np.degrees(solution[2])) * PULSE_PER_DEGREE))
            joint_msg.motor_3 = max(0, min(val3, 2600))


            # joint_msg.motor_1 = int(round(float(np.degrees(solution[0])) * PULSE_PER_DEGREE))
            # joint_msg.motor_2 = int(round(float(np.degrees(solution[1])) * PULSE_PER_DEGREE))
            # joint_msg.motor_3 = int(round(float(np.degrees(solution[2])) * PULSE_PER_DEGREE))
                        
            # 在发布前，先清除“完成”标志，防止收到上一条指令的完成信号
            self.move_done_event.clear()
            
            self.joint_pub.publish(joint_msg)
            self.get_logger().info(
                        f'--- 目标脉冲整数值 --- \n'
                        f'M1: {joint_msg.motor_1}\n'
                        f'M2: {joint_msg.motor_2}\n'
                        f'M3: {joint_msg.motor_3}'
                        f'M1: {float(np.degrees(solution[0]))}\n'
                        f'M2: {float(np.degrees(solution[1]))}\n'
                        f'M3: {float(np.degrees(solution[2]))}'
                    )
            self.get_logger().info(f'2. IK求解成功，已发布控制指令，等待机械臂到位...')
            # 3. 阻塞等待到位信号 (设置超时时间，例如 10 秒)
            timeout_sec = 10.0
            is_completed = self.move_done_event.wait(timeout=timeout_sec)
            
            if is_completed:
                response.result = Targetpoint.Response.SUCCESS # 假设 Targetpoint.srv 有 success 字段
                response.message = "机械臂已成功到达目标点"
                self.get_logger().info('3. 机械臂反馈：已到达！服务返回成功。')
            else:
                response.result = Targetpoint.Response.FAIL
                response.message = f"超时：{timeout_sec}秒内未检测到到位信号"
                self.get_logger().error('3. 等待超时！服务返回失败。')
                
        else:
            response.result = Targetpoint.response.FAIL
            response.message = "IK 逆运动学无解"
            self.get_logger().warn('IK 求解失败，无法到达该位置。')

        return response

def main(args=None):
    rclpy.init(args=args)
    
    node = IKServiceNode()
    
    # --- 核心修改 4: 使用多线程执行器 ---
    # 必须使用 MultiThreadedExecutor，否则 service 调用 wait 时会死锁
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()