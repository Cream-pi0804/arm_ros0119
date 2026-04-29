#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import time
import re
import numpy as np
import math
import os

# 导入自定义的服务接口
# LinearInterpolation: 处理直线运动
# CircularInterpolation: 处理圆弧运动 (通过 起点、途径点、终点 定义)
from arm_interfaces.srv import LinearInterpolation, CircularInterpolation
from std_srvs.srv import Trigger

class GCodeInterpreterNode(Node):
    def __init__(self):
        """
        初始化解释器节点
        功能：维护机械臂状态、解析G代码、同步调用插补服务
        """
        super().__init__('gcode_interpreter_node')
        
        # --- 1. 参数与配置 ---
        self.declare_parameter('default_feed_rate', 0.05)  # 默认线速度 (m/s)
        
        # --- 2. ROS2 服务客户端初始化 ---
        # 解释器作为客户端，向插补服务端发送具体运动指令
        self.linear_client = self.create_client(LinearInterpolation, '/arm/linear_interpolation')
        self.circular_client = self.create_client(CircularInterpolation, '/arm/circular_interpolation')
        
        # --- 3. 解释器内部状态机 (Internal State) ---
        # [重要] current_pos 必须初始化为机械臂真实的物理位置，否则首行指令会产生跳变
        self.current_pos = np.array([0.6, 0.0, 0.5]) 
        self.feed_rate = self.get_parameter('default_feed_rate').value
        
        # 默认加工平面: G17=XY(桌面), G18=ZX(正前方), G19=YZ(侧面)
        self.active_plane = 17  
        self.is_running = False

        self.get_logger().info('==========================================')
        self.get_logger().info('  ROS2 G-Code Interpreter (3D 增强版) 已启动')
        self.get_logger().info('  支持指令: G00/G01, G02/G03, G17/G18/G19, F')
        self.get_logger().info(f'  当前起始坐标: {self.current_pos}')
        self.get_logger().info('==========================================')

    def _parse_line(self, line):
        """
        正则解析函数：将一行文本转为字典映射
        示例: "G01 X0.5 Y-0.2 ; 移动" -> {'G': 1.0, 'X': 0.5, 'Y': -0.2}
        """
        # 1. 去掉括号注释 ( ) 和分号注释 ;
        line = re.sub(r'\(.*?\)', '', line).split(';')[0].upper().strip()
        if not line:
            return None
            
        # 2. 使用正则表达式匹配 [字母][数字] 组合
        # 支持整数、小数、科学计数法及正负号
        matches = re.findall(r'([A-Z])([-+]?\d*\.\d+|\d+)', line)
        
        # 返回结果字典，例如 {'G': 1.0, 'X': 0.7}
        return {k: float(v) for k, v in matches}

    def _calculate_via_point(self, p_start, p_end, i, j, k, is_cw):
        """
        核心数学函数：3D 空间圆弧途径点计算
        原因：G代码使用 (终点 + 圆心偏移量) 定义圆弧，但我们的服务需要 (起点, 途径点, 终点)
        
        参数:
            p_start, p_end: 起始点和终点的 3D 坐标
            i, j, k: 圆心相对于起始点的 X, Y, Z 轴偏移量
            is_cw: 是否为顺时针 (G02)
        """
        try:
            # --- 步骤 1: 根据 G17/18/19 映射坐标轴索引 ---
            # idx1, idx2 为圆弧所在的平面轴；axis_z 为垂直于圆弧平面的轴
            if self.active_plane == 17:   # XY 平面 (I=X偏移, J=Y偏移)
                idx1, idx2, axis_z = 0, 1, 2 
                off1, off2 = i, j
            elif self.active_plane == 18: # ZX 平面 (K=Z偏移, I=X偏移)
                idx1, idx2, axis_z = 2, 0, 1 
                off1, off2 = k, i
            elif self.active_plane == 19: # YZ 平面 (J=Y偏移, K=Z偏移)
                idx1, idx2, axis_z = 1, 2, 0 
                off1, off2 = j, k
            else:
                return None

            # --- 步骤 2: 计算平面内的圆心坐标 ---
            center1 = p_start[idx1] + off1
            center2 = p_start[idx2] + off2
            
            # 根据偏移量计算半径
            radius = math.sqrt(off1**2 + off2**2)
            
            # --- 步骤 3: 使用 atan2 计算起始和终止相位角 (极坐标) ---
            a_start = math.atan2(p_start[idx2] - center2, p_start[idx1] - center1)
            a_end   = math.atan2(p_end[idx2] - center2,   p_end[idx1] - center1)
            
            # --- 步骤 4: 调整角度以匹配旋转方向 ---
            if not is_cw: # CCW 逆时针 (角度增加)
                if a_end <= a_start: a_end += 2 * math.pi
            else:         # CW 顺时针 (角度减小)
                if a_end >= a_start: a_end -= 2 * math.pi
            
            # --- 步骤 5: 取角度中点计算途径点 (Via Point) ---
            a_mid = (a_start + a_end) / 2.0
            via = np.zeros(3)
            # 圆弧平面内的坐标
            via[idx1] = center1 + radius * math.cos(a_mid)
            via[idx2] = center2 + radius * math.sin(a_mid)
            # 垂直轴处理: 线性插值 (实现螺旋线效果)
            via[axis_z] = (p_start[axis_z] + p_end[axis_z]) / 2.0
            
            return via
            
        except Exception as e:
            self.get_logger().error(f"圆弧数学转换失败: {e}")
            return None

    def call_linear_service(self, target, vel):
        """同步调用直线插补服务"""
        while not self.linear_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待直线插补服务启动...')
        
        req = LinearInterpolation.Request()
        req.start_x, req.start_y, req.start_z = self.current_pos.tolist()
        req.x, req.y, req.z = target.tolist()
        req.velocity = float(vel)
        req.sample_time = 0.02 # 50Hz 插补频率
        
        # 使用 async 调用但用 spin_until 阻塞，确保“动作执行完才解析下一行”
        future = self.linear_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        res = future.result()
        if res and res.success:
            self.current_pos = target # 更新逻辑坐标
            return True
        return False

    def call_circular_service(self, via, target, vel):
        """同步调用圆弧插补服务"""
        while not self.circular_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待圆弧插补服务启动...')
        
        req = CircularInterpolation.Request()
        req.start_x, req.start_y, req.start_z = self.current_pos.tolist()
        req.x1, req.y1, req.z1 = via.tolist()    # 途径点
        req.x2, req.y2, req.z2 = target.tolist() # 终点
        req.velocity = float(vel)
        req.sample_time = 0.02
        
        future = self.circular_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        res = future.result()
        if res and res.success:
            self.current_pos = target # 更新逻辑坐标
            return True
        return False

    def execute_gcode_file(self, file_path):
        """读取文件并顺序执行"""
        if not os.path.exists(file_path):
            self.get_logger().error(f"找不到 G代码文件: {file_path}")
            return

        self.get_logger().info(f">>> 开始处理文件: {file_path}")
        self.is_running = True
        
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            # 检查 ROS2 状态或停止标志
            if not rclpy.ok() or not self.is_running:
                break
            
            params = self._parse_line(line)
            if not params: continue # 跳过空行/注释

            # --- F指令: 更新全局进给速度 ---
            if 'F' in params: 
                self.feed_rate = params['F']

            # --- G指令: 处理核心运动逻辑 ---
            if 'G' in params:
                g_val = int(params['G'])
                
                # 1. 平面选择 (G17/18/19)
                if g_val in [17, 18, 19]:
                    self.active_plane = g_val
                    self.get_logger().info(f"行 {i+1}: 切换加工平面至 G{g_val}")
                    continue

                # 2. 直线运动 (G00 快速 / G01 控制速度)
                if g_val in [0, 1]:
                    target = self.current_pos.copy()
                    if 'X' in params: target[0] = params['X']
                    if 'Y' in params: target[1] = params['Y']
                    if 'Z' in params: target[2] = params['Z']
                    
                    # G00 时使用 2倍速，G01 使用设定的 F 值
                    vel = self.feed_rate if g_val == 1 else self.feed_rate * 2
                    if not self.call_linear_service(target, vel):
                        self.get_logger().error(f"行 {i+1}: 直线运动失败，任务终止")
                        break

                # 3. 圆弧运动 (G02 顺时针 / G03 逆时针)
                elif g_val in [2, 3]:
                    target = self.current_pos.copy()
                    if 'X' in params: target[0] = params['X']
                    if 'Y' in params: target[1] = params['Y']
                    if 'Z' in params: target[2] = params['Z']
                    
                    # 获取 IJK 偏移量（如果缺省则为 0.0）
                    i_off = params.get('I', 0.0)
                    j_off = params.get('J', 0.0)
                    k_off = params.get('K', 0.0)
                    
                    # 计算 3D 空间途径点
                    via = self._calculate_via_point(
                        self.current_pos, target, 
                        i_off, j_off, k_off, 
                        is_cw=(g_val == 2)
                    )
                    
                    if via is not None:
                        if not self.call_circular_service(via, target, self.feed_rate):
                            self.get_logger().error(f"行 {i+1}: 圆弧运动失败，任务终止")
                            break
                    else:
                        self.get_logger().error(f"行 {i+1}: 圆弧数学解算异常")
                        break
        
        self.get_logger().info(">>> 所有 G代码行执行完毕。")
        self.is_running = False

def main(args=None):
    rclpy.init(args=args)
    node = GCodeInterpreterNode()

    # 此处演示自动创建并执行一个测试文件
    file_path = 'motion_demo.txt'
    # with open(file_path, 'w') as f:
    #     f.write("G17 ; 选择 XY 平面\n")
    #     f.write("G01 X0.5 Y0.2 Z0.3 F0.2")
    #     f.write("G01 X0.7 Y0.0 Z0.5 F0.2; 直线移至起点\n")
    #     f.write("G03 X0.6 Y0.1 I-0.1 J0.0 F0.2; 逆时针画一个 90度圆弧\n")
    #     # f.write("G19 ; 切换到 YZ 垂直面\n")
    #     # f.write("G03 Y-0.1 Z0.5 J-0.1 K0.0 ; 在侧面画一个立着的弧\n")

    try:
        # 开始执行任务
        node.execute_gcode_file(file_path)
    except KeyboardInterrupt:
        node.get_logger().warn("收到用户中断请求")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()