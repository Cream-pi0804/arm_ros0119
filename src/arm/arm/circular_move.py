#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
import numpy as np

# 假设 CircularInterpolation.srv 已增加 start_x, start_y, start_z
from arm_interfaces.srv import CircularInterpolation, Targetpoint
from std_srvs.srv import Trigger

class UnifiedCircularServer(Node):
    def __init__(self):
        super().__init__('unified_circular_server')
        
        self.callback_group = ReentrantCallbackGroup()
        self.is_moving = False
        self.stop_requested = False
        
        # 1. 圆弧插补服务
        self.service = self.create_service(
            CircularInterpolation,
            '/arm/circular_interpolation',
            self.handle_circular_interpolation,
            callback_group=self.callback_group
        )
        
        # 2. 停止服务
        self.stop_service = self.create_service(
            Trigger,
            '/arm/stop_circular',
            self.handle_stop_request,
            callback_group=self.callback_group
        )
        
        # 3. IK 客户端
        self.ik_client = self.create_client(
            Targetpoint, 'calculate_ik', callback_group=self.callback_group
        )
        
        self.get_logger().info('>>> 坐标同步版圆弧插补服务已就绪')

    def handle_stop_request(self, request, response):
        self.stop_requested = True
        response.success = True
        response.message = "圆弧运动中止指令已发出"
        return response

    def handle_circular_interpolation(self, request, response):
        """处理请求：通过请求参数完全确定空间圆弧"""
        if self.is_moving:
            response.success = False
            response.message = "机械臂忙碌中"
            return response

        # --- 核心同步：从请求中获取起点、途径点和终点 ---
        p0 = np.array([request.start_x, request.start_y, request.start_z])
        p1 = np.array([request.x1, request.y1, request.z1])
        p2 = np.array([request.x2, request.y2, request.z2])
        
        self.get_logger().info(f'[同步] 圆弧起点: {p0}')
        self.get_logger().info(f'[同步] 途径点: {p1}, 终点: {p2}')

        self.stop_requested = False
        success, message = self.execute_circular_move(p0, p1, p2, request.velocity, request.sample_time)
        
        response.success = success
        response.message = message
        return response

    def execute_circular_move(self, p0, p1, p2, vel, dt):
        try:
            # --- 空间圆弧数学计算 ---
            v1 = p1 - p0
            v2 = p2 - p0
            w = np.cross(v1, v2)
            if np.linalg.norm(w) < 1e-6:
                return False, "三点共线错误"
            
            # 建立平面坐标系
            u_axis = v1 / np.linalg.norm(v1)
            w_unit = w / np.linalg.norm(w)
            v_axis = np.cross(w_unit, u_axis)
            
            # 投影到平面求圆心 (pc) 和半径 (r)
            x1 = np.linalg.norm(v1)
            x2 = np.dot(v2, u_axis)
            y2 = np.dot(v2, v_axis)
            
            D = 2 * x1 * y2
            cx = x1 / 2
            cy = (x2**2 + y2**2 - x1*x2) / (2 * y2)
            
            radius = np.sqrt(cx**2 + cy**2)
            pc = p0 + cx * u_axis + cy * v_axis  # 三维空间圆心
            
            # 计算转角
            vec_start = p0 - pc
            vec_end = p2 - pc
            # 重新校准局部坐标系以起点为 0 度
            local_u = vec_start / np.linalg.norm(vec_start)
            local_v = np.cross(w_unit, local_u)
            
            angle_end = np.arctan2(np.dot(vec_end, local_v), np.dot(vec_end, local_u))
            if angle_end <= 0: angle_end += 2 * np.pi # 默认逆时针转动
            
            # --- 开始插补 ---
            num_points = max(2, int((radius * angle_end) / vel / dt))
            self.is_moving = True
            
            for i in range(num_points + 1):
                if self.stop_requested:
                    return False, "用户中止运动"
                
                theta = (i / num_points) * angle_end
                current_pos = pc + radius * (np.cos(theta) * local_u + np.sin(theta) * local_v)
                
                if not self.call_ik_service(current_pos):
                    return False, f"点 {i} IK解算失败"
                
                if i % max(1, num_points // 10) == 0:
                    self.get_logger().info(f'进度: {i/num_points*100:5.1f}% @ {np.round(current_pos, 4)}')
                
                time.sleep(dt)

            return True, "圆弧插补成功"
            
        except Exception as e:
            self.get_logger().error(f"计算异常: {str(e)}")
            return False, str(e)
        finally:
            self.is_moving = False

    def call_ik_service(self, pos):
        req = Targetpoint.Request()
        req.target_x, req.target_y, req.target_z = map(float, pos)
        try:
            res = self.ik_client.call(req)
            return res.SUCCESS
        except:
            return False

def main():
    rclpy.init()
    node = UnifiedCircularServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()