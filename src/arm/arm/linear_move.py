#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
import numpy as np

# ROS2消息和服务
from arm_interfaces.srv import LinearInterpolation, Targetpoint
from std_srvs.srv import Trigger

class UnifiedLinearServer(Node):
    def __init__(self):
        super().__init__('unified_linear_server')
        
        self.callback_group = ReentrantCallbackGroup()
        
        # 虽然我们要听总控的，但本地还是维护一个变量用于日志显示
        self.current_local_pos = np.array([0.0, 0.0, 0.0])
        self.is_moving = False
        self.stop_requested = False

        # 1. 修改后的直线插补服务
        self.service = self.create_service(
            LinearInterpolation,
            '/arm/linear_interpolation',
            self.handle_linear_interpolation,
            callback_group=self.callback_group
        )
        
        # 2. 停止服务
        self.stop_service = self.create_service(
            Trigger,
            '/arm/stop_linear',
            self.handle_stop_request,
            callback_group=self.callback_group
        )
        
        # 3. IK 客户端
        self.ik_client = self.create_client(
            Targetpoint, 'calculate_ik', callback_group=self.callback_group
        )
        
        self.get_logger().info('>>> 坐标同步版直线插补服务已启动')

    def handle_stop_request(self, request, response):
        self.stop_requested = True
        response.success = True
        response.message = "直线运动已触发停止"
        return response

    def handle_linear_interpolation(self, request, response):
        """处理请求：强制同步起点和终点"""
        # --- 核心操作：从请求中同步全局坐标 ---
        start_pt = np.array([request.start_x, request.start_y, request.start_z])
        target_pt = np.array([request.x, request.y, request.z])
        
        self.get_logger().info(f'[同步] 起点: {start_pt} -> 终点: {target_pt}')
        
        if self.is_moving:
            response.success = False
            response.message = "机械臂忙碌中"
            return response

        self.stop_requested = False
        velocity = max(request.velocity, 0.01)
        sample_time = max(request.sample_time, 0.5)

        success, message = self.execute_linear_move(start_pt, target_pt, velocity, sample_time)
        
        response.success = success
        response.message = message
        return response
    
    def execute_linear_move(self, start_pos, end_pos, velocity, sample_time):
        try:
            distance = np.linalg.norm(end_pos - start_pos)
            if distance < 0.0001:
                return True, "已在目标位置"
            
            move_time = distance / velocity
            num_points = max(2, int(move_time / sample_time))
            
            self.is_moving = True
            self.get_logger().info(f'开始直线插补：总距离 {distance:.4f}m')

            for i in range(num_points + 1):
                if self.stop_requested:
                    return False, "中途停止"

                # 插值计算
                t = i / num_points
                current_pos = start_pos + t * (end_pos - start_pos)
                
                # 调用 IK
                if not self.call_ik_service(current_pos):
                    return False, f"点 {i} IK解算失败"
                
                # 更新本地坐标缓存用于监控
                self.current_local_pos = current_pos
                
                if i % max(1, num_points // 10) == 0:
                    self.get_logger().info(f'进度: {i/num_points*100:.1f}% @ {np.round(current_pos, 4)}')
                
                time.sleep(sample_time)

            return True, "直线插补成功"
            
        except Exception as e:
            return False, str(e)
        finally:
            self.is_moving = False

    def call_ik_service(self, pos):
        req = Targetpoint.Request()
        req.target_x, req.target_y, req.target_z = map(float, pos)
        try:
            # 在 MultiThreadedExecutor 下这是安全的同步调用
            res = self.ik_client.call(req)
            return res.SUCCESS
        except Exception:
            return False

def main():
    rclpy.init()
    node = UnifiedLinearServer()
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