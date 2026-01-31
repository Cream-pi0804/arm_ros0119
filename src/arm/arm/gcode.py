#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
import math
import numpy as np
from scipy.interpolate import interp1d
import threading

# ROS2消息和服务
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Pose, Point
from arm_interfaces.srv import Targetpoint 
from arm_interfaces.msg import Jointangle
from arm_interfaces.srv import GCodeTrajectory

class GCodeTrajectoryNode(Node):
    def __init__(self):
        super().__init__('gcode_trajectory_node')
        
        # 参数
        self.declare_parameter('max_velocity', 10.0)  # mm/s
        self.declare_parameter('max_acceleration', 50.0)  # mm/s²
        self.declare_parameter('sample_time', 0.5)  # 采样时间 (s)
        self.declare_parameter('ik_service_name', 'calculate_ik')
        self.declare_parameter('joint_control_topic', '/arm/Jointangle')
        
        # 获取参数
        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_acceleration = self.get_parameter('max_acceleration').value
        self.sample_time = self.get_parameter('sample_time').value
        ik_service_name = self.get_parameter('ik_service_name').value
        joint_topic = self.get_parameter('joint_control_topic').value
        
        # 初始化状态
        self.current_pos = np.array([0.0, 0.0, 0.0])  # X, Y, Z (mm)
        self.current_feedrate = 100.0  # mm/min
        self.is_moving = False
        self.stop_requested = False
        
        # 使用ReentrantCallbackGroup允许多个回调并发执行
        callback_group = ReentrantCallbackGroup()
        
        # 创建服务
        self.service = self.create_service(
            GCodeTrajectory,
            '/arm/gcode_trajectory',
            self.handle_gcode_command,
            callback_group=callback_group
        )
        
        # 创建逆运动学服务客户端
        self.ik_client = self.create_client(
            Targetpoint,
            ik_service_name,
            callback_group=callback_group
        )
        
        # 等待逆运动学服务可用
        while not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'等待逆运动学服务 {ik_service_name}...')
        
        # # 创建关节角度发布器（用于控制机械臂）
        # self.joint_pub = self.create_publisher(
        #     Jointangle,
        #     joint_topic,
        #     10
        # )
        
        # 创建状态发布器
        self.status_pub = self.create_publisher(
            String,
            '/arm/trajectory_status',
            10
        )
        
        self.get_logger().info('G代码轨迹规划节点已启动')
        self.get_logger().info(f'最大速度: {self.max_velocity} mm/s')
        self.get_logger().info(f'最大加速度: {self.max_acceleration} mm/s²')
    
    def handle_gcode_command(self, request, response):
        """处理G代码命令请求"""
        self.get_logger().info(f'收到G代码命令: {request.gcode_command}')
        
        # 重置停止标志
        self.stop_requested = False
        
        try:
            # 解析G代码
            commands = self.parse_gcode(request.gcode_command)
            
            if not commands:
                response.success = False
                response.message = "无法解析G代码"
                return response
            
            # 执行轨迹规划
            success = self.execute_trajectory(commands, request.wait_for_completion)
            
            if success:
                response.success = True
                response.message = "轨迹执行完成"
            else:
                response.success = False
                response.message = "轨迹执行失败或被中断"
                
        except Exception as e:
            self.get_logger().error(f'轨迹执行异常: {str(e)}')
            response.success = False
            response.message = f"执行异常: {str(e)}"
        
        return response
    
    def parse_gcode(self, gcode_str):
        """解析G代码字符串"""
        commands = []
        lines = gcode_str.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):  # 跳过空行和注释
                continue
            
            # 转换为大写，移除空格
            line = line.upper().replace(' ', '')
            
            try:
                cmd = {}
                
                # 解析G代码
                if 'G0' in line or 'G00' in line:
                    cmd['type'] = 'RAPID'
                elif 'G1' in line or 'G01' in line:
                    cmd['type'] = 'LINEAR'
                elif 'G2' in line or 'G02' in line:
                    cmd['type'] = 'CW_ARC'
                elif 'G3' in line or 'G03' in line:
                    cmd['type'] = 'CCW_ARC'
                elif 'G90' in line:
                    cmd['type'] = 'ABSOLUTE'
                    commands.append(cmd)
                    continue
                elif 'G91' in line:
                    cmd['type'] = 'RELATIVE'
                    commands.append(cmd)
                    continue
                else:
                    self.get_logger().warn(f'跳过不支持的G代码: {line}')
                    continue
                
                # 解析坐标参数
                if 'X' in line:
                    cmd['X'] = float(line.split('X')[1].split()[0].replace(';', ''))
                if 'Y' in line:
                    cmd['Y'] = float(line.split('Y')[1].split()[0].replace(';', ''))
                if 'Z' in line:
                    cmd['Z'] = float(line.split('Z')[1].split()[0].replace(';', ''))
                if 'F' in line:
                    feedrate = float(line.split('F')[1].split()[0].replace(';', ''))
                    cmd['F'] = feedrate
                    self.current_feedrate = feedrate
                else:
                    cmd['F'] = self.current_feedrate
                
                commands.append(cmd)
                
            except Exception as e:
                self.get_logger().error(f'解析G代码行失败 {line}: {str(e)}')
        
        return commands
    
    def execute_trajectory(self, commands, wait_for_completion=True):
        """执行轨迹规划"""
        self.is_moving = True
        self.publish_status("STARTED")
        
        try:
            for i, cmd in enumerate(commands):
                if self.stop_requested:
                    self.get_logger().info("收到停止请求，中断轨迹执行")
                    self.publish_status("STOPPED")
                    return False
                
                self.get_logger().info(f'执行命令 {i+1}/{len(commands)}: {cmd}')
                
                if cmd['type'] in ['ABSOLUTE', 'RELATIVE']:
                    # 设置坐标模式
                    self.get_logger().info(f"设置为{cmd['type']}模式")
                    continue
                
                # 计算目标位置
                target_pos = self.calculate_target_position(cmd)
                
                # 线性插补轨迹规划
                success = self.execute_linear_move(target_pos, cmd['F'])
                
                if not success:
                    self.publish_status("ERROR")
                    return False
            
            self.publish_status("COMPLETED")
            return True
            
        finally:
            self.is_moving = False
    
    def calculate_target_position(self, cmd):
        """计算目标位置"""
        target = np.copy(self.current_pos)
        
        if 'X' in cmd:
            target[0] = cmd['X']
        if 'Y' in cmd:
            target[1] = cmd['Y']
        if 'Z' in cmd:
            target[2] = cmd['Z']
        
        return target
    
    def execute_linear_move(self, target_pos, feedrate):
        """执行线性移动"""
        # 将mm/min转换为mm/s
        velocity = min(feedrate / 60.0, self.max_velocity)
        
        # 计算距离
        distance = np.linalg.norm(target_pos - self.current_pos)
        
        if distance < 0.001:  # 距离太小，直接返回成功
            return True
        
        # 计算运动时间参数
        t_acc = velocity / self.max_acceleration  # 加速时间
        s_acc = 0.5 * self.max_acceleration * t_acc**2  # 加速段距离
        
        if distance <= 2 * s_acc:  # 三角形速度规划
            t_total = 2 * math.sqrt(distance / self.max_acceleration)
            # 生成梯形速度规划的点
            points = self.generate_trapezoidal_profile(
                self.current_pos, target_pos, 
                distance, velocity, t_total
            )
        else:  # 梯形速度规划
            t_const = (distance - 2 * s_acc) / velocity
            t_total = 2 * t_acc + t_const
            points = self.generate_trapezoidal_profile(
                self.current_pos, target_pos, 
                distance, velocity, t_total
            )
        
        # 执行插补点
        for point in points:
            if self.stop_requested:
                return False
            
            success = self.execute_single_point(point)
            if not success:
                return False
            
            time.sleep(self.sample_time)
        
        # 更新当前位置
        self.current_pos = target_pos
        return True
    
    def generate_trapezoidal_profile(self, start, end, distance, max_vel, total_time):
        """生成梯形速度规划的插补点"""
        num_points = int(total_time / self.sample_time) + 1
        points = []
        
        for i in range(num_points):
            t = i * self.sample_time
            # 线性插值
            if distance > 0:
                ratio = t / total_time
                point = start + ratio * (end - start)
                points.append(point)
        
        return points
    
    def execute_single_point(self, cartesian_point):
        """执行单个笛卡尔空间点"""
        # 调用逆运动学服务
        ik_request = Targetpoint.Request()
        ik_request.x = float(cartesian_point[0])
        ik_request.y = float(cartesian_point[1])
        ik_request.z = float(cartesian_point[2])
        
        # 设置默认姿态（可根据需要扩展）
        ik_request.roll = 0.0
        ik_request.pitch = 0.0
        ik_request.yaw = 0.0
        
        try:
            future = self.ik_client.call_async(ik_request)
            
            # 等待响应
            start_time = time.time()
            while rclpy.ok():
                if future.done():
                    break
                if time.time() - start_time > 5.0:  # 5秒超时
                    self.get_logger().error("逆运动学服务调用超时")
                    return False
                time.sleep(0.01)
            
            if future.result() is not None:
                ik_response = future.result()
                
                if ik_response.success:
                    # 发布关节角度
                    joint_msg = Jointangle()
                    joint_msg.motor_1 = float(ik_response.joint_angles[0])
                    joint_msg.motor_2 = float(ik_response.joint_angles[1])
                    joint_msg.motor_3 = float(ik_response.joint_angles[2])
                    # 可以根据需要添加更多关节
                    
                    self.joint_pub.publish(joint_msg)
                    return True
                else:
                    self.get_logger().error(f"逆运动学求解失败: {ik_response.message}")
                    return False
            else:
                self.get_logger().error("逆运动学服务无响应")
                return False
                
        except Exception as e:
            self.get_logger().error(f"调用逆运动学服务异常: {str(e)}")
            return False
    
    def publish_status(self, status):
        """发布状态信息"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
    
    def stop_trajectory(self):
        """停止轨迹执行"""
        self.stop_requested = True
        self.get_logger().info("轨迹停止请求已发送")


def main():
    rclpy.init()
    
    try:
        node = GCodeTrajectoryNode()
        
        # 使用多线程执行器
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        
        node.get_logger().info("G代码轨迹规划节点运行中...")
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info("收到键盘中断信号")
        finally:
            # 停止所有运动
            node.stop_trajectory()
            executor.shutdown()
            node.destroy_node()
            
    except Exception as e:
        print(f"节点启动失败: {str(e)}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()