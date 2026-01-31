#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
import serial
import threading
import numpy as np

# ROS2消息和服务
from arm_interfaces.msg import Jointangle, Pointnow
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter

class MotorSerial(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        
        # 1. 声明参数
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('send_interval', 0.1)
        self.declare_parameter('auto_reconnect', True)
        self.declare_parameter('reconnect_interval', 2.0)
        
        # 2. 初始化变量
        self.ser = None
        self.serial_lock = threading.RLock()
        self.running = True
        self.last_reconnect_time = 0
        
        # --- 新增：目标角度缓存与比对容差 ---
        self.target_angles = [0.0, 0.0, 0.0]  # 存储最近一次订阅到的目标
        self.angle_tolerance = 0.5            # 到达判定容差（度）
        
        # 3. 连接串口
        if not self.connect_serial():
            self.get_logger().error("串口连接失败，将尝试自动重连")
        
        # 4. 创建订阅与发布
        self.jointangle_subscriber = self.create_subscription(
            Jointangle, 
            '/arm/Jointangle', 
            self.send_serial_command, 
            10
        )   
        # 发布实际位置及到达标志
        self.joint_now_pub = self.create_publisher(Pointnow, '/arm/Pointnow', 10)
        
        # 5. 其他组件
        self.heartbeat_timer = self.create_timer(5.0, self.check_connection)
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # 6. 启动接收线程
        self.receive_thread = threading.Thread(
            target=self.read_serial_data_thread,
            daemon=True,
            name="ReceiveThread"
        )
        self.receive_thread.start()
        
        self.get_logger().info(f'{node_name} 初始化完成，等待目标角度...')

    def send_serial_command(self, msg):
        """发送串口命令并更新目标缓存"""
        # --- 核心修改：记录目标值 ---
        self.target_angles = [msg.motor_1, msg.motor_2, msg.motor_3]
        
        if not self.check_serial_connection():
            if not self.try_reconnect():
                return False
        
        try:
            # 格式化命令
            cmd = f"RPM:{msg.motor_1:.2f},{msg.motor_2:.2f},{msg.motor_3:.2f}\n"
            
            with self.serial_lock:
                if self.ser and self.ser.is_open:
                    self.ser.write(cmd.encode('utf-8'))
                    self.ser.flush()
                    return True
                else:
                    return False
        except Exception as e:
            self.get_logger().error(f'发送异常: {e}')
            return False

    def process_motor_data(self, data_str):
        """处理电机反馈并进行到达比对"""
        try:
            # 假设串口协议格式为 DATA:角1,角2,角3
            if data_str.startswith('DATA:'):
                values_str = data_str[5:].split(',')
                if len(values_str) >= 3:
                    curr_1 = float(values_str[0])
                    curr_2 = float(values_str[1])
                    curr_3 = float(values_str[2])
                    
                    # --- 核心修改：比较当前值与目标值 ---
                    diffs = [
                        abs(curr_1 - self.target_angles[0]),
                        abs(curr_2 - self.target_angles[1]),
                        abs(curr_3 - self.target_angles[2])
                    ]
                    
                    # 判断是否所有关节都进入容差范围
                    # 1为到达，0为未到达
                    is_reached = 1 if all(d < self.angle_tolerance for d in diffs) else 0
                    
                    # 封装消息发布
                    now_msg = Pointnow()
                    now_msg.x = curr_1
                    now_msg.y = curr_2
                    now_msg.z = curr_3
                    now_msg.is_reached = is_reached
                    
                    self.joint_now_pub.publish(now_msg)
                    
                    # 可视化反馈
                    if is_reached:
                        self.get_logger().debug("状态：已到达目标位置")
                    
        except Exception as e:
            self.get_logger().error(f'处理电机数据失败: {e}')

    def parse_serial_data(self, data):
        """解析串口原始数据"""
        try:
            data_str = data.decode('GBK', errors='ignore').strip()
            if not data_str:
                return

            if data_str.startswith('DATA:'):
                self.process_motor_data(data_str)
            elif data_str.startswith('ACK:'):
                self.get_logger().debug(f'收到确认: {data_str}')
            elif data_str.startswith('ERR:'):
                self.get_logger().error(f'收到电机错误: {data_str}')
            else:
                self.get_logger().info(f'串口透传: {data_str}')
                
        except Exception as e:
            self.get_logger().error(f'解析失败: {e}')

    # --- 以下串口底层逻辑保持不变 ---

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'port': self.get_logger().info(f"端口变更: {param.value}")
        return SetParametersResult(successful=True)

    def connect_serial(self):
        with self.serial_lock:
            port = self.get_parameter('port').value
            baud = self.get_parameter('baudrate').value
            try:
                self.disconnect_serial()
                self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.1)
                return self.ser.is_open
            except Exception as e:
                self.get_logger().error(f'连接异常: {e}')
                return False
    
    def disconnect_serial(self):
        if self.ser:
            try: self.ser.close()
            except: pass
            self.ser = None
    
    def check_serial_connection(self):
        with self.serial_lock:
            return self.ser is not None and self.ser.is_open
    
    def try_reconnect(self):
        if not self.get_parameter('auto_reconnect').value: return False
        if (time.time() - self.last_reconnect_time) < self.get_parameter('reconnect_interval').value:
            return False
        self.last_reconnect_time = time.time()
        return self.connect_serial()
    
    def check_connection(self):
        if not self.check_serial_connection():
            self.try_reconnect()

    def read_serial_data_thread(self):
        buffer = bytearray()
        while self.running and rclpy.ok():
            try:
                if not self.check_serial_connection():
                    time.sleep(1.0)
                    continue
                
                with self.serial_lock:
                    if self.ser and self.ser.in_waiting > 0:
                        buffer.extend(self.ser.read(self.ser.in_waiting))
                
                while b'\n' in buffer:
                    idx = buffer.index(b'\n')
                    packet = buffer[:idx]
                    buffer = buffer[idx+1:]
                    if packet:
                        self.parse_serial_data(packet)
                time.sleep(0.001)
            except Exception as e:
                self.get_logger().error(f'接收异常: {e}')
                time.sleep(1.0)

    def on_shutdown(self):
        self.running = False
        if self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
        self.disconnect_serial()

def main():
    rclpy.init()
    node = MotorSerial('motor_serial')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()