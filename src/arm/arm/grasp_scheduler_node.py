import rclpy
from rclpy.node import Node
import time  # 告别 asyncio，回归最简单的 time.sleep

# 引入多线程和回调组
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# 引入消息和服务定义
from geometry_msgs.msg import Pose          
from std_msgs.msg import Int32               
from arm_interfaces.srv import Targetpoint   
from arm_interfaces.msg import End  

class GraspSchedulerNode(Node):
    def __init__(self):
        super().__init__('grasp_scheduler_node')
        
        self.is_busy = False  
        self.place_position = {'x': 0.8, 'y': -0.3, 'z': 0.3}

        # --- 核心：重入回调组 ---
        self.callback_group = ReentrantCallbackGroup()

        self.target_sub = self.create_subscription(
            Pose,
            '/vision/target_Pose',
            self.vision_callback,
            10,
            callback_group=self.callback_group
        )
        
        self.ik_client = self.create_client(
            Targetpoint, 
            'calculate_ik',
            callback_group=self.callback_group
        )
        
        self.gripper_pub = self.create_publisher(End, '/arm/endd', 10)
        self.get_logger().info('✅ 视觉抓取调度节点已启动 (多线程同步版) ...')

    def vision_callback(self, msg):
        """去掉了 async，变成最普通的同步函数"""
        if self.is_busy:
            self.get_logger().warn('当前正在执行抓取任务，忽略新到达的目标！')
            return
            
        self.is_busy = True
        self.get_logger().info('\n=====================================')
        self.get_logger().info(f'🎯 接收到新视觉目标: x={msg.position.x:.3f}, y={msg.position.y:.3f}, z={msg.position.z:.3f}')

        try:
            # --- 步骤 1: 准备点 ---
            self.get_logger().info('步骤 1/5: 发送指令移动至准备点...')
            success = self.call_ik_service(msg.position.x+0.0, msg.position.y-0.06, msg.position.z - 0.15)
            if not success: return
            self.get_logger().info('--> 等待机械臂物理移动至准备点 (3秒)...')
            time.sleep(6.0)  # 在多线程下，直接 sleep 是绝对安全的！

            # --- 步骤 2: 下降 ---
            self.get_logger().info('步骤 2/5: 发送指令下降至抓取点...')
            success = self.call_ik_service(msg.position.x+0.1, msg.position.y-0.06, msg.position.z)
            if not success: return
            self.get_logger().info('--> 等待机械臂平稳下降 (2秒)...')
            time.sleep(5.0)

            # --- 步骤 3: 抓取 ---
            self.get_logger().info('步骤 3/5: 到达目标点，闭合爪子执行抓取...')
            self.control_gripper(-1800.0,500)  
            time.sleep(10.0)
            
            self.get_logger().info('步骤 3.5: 发送指令提起机械臂...')
            success = self.call_ik_service(msg.position.x, msg.position.y-0.06, msg.position.z - 0.25)
            if not success: return
            self.get_logger().info('--> 等待机械臂提起 (2秒)...')
            time.sleep(2.0)

            # --- 步骤 4: 放置点 ---
            self.get_logger().info(f"步骤 4/5: 发送指令移动至放置点: {self.place_position}")
            success = self.call_ik_service(self.place_position['x']-0.1, self.place_position['y'], self.place_position['z']+0.05)
            if not success: return
            self.get_logger().info('--> 等待机械臂长距离移动至放置点 (4秒)...')
            time.sleep(4.0)

            # --- 步骤 5: 松开 ---
            self.get_logger().info('步骤 5/5: 到达放置点，松开爪子...')
            self.control_gripper(1500.0, 1000)  
            time.sleep(9.0)
            success = self.call_ik_service(self.place_position['x']-0.1, self.place_position['y'], self.place_position['z']-0.1)
            if not success: return
            self.get_logger().info('--> 等待机械臂长距离移动至放置点 (4秒)...')
            time.sleep(4.0)
            success = self.call_ik_service(0.7, 0, 0.1)
            if not success: return
            self.get_logger().info('--> 等待机械臂长距离移动至放置点 (4秒)...')
            time.sleep(4.0)
            self.get_logger().info('🎉 >>> 抓取放置流水线任务圆满完成！ <<<')

        except Exception as e:
            self.get_logger().error(f'🚨 执行任务期间发生异常: {e}')
            
        finally:
            self.is_busy = False
            self.get_logger().info('等待下一个视觉目标...\n=====================================')

    def call_ik_service(self, x, y, z):
        """去掉了 async，使用最稳定的直接阻塞调用"""
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 IK 服务上线...')
            
        req = Targetpoint.Request()
        req.target_x = float(x)
        req.target_y = float(y)
        req.target_z = float(z)
        
        # 直接使用 .call() 同步请求！
        # 因为在 MultiThreadedExecutor 中，当前线程被挂起时，别的线程会去接收 Response
        response = self.ik_client.call(req) 
        
        if response.result == Targetpoint.Response.SUCCESS: 
            return True
        else:
            self.get_logger().error(f'IK 服务反馈错误: {response.message}')
            return False

    def control_gripper(self, action_code, action_time):
        msg = End()
        msg.end_vaule = action_code 
        msg.reach_time = action_time
        self.gripper_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GraspSchedulerNode()
    
    # 必须使用多线程执行器
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