import rclpy
from rclpy.node import Node
import asyncio

# 引入消息和服务定义
from geometry_msgs.msg import Pose          # 假设视觉发送的是 Point 消息
from std_msgs.msg import Int32               # 假设舵机使用 Int32 (1为抓取, 0为松开)
from arm_interfaces.srv import Targetpoint   # 你自定义的 IK 服务
from arm_interfaces.msg import End  
class GraspSchedulerNode(Node):
    def __init__(self):
        super().__init__('grasp_scheduler_node')
        
        # --- 1. 状态标志 ---
        self.is_busy = False  # 防止任务重叠（正在抓取时忽略新的视觉目标）
        
        # --- 2. 放置点坐标配置 (请根据实际物理空间修改) ---
        self.place_position = {'x': 0.2, 'y': 0.0, 'z': 0.15}

        # --- 3. 创建 ROS 2 通信接口 ---
        
        # A. 订阅视觉传来的目标位置 (假设话题名为 /vision/target_point)
        self.target_sub = self.create_subscription(
            Pose,
            '/vision/target_Pose',
            self.vision_callback,
            10
        )
        
        # B. 创建 IK 服务客户端 (连接到你上面写的 IKServiceNode)
        self.ik_client = self.create_client(Targetpoint, 'calculate_ik')
        
        # C. 创建舵机控制发布者 (假设话题名为 /arm/gripper)
        self.gripper_pub = self.create_publisher(End, '/arm/endd', 10)
        
        self.get_logger().info('视觉抓取调度节点已启动，等待视觉目标...')

    async def vision_callback(self, msg):
        """
        接收到视觉目标位置后的异步回调函数
        使用 async def 可以让我们在内部使用 await 挂起等待服务结果，而不会卡死 ROS 2
        """
        if self.is_busy:
            self.get_logger().warn('当前正在执行抓取任务，忽略新到达的目标！')
            return
            
        self.is_busy = True
        self.get_logger().info('=====================================')
        self.get_logger().info(f'接收到新视觉目标: x={msg.position.x:.3f}, y={msg.position.y:.3f}, z={msg.position.z:.3f}')

        try:
            # ----------------------------------------
            # 步骤 1: 调用 IK 服务前往【视觉目标点】
            # ----------------------------------------
            self.get_logger().info('步骤 1: 正在移动至抓取目标点...')
            success = await self.call_ik_service(msg.position.x, msg.position.y, msg.position.z)
            if not success:
                self.get_logger().error('无法到达目标点，抓取任务中止。')
                return
            
            # ----------------------------------------
            # 步骤 2: 执行舵机闭合 (抓取)
            # ----------------------------------------
            self.get_logger().info('步骤 2: 到达目标点，执行抓取...')
            self.control_gripper(1,5000)  # 1 表示闭合抓取 (根据你的舵机逻辑修改)
            # 使用 asyncio.sleep 代替 time.sleep，不会阻塞节点其他回调
            await asyncio.sleep(1.5) 

            # ----------------------------------------
            # 步骤 3: 调用 IK 服务前往【放置点】
            # ----------------------------------------
            self.get_logger().info(f"步骤 3: 抓取成功，移动至放置点: {self.place_position}")
            success = await self.call_ik_service(
                self.place_position['x'], 
                self.place_position['y'], 
                self.place_position['z']
            )
            if not success:
                self.get_logger().error('无法到达放置点，任务中止。')
                return

            # ----------------------------------------
            # 步骤 4: 执行舵机张开 (松开)
            # ----------------------------------------
            self.get_logger().info('步骤 4: 到达放置点，松开爪子...')
            self.control_gripper(0,5000)  # 0 表示松开张开
            await asyncio.sleep(1.0) 
            
            self.get_logger().info('>>> 抓取放置流水线任务圆满完成！ <<<')

        except Exception as e:
            self.get_logger().error(f'执行任务期间发生异常: {e}')
            
        finally:
            # 无论任务成功还是失败，最后必须释放忙碌锁，允许接受下一次抓取任务
            self.is_busy = False

    async def call_ik_service(self, x, y, z):
        """调用 IK 逆解服务的异步辅助函数"""
        # 阻塞等待服务上线
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('IK 服务不可用，等待中...')
            
        req = Targetpoint.Request()
        req.target_x = float(x)
        req.target_y = float(y)
        req.target_z = float(z)
        
        # call_async 返回一个 Future，await 可以将其挂起直至服务返回结果
        future = self.ik_client.call_async(req)
        response = await future 
        
        # 检查你的 Targetpoint.srv 返回的结果标志位
        if response.result == Targetpoint.Response.SUCCESS: 
            return True
        else:
            self.get_logger().error(f'IK 服务反馈错误信息: {response.message}')
            return False

    def control_gripper(self, action_code,action_time):
        """控制舵机张合的辅助函数"""
        msg = End()
        msg.end_vaule = action_code
        msg.reach_time = action_time
        self.gripper_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GraspSchedulerNode()
    
    # 注意：因为我们使用了 async/await 异步回调机制，
    # 简单的默认单线程执行器 (spin) 就足以完美运行流线型任务，不会死锁。
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()