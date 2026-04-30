import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time

class CameraProviderMultiThread(Node):
    def __init__(self):
        super().__init__('camera_provider')
        # 参数配置（可以通过 ROS 2 参数服务器修改）
        self.declare_parameter('rtsp_url', "rtsp://192.168.1.168:554/stream_0")
        self.declare_parameter('publish_topic', 'raw_image')
        self.declare_parameter('publish_fps', 30.0)
        self.declare_parameter('reconnect_delay_s', 2.0)
        
        # 获取参数
        self.url = self.get_parameter('rtsp_url').get_parameter_value().string_value
        self.publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value
        self.publish_fps = self.get_parameter('publish_fps').get_parameter_value().double_value
        self.reconnect_delay = self.get_parameter('reconnect_delay_s').get_parameter_value().double_value
        
        # ROS 2 组件
        self.publisher_ = self.create_publisher(Image, self.publish_topic, 30)
        self.bridge = CvBridge()
        
        # 摄像头连接和线程共享变量
        self.cap = None
        self.frame = None           # 共享变量：存储最新的帧
        self.frame_lock = threading.Lock() # 锁：保护共享变量
        self.is_running = True     # 标志：控制后台线程退出
        
        # 启动后台读取线程
        self.get_logger().info(f"正在建立 RTSP 连接: {self.url}...")
        self.capture_thread = threading.Thread(target=self._camera_capture_thread, daemon=True)
        self.capture_thread.start()
        
        # 定时器：负责发布视频，频率与后台读取解耦
        timer_period = 1.0 / self.publish_fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f"CameraProvider(多线程版) 已启动，发布频率: {self.publish_fps} FPS")

    def _camera_capture_thread(self):
        """后台线程：专职、不间断读取摄像头帧，确保 frame 永远是最新的。"""
        self.get_logger().info("摄像头读取线程已启动")
        
        while self.is_running:
            # 1. 检查并建立连接
            if self.cap is None or not self.cap.isOpened():
                if self.cap is not None:
                    self.cap.release()
                
                # 尝试连接，可以通过增加参数来尝试不同的 OpenCV 后端 (e.g., cv2.CAP_FFMPEG)
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                
                # 虽然多线程解决了堆积，但设置缓冲区为 1 仍是好习惯
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if not self.cap.isOpened():
                    self.get_logger().error(f"连接摄像头失败，{self.reconnect_delay}秒后重试...")
                    time.sleep(self.reconnect_delay)
                    continue
                else:
                    self.get_logger().info("RTSP 摄像头连接成功")
            
            # 2. 读取一帧
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                self.get_logger().warn("读取帧失败，准备重连...")
                self.cap.release()
                self.cap = None # 下次循环会重连
                # 适当等待，避免连续失败导致 CPU 占用过高
                time.sleep(0.1)
                continue
            
            # 3. 将新帧写入共享变量（加锁保护）
            with self.frame_lock:
                self.frame = frame.copy() # copy() 确保发布线程拿到的数据不会被这里意外修改

    def timer_callback(self):
        """主线程定时器：只负责从共享变量中拿取最新帧并发布。"""
        
        current_frame = None
        
        # 1. 获取最新帧（加锁）
        with self.frame_lock:
            if self.frame is not None:
                current_frame = self.frame # 指向，不需要 copy，因为后台线程写的时候会 copy
                # 发布后可以选择清除，防止发布旧帧（如果你更倾向于宁可不发也不发旧的）
                # self.frame = None
            
        if current_frame is None:
            # 如果后台线程还没读到第一帧，或者连接断开了
            return
        
        # 2. 发布图像
        try:
            # 自动添加 header (包括时间戳和 frame_id)
            img_msg = self.bridge.cv2_to_imgmsg(current_frame, 'bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera_link"
            self.publisher_.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"图像转换或发布失败: {e}")

    def destroy_node(self):
        """节点销毁时释放资源，并停止后台线程"""
        self.is_running = False # 信号：通知线程退出
        self.get_logger().info("正在停止摄像头读取线程...")
        
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0) # 等待线程结束
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraProviderMultiThread()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断，正在关闭...")
    finally:
        node.destroy_node()
        # 确保 shutdown 被调用
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()