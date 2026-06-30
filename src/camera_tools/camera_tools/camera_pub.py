import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time
import os

class CameraProviderMultiThread(Node):
    def __init__(self):
        super().__init__('camera_provider')
        # 参数配置（可以通过 ROS 2 参数服务器修改）
        self.declare_parameter('rtsp_url', "rtsp://192.168.1.168:554/stream_0")
        self.declare_parameter('publish_topic', 'raw_image')
        self.declare_parameter('publish_fps', 20.0)
        self.declare_parameter('reconnect_delay_s', 2.0)
        self.declare_parameter('read_timeout_s', 5.0)  # read() 超时阈值，超过此时间认为断连
        self.declare_parameter('max_read_retries', 10)  # 连续读取失败多少次才触发重连（容忍瞬时解码错误）

        # 获取参数
        self.url = self.get_parameter('rtsp_url').get_parameter_value().string_value
        self.publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value
        self.publish_fps = self.get_parameter('publish_fps').get_parameter_value().double_value
        self.reconnect_delay = self.get_parameter('reconnect_delay_s').get_parameter_value().double_value
        self.read_timeout = self.get_parameter('read_timeout_s').get_parameter_value().double_value
        self.max_read_retries = self.get_parameter('max_read_retries').get_parameter_value().integer_value

        # 设置 FFmpeg 环境变量，限制底层超时（减少 cap.read() 阻塞时间）
        # timeout 单位是微秒，默认约 30s，这里设为 read_timeout_s + 2 作为兜底
        # 添加 err_detect 相关选项让 FFmpeg 对轻微解码错误更宽容
        ffmpeg_timeout_us = int((self.read_timeout + 2) * 1_000_000)
        os.environ.setdefault(
            'OPENCV_FFMPEG_CAPTURE_OPTIONS',
            f'rtsp_transport;tcp|timeout;{ffmpeg_timeout_us}|max_delay;500000|flags;low_delay|fflags;nobuffer|err_detect;ignore_err'
        )

        # ROS 2 组件
        self.publisher_ = self.create_publisher(Image, self.publish_topic, 30)
        self.bridge = CvBridge()

        # 摄像头连接和线程共享变量
        self.cap = None
        self.frame = None           # 共享变量：存储最新的帧
        self.frame_lock = threading.Lock() # 锁：保护共享变量
        self.frame_timestamp = 0.0  # 记录最新帧的时间戳
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

                # 设置缓冲区为 1 以减少延迟
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not self.cap.isOpened():
                    self.get_logger().error(f"连接摄像头失败，{self.reconnect_delay}秒后重试...")
                    time.sleep(self.reconnect_delay)
                    continue
                else:
                    self.get_logger().info("RTSP 摄像头连接成功")

            # 2. 读取一帧（记录耗时，用于诊断超时）
            t0 = time.time()
            ret, frame = self.cap.read()
            elapsed = time.time() - t0

            if elapsed > self.read_timeout:
                self.get_logger().warn(
                    f"cap.read() 耗时 {elapsed:.1f}s，超过阈值 {self.read_timeout}s，"
                    f"可能网络不稳定或摄像头断连"
                )

            if not ret or frame is None:
                self.get_logger().warn("读取帧失败，准备重连...")
                self.cap.release()
                self.cap = None  # 下次循环会重连
                # 清除旧帧，避免发布端持续发送过期画面
                with self.frame_lock:
                    self.frame = None
                time.sleep(0.1)
                continue

            # 3. 将新帧写入共享变量（加锁保护）
            with self.frame_lock:
                self.frame = frame.copy()  # copy() 确保发布线程拿到的数据不会被这里意外修改
                self.frame_timestamp = t0

    def timer_callback(self):
        """主线程定时器：只负责从共享变量中拿取最新帧并发布。"""

        current_frame = None
        frame_age = float('inf')

        # 1. 获取最新帧（加锁）
        with self.frame_lock:
            if self.frame is not None:
                current_frame = self.frame
                frame_age = time.time() - self.frame_timestamp

        if current_frame is None:
            return  # 后台线程还没读到第一帧，或者连接已断开清空了帧

        # 2. 检查帧是否过期（如果超过 read_timeout 没更新，说明连接可能断了）
        if frame_age > self.read_timeout:
            self.get_logger().warn(
                f"最新帧已过时 {frame_age:.1f}s，可能断连，跳过发布"
            )
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
        self.is_running = False  # 信号：通知线程退出
        self.get_logger().info("正在停止摄像头读取线程...")

        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)  # 等待线程结束（稍微放长一点）

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # 清除共享帧
        with self.frame_lock:
            self.frame = None
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