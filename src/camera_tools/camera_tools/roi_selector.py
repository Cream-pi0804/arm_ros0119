import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class TargetExtractor(Node):
    def __init__(self):
        super().__init__('target_extractor')
        self.bridge = CvBridge()
        cv2.namedWindow("Monitor", cv2.WINDOW_NORMAL)
        # 订阅原始监控画面
        self.subscription = self.create_subscription(
            Image, 'raw_image', self.image_callback, 10)
        
        # 发布框选后的单次目标图
        self.publisher_ = self.create_publisher(Image, 'target_crop', 10)
        
        self.current_frame = None
        self.get_logger().info("监控中... [按 's' 键开始框选目标, 按 'q' 退出]")

    def image_callback(self, msg):
        # 转换并存储当前帧
        # self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # 1. 转换原始帧（保留完整分辨率用于裁剪）
        self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # 2. 创建一个缩小的副本用于预览
        # fx, fy 是缩放因子，0.5 表示缩小一半
        preview_frame = cv2.resize(self.current_frame, (0, 0), fx=0.35, fy=0.35)
        
        # 3. 显示缩小后的画面
        cv2.imshow("Monitor (Scaled)", preview_frame)
        # # 在监控窗口显示实时画面
        # cv2.imshow("Monitor", self.current_frame)
        
        # 等待按键事件 (1ms)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'): # 发现目标，按下 's'
            self.extract_target()
        elif key == ord('q'):
            rclpy.shutdown()

    def extract_target(self):
        if self.current_frame is None:
            return

        self.get_logger().info("暂停监控，请框选目标...")
            # 1. 先创建一个名为 "Select Target" 的窗口
        # WINDOW_NORMAL 允许我们手动或通过代码调整窗口大小
        cv2.namedWindow("Select Target", cv2.WINDOW_NORMAL)

        # 2. 设定你想要的窗口显示尺寸 (宽, 高)
        # 你可以根据需要调整这个数值，比如 800, 600
        cv2.resizeWindow("Select Target", 960, 540) 

        # 3. 在这个已经存在的窗口中进行框选
        # 注意：第一个参数 窗口名 必须和上面创建的一模一样
        # 弹窗进行框选 (会阻塞 image_callback 直到按下空格或回车)
        roi = cv2.selectROI("Select Target", self.current_frame, fromCenter=False, showCrosshair=True)
        
        x, y, w, h = roi
        if w > 0 and h > 0:
            # 裁剪图像
            crop_img = self.current_frame[y:y+h, x:x+w]
            
            # 发布裁剪后的图像
            crop_msg = self.bridge.cv2_to_imgmsg(crop_img, 'bgr8')
            # 保持时间戳同步
            crop_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(crop_msg)
            cv2.imshow("Target", crop_img)
            self.get_logger().info(f"目标已发布！尺寸: {w}x{h}")
        
        # 关闭框选窗口，回到 Monitor 窗口
        cv2.destroyWindow("Select Target")
        self.get_logger().info("继续监控...")

def main():
    rclpy.init()
    node = TargetExtractor()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()