import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraProvider(Node):
    def __init__(self):
        super().__init__('camera_provider')
        self.publisher_ = self.create_publisher(Image, 'raw_image', 10)
        self.cap = cv2.VideoCapture("rtsp://192.168.1.168:554/stream_0") # 这里的0可以换成你的网络摄像头URL
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.033, self.timer_callback) # 约30fps

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            self.publisher_.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))

def main():
    rclpy.init()
    node = CameraProvider()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()