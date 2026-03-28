import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        
        # --- 1. 相机内参配置 (关键：决定了测距的准确度) ---
        # 这些参数必须从你之前的 camera_calibration 结果中获取
        # K 是相机矩阵: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        # fx, fy 是焦距（像素单位）；cx, cy 是光学中心（主点）
        self.camera_matrix = np.array([[1443.903630, 1.202746, 1295.131657], 
                                       [0.0, 1448.206356, 971.550376], 
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        
        # D 是畸变系数: [k1, k2, p1, p2, k3]
        # 如果标定结果显示畸变很小，可以暂时全设为 0
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # --- 2. ArUco 字典与检测参数 ---
        # 选择字典：DICT_4X4_50 表示标记是 4x4 网格，共有 50 种不同的 ID
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
        # 默认检测参数（包含阈值处理、轮廓过滤等）
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # 标记的实际物理尺寸 (单位：米)
        # !!! 非常重要：如果你打印出的二维码边长是 5厘米，这里必须写 0.05
        # 这个值直接决定了 Z 轴（距离）的计算结果
        self.marker_length = 0.10  

        # --- 3. ROS 通信接口 ---
        self.bridge = CvBridge()
        # 订阅你摄像头发布的原始图像话题
        self.subscription = self.create_subscription(
            Image, '/raw_image', self.image_callback, 10)
        
        # 发布识别到的位姿 (Pose 类型包含 Position 和 Orientation)
        self.pose_pub = self.create_publisher(Pose, '/aruco_pose', 10)

        self.get_logger().info("ArUco 识别节点已启动，正在监听 /raw_image...")

    def image_callback(self, msg):
        """图像处理回调函数"""
        try:
            # 将 ROS 图像消息转换为 OpenCV 能够处理的 BGR 矩阵
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return

        # 转为灰度图能提高检测速度和鲁棒性
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 4. 检测图像中的 ArUco 标记 ---
        # corners: 包含每个检测到的标记的四个角点像素坐标
        # ids: 每个标记对应的 ID 编号
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        # 如果检测到了至少一个标记
        if ids is not None:
            # --- 5. 估计每个标记的位姿 (PnP 算法) ---
            # rvecs: 旋转向量 (Rotation Vector) - 相对于相机的旋转
            # tvecs: 平移向量 (Translation Vector) - 相对于相机的坐标 (x, y, z)
            # 注意：estimatePoseSingleMarkers 在新版 OpenCV 中可能有变化，这是最通用的写法
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

            for i in range(len(ids)):
                # 绘制标记的绿色边框
                cv2.aruco.drawDetectedMarkers(frame, corners)
                
                # 在标记中心绘制 3D 坐标轴 (红:X, 绿:Y, 蓝:Z)
                # 最后一个参数 0.03 是坐标轴绘制的长度（米）
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.03)

                # --- 6. 构建并发布 ROS Pose 消息 ---
                pose_msg = Pose()
                # tvecs[i][0] 分别对应:
                # x: 水平偏移 (左负右正)
                # y: 垂直偏移 (上负下正)
                # z: 距离相机的深度 (单位：米)
                pose_msg.position.x = float(tvecs[i][0][0])
                pose_msg.position.y = float(tvecs[i][0][1])
                pose_msg.position.z = float(tvecs[i][0][2])

                # 提示：rvec 是旋转向量，若要得到四元数(Quaternion)供导航使用，
                # 需要调用 cv2.Rodrigues() 将其转为旋转矩阵，再转为四元数。
                # 此处为了简单，只发布位置信息。
                
                self.pose_pub.publish(pose_msg)
                
                # 在终端实时打印距离信息
                self.get_logger().info(f"检测到 ID: {ids[i][0]}, 距离相机: {pose_msg.position.z:.3f} 米")

        # --- 7. 本地可视化窗口 ---
        frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)
        cv2.imshow("ArUco Monitor", frame)
        # 等待 1ms 以刷新窗口，按下任意键继续
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点正在关闭...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()