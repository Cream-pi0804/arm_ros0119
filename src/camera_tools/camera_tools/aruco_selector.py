import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
# 新增：引入 scipy 的 Rotation 模块，专门用于处理旋转矩阵、欧拉角和四元数之间的转换
from scipy.spatial.transform import Rotation as R

class ArucoTargetSelector(Node):
    """
    ArUco 目标选择与抓取节点 (包含完整的 6D 位姿与旋转姿态转换)
    """
    def __init__(self):
        super().__init__('aruco_target_selector')
        
        # ==========================================
        # 1. 相机内参配置
        # ==========================================
        self.camera_matrix = np.array([[1443.903630, 0.0, 1295.131657], 
                                       [0.0, 1448.206356, 971.550376], 
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # ==========================================
        # 2. ArUco 物理尺寸与字典配置
        # ==========================================
        self.marker_length = 0.05  
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # ==========================================
        # 3. 手眼标定矩阵 (Hand-Eye Calibration Matrix)
        # ==========================================
        # 这是一个 4x4 的矩阵，左上角 3x3 是旋转，右上角 3x1 是平移
        self.T_base_camera = np.array([



[-0.236196,0.937533 ,0.255426 ,-0.099227],
 [-0.189322, 0.213425, -0.958440 ,0.670866],
 [-0.953084 ,-0.274738, 0.127086 ,0.470987],
 [0.000000,0.000000 ,0.000000 ,1.000000]

            
        ], dtype=np.float32)

        # ==========================================
        # 4. ROS 通信接口配置
        # ==========================================
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/raw_image', self.image_callback, 1)
        self.target_pose_pub = self.create_publisher(Pose, '/vision/target_Pose', 10)
        self.annotated_img_pub = self.create_publisher(Image, '/processed_image', 10)
        
        self.current_frame = None
        self.detected_markers = [] 
        
        self.get_logger().info("系统就绪！[按 's' 暂停并选定目标, 按 'q' 退出]")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return

        self.current_frame = frame.copy()
        self.detected_markers.clear() 
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            # 估算姿态：不仅返回平移 (tvecs)，还返回旋转向量 (rvecs)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            for i in range(len(ids)):
                c = corners[i][0]
                u = int(np.mean(c[:, 0])) 
                v = int(np.mean(c[:, 1])) 
                
                # --- 新增：把旋转向量(rvec)也存起来 ---
                self.detected_markers.append({
                    'id': ids[i][0],      
                    'pixel_u': u,         
                    'pixel_v': v,         
                    'tvec': tvecs[i][0],  # 平移向量
                    'rvec': rvecs[i][0]   # 旋转向量
                })
                
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.05)
                cv2.putText(frame, f"ID:{ids[i][0]}", (u - 20, v - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        preview_frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)
        cv2.imshow("Monitor", preview_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'): 
            if len(self.detected_markers) == 0:
                self.get_logger().warning("当前画面未检测到任何目标，无法进行框选！")
            else:
                self.select_and_publish_target()
        elif key == ord('q'):
            rclpy.shutdown()

    def select_and_publish_target(self):
        display_frame = self.current_frame.copy()
        
        win_name = "Select Target to Grasp"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720) 

        self.get_logger().info("已暂停。请框住你要抓取的那个目标，按 Enter/Space 确认。")
        roi = cv2.selectROI(win_name, display_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win_name)

        x, y, w, h = roi
        if w == 0 or h == 0:
            self.get_logger().info("取消框选，恢复监控。")
            return

        selected_target = None
        for target in self.detected_markers:
            u, v = target['pixel_u'], target['pixel_v']
            if x < u < x + w and y < v < y + h:
                selected_target = target
                break 

        if selected_target is None:
            self.get_logger().warning("框选区域内没有检测到有效目标！请重新框选。")
            return

        # ==========================================
        # ★★★ 核心：提取目标坐标并包含旋转姿态的转换 ★★★
        # ==========================================
        target_id = selected_target['id']
        tvec = selected_target['tvec'] # 相机系下的平移
        rvec = selected_target['rvec'] # 相机系下的旋转向量 (1x3)

        # 1. 将旋转向量 (rvec) 转换为 3x3 旋转矩阵
        # cv2.Rodrigues 是专门将角轴表示的旋转向量转为矩阵的函数
        R_cam, _ = cv2.Rodrigues(rvec)

        # 2. 构建目标在相机系下的 4x4 齐次变换矩阵 (T_camera_target)
        T_camera_target = np.eye(4)
        T_camera_target[:3, :3] = R_cam               # 左上角放入旋转矩阵
        T_camera_target[:3, 3] = tvec.reshape(3)      # 右上角放入平移向量

        # 3. 进行刚体变换计算
        # 目标在机械臂基座系下的矩阵 = 手眼矩阵 × 目标在相机系下的矩阵
        T_base_target = np.dot(self.T_base_camera, T_camera_target)

        # 4. 从结果矩阵中剥离出新的平移坐标 (X, Y, Z)
        rx, ry, rz = float(T_base_target[0, 3]), float(T_base_target[1, 3]), float(T_base_target[2, 3])
        rx1, ry1, rz1 = float(T_camera_target[0, 3]), float(T_camera_target[1, 3]), float(T_camera_target[2, 3])

        # 5. 从结果矩阵中剥离出新的旋转矩阵，并转成 ROS 需要的四元数 (Quaternion)
        R_base = T_base_target[:3, :3]
        rotation_obj = R.from_matrix(R_base)
        qx, qy, qz, qw = rotation_obj.as_quat() # 格式为 [x, y, z, w]

        # (可选) 为了让人类在终端能看懂当前的姿态，转成欧拉角 (Roll, Pitch, Yaw)
        euler_angles = rotation_obj.as_euler('xyz', degrees=True)
        roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]

        # ==========================================
        # 填充完整的 Pose 消息并发布
        # ==========================================
        target_pose = Pose()
        # 位置
        target_pose.position.x = rx
        target_pose.position.y = ry
        target_pose.position.z = rz
        # 姿态 (由于经历了手眼矩阵相乘，现在的姿态已经是相对机械臂基座的了)
        target_pose.orientation.x = float(qx)
        target_pose.orientation.y = float(qy)
        target_pose.orientation.z = float(qz)
        target_pose.orientation.w = float(qw)
        
        self.target_pose_pub.publish(target_pose)
        
        # 打印详细结果，包含欧拉角姿态以便调试
        self.get_logger().info(f"★★★ 成功锁定 ID: {target_id} ★★★")
        self.get_logger().info(f"📍 位置(米): X:{rx:.3f}, Y:{ry:.3f}, Z:{rz:.3f}")
        self.get_logger().info(f"📍 位置(米): X1:{rx1:.3f}, Y1:{ry1:.3f}, Z1:{rz1:.3f}")
        self.get_logger().info(f"🔄 姿态(度): Roll:{roll:.1f}, Pitch:{pitch:.1f}, Yaw:{yaw:.1f}")
        self.get_logger().info(f"📐 四元数: x:{qx:.3f}, y:{qy:.3f}, z:{qz:.3f}, w:{qw:.3f}")

        # ==========================================
        # 可视化图像输出
        # ==========================================
        cv2.rectangle(display_frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 4)
        label = f"ID:{target_id} [X:{rx:.2f},Y:{ry:.2f},Z:{rz:.2f}]"
        cv2.putText(display_frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        current_time = self.get_clock().now().to_msg()
        processed_img_msg = self.bridge.cv2_to_imgmsg(display_frame, 'bgr8')
        processed_img_msg.header.stamp = current_time
        processed_img_msg.header.frame_id = "camera_link_optical" 
        self.annotated_img_pub.publish(processed_img_msg)

        feedback_frame = cv2.resize(display_frame, (0, 0), fx=0.35, fy=0.35)
        cv2.imshow("Result", feedback_frame)
        cv2.waitKey(2000) 
        cv2.destroyWindow("Result")

def main():
    rclpy.init()
    node = ArucoTargetSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()