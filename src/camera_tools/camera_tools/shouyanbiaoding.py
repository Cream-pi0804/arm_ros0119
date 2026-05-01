import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import time

# --- 导入你的自定义服务 ---
from arm_interfaces.srv import Targetpoint

class FinalEyeToHandCalibrator(Node):
    def __init__(self):
        super().__init__('final_eye_to_hand_calibrator')
        
        # --- 1. 相机与 ArUco 配置 ---
        self.camera_matrix = np.array([[1443.903630, 0.0, 1295.131657], 
                                       [0.0, 1448.206356, 971.550376], 
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.marker_length = 0.05  
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # --- 2. 核心数据缓存 ---
        self.samples_dh_robot = []  
        self.samples_marker = []    
        self.latest_dh_matrix = None
        self.latest_marker_matrix = None

        # --- 3. ROS 通信接口 ---
        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(Image, '/raw_image', self.image_callback, 1)
        self.sub_dh_pose = self.create_subscription(Pose, '/dh_robot_pose', self.dh_pose_callback, 10)

        # 【新增】：创建对你的 IK 服务的客户端
        self.ik_client = self.create_client(Targetpoint, 'calculate_ik')

        cv2.namedWindow("Eye-to-Hand Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Eye-to-Hand Calibration", 960, 540)

        # --- 4. 启动自动化标定线程 ---
        self.get_logger().info("=========================================")
        self.get_logger().info("【眼在手外】自动化标定流水线就绪")
        self.get_logger().info("将在 3 秒后自动开始执行点位...")
        self.get_logger().info("=========================================")
        self.calib_thread = threading.Thread(target=self.automated_calibration_pipeline)
        self.calib_thread.start()

    def generate_grid_points(self):
        """生成纯 XYZ 平移网格"""
        target_poses = []
        grid_offsets = [-0.1, -0.05, 0.0, 0.05, 0.1]
        
        base_x = 0.6
        base_y = 0.0
        base_z = 0.6

        for dx in grid_offsets:
            for dy in grid_offsets:
                T = np.eye(4)
                T[0, 3] = base_x + dx
                T[1, 3] = base_y + dy
                T[2, 3] = base_z 
                target_poses.append(T)
                
        return target_poses

    def move_robot_via_ik(self, target_matrix):
        """调用你的 IK 服务控制机械臂移动"""
        # 1. 检查服务是否在线
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 IK 服务 "calculate_ik" 上线...')

        # 2. 构造请求
        t = target_matrix[:3, 3]
        self.get_logger().info(f"-> 下发 IK 目标点: X={t[0]:.3f}, Y={t[1]:.3f}, Z={t[2]:.3f}")
        
        req = Targetpoint.Request()
        req.target_x = float(t[0])
        req.target_y = float(t[1])
        req.target_z = float(t[2])

        # 3. 异步调用并等待完成 (由于在独立线程，这种等法绝对安全且不阻塞ROS)
        future = self.ik_client.call_async(req)
        while rclpy.ok() and not future.done():
            time.sleep(0.1)

        # 4. 解析结果
        try:
            res = future.result()
            # 假设你的 service 返回 0 表示成功，1 表示失败 (请根据实际 Targetpoint 结构修改判断)
            if hasattr(res, 'result'):
                # 简单判断只要不是明显的 FAIL 常量即可
                self.get_logger().info(f"机械臂反馈: {res.message}")
                return True
            else:
                return True # 备选返回
        except Exception as e:
            self.get_logger().error(f"调用 IK 服务异常: {e}")
            return False

    def automated_calibration_pipeline(self):
        time.sleep(3.0) 
        waypoints = self.generate_grid_points()
        total_pts = len(waypoints)
        
        for idx, target in enumerate(waypoints):
            self.get_logger().info(f"\n--- 进度: {idx+1}/{total_pts} ---")
            
            # 1. 移动机械臂
            if not self.move_robot_via_ik(target):
                self.get_logger().error("移动失败，放弃该点！")
                continue
                
            # 2. 等待机械臂停稳，消除抖动
            time.sleep(5.0) 
            
            # 3. 抓取此刻的真实数据
            dh_mat = self.latest_dh_matrix
            marker_mat = self.latest_marker_matrix
            
            if dh_mat is None:
                self.get_logger().warning("未收到 DH 位姿，跳过！")
                continue
            if marker_mat is None:
                self.get_logger().warning("未识别到 ArUco 码，跳过！")
                continue
                
            self.samples_dh_robot.append(dh_mat.copy())
            self.samples_marker.append(marker_mat.copy())
            self.get_logger().info(f"✅ 数据采集成功 (有效组数: {len(self.samples_dh_robot)})")
            
        self.get_logger().info("\n=== 轨迹执行完毕，开始计算眼在手外矩阵 ===")
        self.calculate_calibration()

    def dh_pose_callback(self, msg):
        T = np.eye(4)
        T[0, 3] = msg.position.x
        T[1, 3] = msg.position.y
        T[2, 3] = msg.position.z
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        T[:3, :3] = R.from_quat(q).as_matrix()
        self.latest_dh_matrix = T

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        self.latest_marker_matrix = None 

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[0], tvecs[0], 0.05)

            R_cam, _ = cv2.Rodrigues(rvecs[0])
            T_marker = np.eye(4)
            T_marker[:3, :3] = R_cam
            T_marker[:3, 3] = tvecs[0].reshape(3)
            self.latest_marker_matrix = T_marker
            
            cv2.putText(frame, "Tracking...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.putText(frame, f"Data points: {len(self.samples_dh_robot)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        preview = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Eye-to-Hand Calibration", preview)
        cv2.waitKey(1)

    def calculate_calibration(self):
        n = len(self.samples_dh_robot)
        if n < 3:
            self.get_logger().error("数据不足，标定失败！")
            return

        A_motions = []
        B_motions = []
        
        for i in range(1, n):
            A = np.dot(self.samples_dh_robot[i], np.linalg.inv(self.samples_dh_robot[i-1]))
            B = np.dot(self.samples_marker[i], np.linalg.inv(self.samples_marker[i-1]))
            A_motions.append(A)
            B_motions.append(B)

        try:
            M = np.zeros((3, 3))
            for A, B in zip(A_motions, B_motions):
                alpha = R.from_matrix(A[:3, :3]).as_rotvec()
                beta = R.from_matrix(B[:3, :3]).as_rotvec()
                M += np.outer(beta, alpha)

            U, S, Vh = np.linalg.svd(M)
            R_x = np.dot(Vh.T, U.T)
            
            if np.linalg.det(R_x) < 0:
                Vh[2, :] *= -1
                R_x = np.dot(Vh.T, U.T)

            C_stack = []
            d_stack = []
            I3 = np.eye(3)
            
            for A, B in zip(A_motions, B_motions):
                R_a = A[:3, :3]
                t_a = A[:3, 3]
                t_b = B[:3, 3]
                
                C_stack.append(R_a - I3)
                d_stack.append(np.dot(R_x, t_b) - t_a)

            C = np.vstack(C_stack)
            d = np.concatenate(d_stack)
            
            t_x, residuals, rank, s = np.linalg.lstsq(C, d, rcond=None)

            X = np.eye(4)
            X[:3, :3] = R_x
            X[:3, 3] = t_x

            self.get_logger().info("\n=========================================")
            self.get_logger().info("🎉 计算成功！")
            self.get_logger().info("转换矩阵 (机械臂基座 -> 相机):")
            self.get_logger().info(f"\n{np.array2string(X, formatter={'float_kind':lambda x: '%.6f' % x})}")
            self.get_logger().info("=========================================")

        except Exception as e:
            self.get_logger().error(f"计算发生异常: {e}")

def main():
    rclpy.init()
    node = FinalEyeToHandCalibrator()
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