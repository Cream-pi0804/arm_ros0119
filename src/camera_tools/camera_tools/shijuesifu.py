#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【视觉伺服抓取 (Visual Servoing) 节点】—— 增量 IK 服务的"上层大脑"。

整体闭环逻辑（Eye-to-Hand 手眼分离构型）：
    1. 相机持续检测画面里的 ArUco 码，估计每个码在相机系下的 3D 位置
    2. 用户鼠标点选目标码，锁定抓取对象
    3. 控制循环不断计算"夹爪码 → 目标码"的相对误差，转到机械臂基座坐标系
    4. 用比例控制器(P)取一小步，调用 calculate_incremental_ik 服务驱动机械臂靠近
    5. 误差小于阈值即认为到位，停止伺服

它只负责"看 + 决策"，真正的运动学解算和电机下发交给 vik.py 里的 IK 服务。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image   # 订阅相机原始图像
from cv_bridge import CvBridge      # ROS Image 消息 <-> OpenCV ndarray 互转
import cv2                          # OpenCV：ArUco 检测、位姿估计、UI 显示
import numpy as np                  # 矩阵 / 向量运算
import threading                    # 预留（当前未直接使用）

# 导入增量控制服务（即 vik.py 提供的服务接口）
from arm_interfaces.srv import Targetpoint

class VisualServoGraspNode(Node):
    """
    视觉伺服抓取节点：检测 ArUco、计算误差、闭环调用增量 IK 服务。
    """
    def __init__(self):
        super().__init__('visual_servo_grasp_node')
        
        # ==========================================
        # 1. 核心配置参数
        # ==========================================
        # 相机内参矩阵 K：fx/fy 为焦距(像素)，cx/cy 为主点(像素)，由相机标定得到
        # 用于把图像像素 ↔ 相机系下的 3D 射线相互换算，是位姿估计的基础
        self.camera_matrix = np.array([[1588.195699, 0.0, 1337.086397],
                                       [0.0, 1590.571819, 1051.471162],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        # 畸变系数 [k1, k2, p1, p2, k3]：校正镜头的径向/切向畸变
        self.dist_coeffs = np.array([-0.297539, 0.089308, -0.000486, 0.000289, 0.0], dtype=np.float32)
        self.marker_length = 0.05  # ArUco 码的实际边长(米)，位姿估计的尺度基准，必须和打印件一致
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)  # 使用的字典：4x4 共 1000 个码
        self.aruco_params = cv2.aruco.DetectorParameters()  # 检测参数（用默认值）

        # 手眼标定矩阵 (用于将相机坐标系下的误差 旋转到 机械臂基座坐标系)
        # 视觉伺服中，平移部分其实不重要，我们只用到左上角的 3x3 旋转矩阵 R_base_cam
        self.T_base_camera = np.array([
            [ 0.076706, -0.030885, -0.996575,  0.949401],
            [ 0.989038,  0.128847,  0.072133, -0.032458],
            [ 0.126178, -0.991183,  0.040429,  0.183285],
            [ 0.000000,  0.000000,  0.000000,  1.000000]
        ], dtype=np.float32)
        self.R_base_cam = self.T_base_camera[:3, :3]

        # 抓取逻辑参数
        self.GRIPPER_MARKER_ID = 0  # 假设固定在夹爪上的 ArUco 码 ID 为 0
        self.target_marker_id = None # 目标码 ID (由用户鼠标点击选择)
        
        # 比例控制器参数 (P-Controller)
        self.Kp = 0.4          # 增益系数：每次只走误差距离的 40%，防止超调和震荡
        self.MAX_STEP = 0.02   # 安全限制：每次最大只允许走 2 厘米
        self.STOP_TOLERANCE = 0.005 # 停止阈值：误差小于 5 毫米认为到达目标

        # 抓取物理偏置 (非常重要)
        # 手爪上的码和目标上的码不能完全重合，否则会撞坏。
        # 假设手爪码需要停在目标码正上方 Z 轴 10 厘米处、Y 轴偏移 2 厘米处
        self.grasp_offset_base = np.array([0.0, 0.02, 0.10]) 

        # ==========================================
        # 2. 状态缓存与 ROS 接口
        # ==========================================
        self.latest_markers = {} # 当前帧检测到的所有码：{id: {'tvec':平移向量, 'pixel':像素中心}}
        self.is_servoing = False     # 伺服开关：False=待机，True=正在闭环追踪目标
        self.is_waiting_ik = False # 是否在等 IK 服务返回；防止上一步没走完就连发下一条指令

        self.bridge = CvBridge()
        # 订阅相机图像，队列深度 1（只关心最新帧，旧帧丢弃以降低延迟）
        self.sub_image = self.create_subscription(Image, '/raw_image', self.image_callback, 1)
        # 创建增量 IK 服务的客户端（对应 vik.py 的服务端）
        self.ik_client = self.create_client(Targetpoint, 'calculate_incremental_ik')

        # 启动控制循环定时器：每 0.1 秒触发一次 servo_control_loop（10Hz 闭环频率）
        self.control_timer = self.create_timer(0.3, self.servo_control_loop)

        # OpenCV 窗口 + 鼠标回调：让用户能在画面上点选目标码
        cv2.namedWindow("Visual Servoing")
        cv2.setMouseCallback("Visual Servoing", self.mouse_click_event)
        self.get_logger().info("✅ 视觉伺服节点启动！请在画面中左键点击目标 ArUco 码进行锁定。")

    def mouse_click_event(self, event, x, y, flags, param):
        """鼠标回调：左键点击画面，选中离点击位置最近的码作为抓取目标并开启伺服。"""
        # 只处理左键按下事件；x, y 是点击处的像素坐标（注意是缩小显示前还是后取决于窗口设置）
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.latest_markers:
                self.get_logger().warn("当前画面未检测到任何码！")
                return

            # 遍历所有码，找到离点击点最近、且在 50 像素范围内的那个
            min_dist = float('inf')
            selected_id = None
            for marker_id, info in self.latest_markers.items():
                if marker_id == self.GRIPPER_MARKER_ID:
                    continue # 跳过夹爪自身的码，它不能作为抓取目标

                u, v = info['pixel']
                dist = np.sqrt((x - u)**2 + (y - v)**2)  # 点击点到码中心的欧氏距离
                if dist < min_dist and dist < 50: # 必须落在码附近 50 像素内才算选中
                    min_dist = dist
                    selected_id = marker_id

            if selected_id is not None:
                # 选中成功：记录目标 ID，打开伺服开关，控制循环随即开始工作
                self.target_marker_id = selected_id
                self.is_servoing = True
                self.get_logger().info(f"🎯 成功锁定目标物 ID: {self.target_marker_id}，开始视觉伺服闭环！")
            else:
                # 附近没有有效码：视为取消，关闭伺服
                self.is_servoing = False
                self.target_marker_id = None
                self.get_logger().info("🛑 取消伺服伺服。")

    def image_callback(self, msg):
        """图像回调：每来一帧就检测所有 ArUco 码、估计其 3D 位置并缓存，同时刷新 UI。"""
        # ROS Image 消息 → OpenCV BGR 图像；转换失败（如格式不符）直接丢弃该帧
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            return

        # 转灰度后做 ArUco 检测，返回每个码的四个角点 corners 和对应的 ids
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        current_markers = {}
        if ids is not None:
            # 对每个码做单目位姿估计，得到旋转向量 rvecs 和平移向量 tvecs（码在相机系下的 3D 位姿）
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

            for i in range(len(ids)):
                m_id = ids[i][0]        # 当前码的 ID
                tvec = tvecs[i][0]      # 当前码在相机系下的平移向量 (x, y, z)，单位米

                # 取四个角点的均值作为像素中心 (u, v)，供鼠标点选和 UI 文字定位用
                c = corners[i][0]
                u = int(np.mean(c[:, 0]))
                v = int(np.mean(c[:, 1]))

                current_markers[m_id] = {'tvec': tvec, 'pixel': (u, v)}

                # UI 绘制：目标码=绿色，夹爪码=蓝色，其它码=黄色
                color = (0, 255, 0) if m_id == self.target_marker_id else ((255, 0, 0) if m_id == self.GRIPPER_MARKER_ID else (0, 255, 255))
                cv2.aruco.drawDetectedMarkers(frame, corners)  # 画出码的边框
                cv2.putText(frame, f"ID:{m_id}", (u - 20, v - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  # 标注 ID

        # 整体替换缓存（一次性赋值，避免控制循环读到半更新的数据）
        self.latest_markers = current_markers

        # 左上角绘制状态提示：当前是否在伺服、锁定了哪个目标 ID
        status_text = f"Servoing: {'ON (Target ID:'+str(self.target_marker_id)+')' if self.is_servoing else 'OFF (Click to select target)'}"
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if not self.is_servoing else (0, 255, 0), 2)

        # 缩小到一半尺寸显示（原图分辨率较高，省屏幕空间）；waitKey(1) 让 OpenCV 刷新窗口并响应鼠标
        cv2.imshow("Visual Servoing", cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1)

    def servo_control_loop(self):
        """核心控制循环：计算误差并下发增量 (运行在定时器线程)"""
        if not self.is_servoing or self.is_waiting_ik:
            return

        # 1. 检查是否同时看到了手爪和目标
        if self.GRIPPER_MARKER_ID not in self.latest_markers or self.target_marker_id not in self.latest_markers:
            self.get_logger().warn("丢失视野！手爪码或目标码被遮挡，暂停移动。")
            return

        # 2. 获取两者在【相机坐标系】下的 3D 平移向量 (单位: 米)
        t_cam_gripper = self.latest_markers[self.GRIPPER_MARKER_ID]['tvec']
        t_cam_target = self.latest_markers[self.target_marker_id]['tvec']

        # 3. 计算【相机坐标系】下的相对误差向量 (指向目标的向量)
        error_cam = t_cam_target - t_cam_gripper

        # 4. 坐标系转换：将相机系下的误差，旋转到【机械臂基座坐标系】下
        # 只有在基座坐标系下，机械臂才知道 dx, dy, dz 该怎么走
        error_base = self.R_base_cam @ error_cam

        # 5. 加上抓取物理偏置
        # 比如：手爪码不能和目标码重合，我们需要手爪停在目标上方 10cm 处
        # 即，理想的误差不是变成 0，而是变成预设的偏置
        # 所以实际需要走的向量 = 总误差向量 - 偏置向量
        step_base = error_base - self.grasp_offset_base

        # 6. 计算标量距离，判断是否到达
        distance = np.linalg.norm(step_base)
        if distance < self.STOP_TOLERANCE:
            self.get_logger().info("🎉 到达抓取位置！停止伺服。可以触发闭合手爪指令！")
            self.is_servoing = False
            return

        # 7. 比例控制器 (P-Controller) & 安全限幅
        # 我们不一次性走完，只走 40% (Kp=0.4)，这样动作平滑，且能抵抗识别跳动
        cmd_step = self.Kp * step_base
        
        # 限制单次最大步长 (勾股定理求模长，等比例缩小)
        cmd_norm = np.linalg.norm(cmd_step)
        if cmd_norm > self.MAX_STEP:
            cmd_step = cmd_step * (self.MAX_STEP / cmd_norm)

        # 8. 异步调用我们写好的雅可比增量 IK 服务
        self.get_logger().info(f"-> 闭环微调增量: dx={cmd_step[0]:.4f}, dy={cmd_step[1]:.4f}, dz={cmd_step[2]:.4f} (距目标: {distance*100:.1f}cm)")
        
        req = Targetpoint.Request()
        req.target_x = float(cmd_step[0])
        req.target_y = float(cmd_step[1])
        req.target_z = float(cmd_step[2])
        
        self.is_waiting_ik = True
        future = self.ik_client.call_async(req)
        future.add_done_callback(self.ik_callback)

    def ik_callback(self, future):
        """IK 服务的异步回调：一步增量执行完毕后被触发，负责"解锁"以允许下一步。"""
        # 无论成功失败都要复位等待标志，否则控制循环会永远卡在 is_waiting_ik=True
        self.is_waiting_ik = False
        try:
            res = future.result()
            # 服务返回非 SUCCESS（如下位机超时）只告警，不中断伺服，下一拍会重新尝试
            if res.result != Targetpoint.Response.SUCCESS:
                self.get_logger().warn(f"步进反馈异常: {res.message}")
        except Exception as e:
            # 服务调用本身异常（如服务端没启动、通信中断）
            self.get_logger().error(f"IK 服务调用失败: {e}")

def main():
    rclpy.init()                      # 初始化 ROS 2 通信
    node = VisualServoGraspNode()     # 创建视觉伺服节点
    try:
        rclpy.spin(node)              # 进入事件循环，持续处理图像/定时器/服务回调
    except KeyboardInterrupt:
        pass                          # Ctrl+C 优雅退出
    finally:
        cv2.destroyAllWindows()       # 关闭 OpenCV 窗口
        node.destroy_node()           # 销毁节点
        rclpy.shutdown()              # 关闭 ROS 2

if __name__ == '__main__':
    main()