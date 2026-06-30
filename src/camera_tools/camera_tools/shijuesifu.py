#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【视觉伺服抓取 (Visual Servoing) 节点】—— 增量 IK 服务的"上层大脑"。

整体闭环逻辑（Eye-to-Hand 手眼分离构型）：
    1. 相机持续检测画面里的 ArUco 码，估计每个码在相机系下的 3D 位置
    2. 用户鼠标点选目标码，锁定抓取对象
    3. 【新增】先计算预抓取位置，通过绝对 IK 服务将机械臂移动到目标附近
    4. 控制循环不断计算"夹爪码 → 目标码"的相对误差，转到机械臂基座坐标系
    5. 用比例控制器(P)取一小步，调用 calculate_incremental_ik 服务驱动机械臂靠近
    6. 误差小于阈值即认为到位，停止伺服

它只负责"看 + 决策"，真正的运动学解算和电机下发交给 vik.py 里的 IK 服务。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image   # 订阅相机原始图像
from cv_bridge import CvBridge      # ROS Image 消息 <-> OpenCV ndarray 互转
import cv2                          # OpenCV：ArUco 检测、位姿估计、UI 显示
import numpy as np                  # 矩阵 / 向量运算

# 导入增量控制服务（即 vik.py 提供的服务接口）
from arm_interfaces.srv import Targetpoint

class VisualServoGraspNode(Node):
    """
    视觉伺服抓取节点：检测 ArUco、预抓取接近、闭环调用增量 IK 服务。
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
        self.GRIPPER_MARKER_ID = 2  # 假设固定在夹爪上的 ArUco 码 ID 为 0
        self.target_marker_id = None # 目标码 ID (由用户鼠标点击选择)

        # 【新增】跳帧检测：ArUco 检测很耗 CPU（~30-80ms），
        # 不需要每帧都检测，每 N 帧检测一次即可，中间帧复用上一次结果
        self._frame_count = 0
        self._DETECT_EVERY_N = 1      # 每 3 帧做一次全量检测
        self._cached_detect_frame = None  # 缓存上一次检测+绘制好的画面，跳帧时直接显示

        # 比例控制器参数 (P-Controller)
        self.Kp = 0.4          # 增益系数：每次只走误差距离的 40%，防止超调和震荡
        self.MAX_STEP = 0.02   # 安全限制：每次最大只允许走 2 厘米
        self.STOP_TOLERANCE = 0.005 # 停止阈值：误差小于 5 毫米认为到达目标

        # 抓取物理偏置 (非常重要)
        # 手爪上的码和目标上的码不能完全重合，否则会撞坏。
        # 假设手爪码需要停在目标码正上方 Z 轴 10 厘米处、Y 轴偏移 2 厘米处
        self.grasp_offset_base = np.array([0.0, 0.02, 0.10])

        # 【新增】预抓取阶段参数
        # 预抓取位置比最终伺服目标多一段安全距离，确保机械臂先到"附近"
        # 然后再由视觉伺服闭环精调至精确抓取位
        self.PRE_GRASP_Z_EXTRA = 0.05  # 预抓取时比最终目标多 5cm 的 Z 向安全距离

        # ==========================================
        # 2. 状态缓存与 ROS 接口
        # ==========================================
        self.latest_markers = {} # 当前帧检测到的所有码：{id: {'tvec':平移向量, 'pixel':像素中心}}
        self.is_servoing = False     # 伺服开关：False=待机，True=正在闭环追踪目标
        self.is_waiting_ik = False # 是否在等 IK 服务返回；防止上一步没走完就连发下一条指令

        # 【新增】预抓取状态机
        # 'idle'             - 空闲，等待用户点选目标
        # 'pending_approach' - 已锁定目标，等待下一个控制周期发起预抓取移动
        # 'approaching'      - 预抓取移动中，等待绝对 IK 服务返回
        # 'servoing'         - 预抓取到位，视觉伺服闭环精调中
        self.pre_grasp_state = 'idle'

        self.bridge = CvBridge()
        # 订阅相机图像，队列深度 1（只关心最新帧，旧帧丢弃以降低延迟）
        self.sub_image = self.create_subscription(Image, '/raw_image', self.image_callback, 1)
        # 增量 IK 服务的客户端（视觉伺服闭环微调用）
        self.ik_client = self.create_client(Targetpoint, 'calculate_incremental_ik')
        # 【新增】绝对 IK 服务的客户端（预抓取大范围移动用）
        self.ik_absolute_client = self.create_client(Targetpoint, 'calculate_ik')

        # 启动控制循环定时器：每 0.3 秒触发一次 servo_control_loop
        self.control_timer = self.create_timer(0.3, self.servo_control_loop)

        # OpenCV 窗口 + 鼠标回调：让用户能在画面上点选目标码
        cv2.namedWindow("Visual Servoing")
        cv2.setMouseCallback("Visual Servoing", self.mouse_click_event)
        self.get_logger().info("✅ 视觉伺服节点启动！左键点击目标 ArUco 码 → 预抓取接近 → 视觉伺服精调。")

    def mouse_click_event(self, event, x, y, _flags, _param):
        """
        鼠标回调：左键点击画面，选中离点击位置最近的码作为抓取目标。
        【修改】点击后不直接开启伺服，而是先进入预抓取接近阶段：
              1. 锁定目标，标记为 pending_approach
              2. 控制循环检测到后，计算预抓取位置并通过绝对 IK 移动机械臂
              3. 移动到位后，自动开启视觉伺服闭环精调
        点击空白区域则取消一切操作。
        """
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
                # 【修改】锁定目标，先进入预抓取接近阶段，不直接开启伺服
                self.target_marker_id = selected_id
                self.is_servoing = False
                self.pre_grasp_state = 'pending_approach'
                self.get_logger().info(f"🎯 锁定目标 ID: {selected_id}，即将先到达附近位置...")
            else:
                # 点击空白区域 → 取消当前一切操作
                self.cancel_servo()

    def cancel_servo(self):
        """取消当前的伺服或预抓取操作，回到空闲状态。"""
        if self.pre_grasp_state != 'idle':
            self.get_logger().info("🛑 取消当前操作，回到空闲状态。")
        self.is_servoing = False
        self.pre_grasp_state = 'idle'
        self.target_marker_id = None

    def _start_approach(self):
        """
        【新增】计算预抓取位置并发送绝对 IK 命令，让机械臂先到达目标附近。

        预抓取位置计算：
            1. 获取目标标记物在相机系下的 3D 位置
            2. 通过手眼矩阵转换到机械臂基座系
            3. 加上预抓取偏置（最终抓取偏置 + 额外 Z 向安全距离）
            4. 通过 calculate_ik 服务发送绝对位置指令

        注意：由于 calculate_ik 期望的是末端法兰盘位置，而此处计算的是
        标记物位置，两者之间存在 T_end_marker 的固定偏置。该偏置较小（~5cm），
        预抓取阶段容忍此误差，后续视觉伺服会自动修正。
        """
        if self.target_marker_id is None or self.target_marker_id not in self.latest_markers:
            self.get_logger().error("❌ 无法启动预抓取：目标标记物丢失！请重新点选。")
            self.cancel_servo()
            return

        # 1. 获取目标在相机坐标系下的 3D 位置（平移向量）
        t_cam_target = self.latest_markers[self.target_marker_id]['tvec']

        # 2. 转换到机械臂基座坐标系
        #    P_base = R_base_cam * P_cam + t_base_cam
        t_base_target = self.R_base_cam @ t_cam_target + self.T_base_camera[:3, 3]

        # 3. 计算预抓取位置（基座系下）
        #    预抓取偏置 = 最终抓取偏置 + 额外 Z 向安全距离
        #    这样机械臂先停在比最终位置更"远"的地方，再由伺服从容趋近
        pre_grasp_offset = self.grasp_offset_base.copy()
        pre_grasp_offset[2] += self.PRE_GRASP_Z_EXTRA
        pre_grasp_base = t_base_target + pre_grasp_offset

        self.get_logger().info(
            f"📍 目标位置(基座系): [{t_base_target[0]:.3f}, {t_base_target[1]:.3f}, {t_base_target[2]:.3f}]"
        )
        self.get_logger().info(
            f"🚀 预抓取位置(基座系): [{pre_grasp_base[0]:.3f}, {pre_grasp_base[1]:.3f}, {pre_grasp_base[2]:.3f}]"
        )

        # 4. 等待绝对 IK 服务上线
        if not self.ik_absolute_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("❌ 绝对 IK 服务 'calculate_ik' 超时未上线！")
            self.cancel_servo()
            return

        self.pre_grasp_state = 'approaching'
        self.get_logger().info("⏳ 已发送预抓取指令，等待机械臂移动到附近位置...")

        # 5. 发送绝对 IK 命令（异步，完成后由 ik_absolute_callback 处理）
        req = Targetpoint.Request()
        req.target_x = float(pre_grasp_base[0])
        req.target_y = float(pre_grasp_base[1])
        req.target_z = float(pre_grasp_base[2])

        future = self.ik_absolute_client.call_async(req)
        future.add_done_callback(self.ik_absolute_callback)

    def ik_absolute_callback(self, future):
        """
        【新增】绝对 IK 完成回调。
        - 成功：启动视觉伺服闭环精调
        - 失败：取消操作并报告错误
        - 已被取消：忽略结果
        """
        # 检查是否在等待期间被用户取消了
        if self.pre_grasp_state != 'approaching':
            self.get_logger().info("预抓取已被用户取消，忽略 IK 结果。")
            return

        try:
            res = future.result()
            if hasattr(res, 'result') and res.result == Targetpoint.Response.SUCCESS:
                self.get_logger().info("✅ 预抓取到位！启动视觉伺服闭环精调...")
                self.pre_grasp_state = 'servoing'
                self.is_servoing = True
            else:
                msg = res.message if hasattr(res, 'message') else "未知错误"
                self.get_logger().error(f"❌ 预抓取移动失败: {msg}")
                self.cancel_servo()
        except Exception as e:
            self.get_logger().error(f"❌ 预抓取 IK 调用异常: {e}")
            self.cancel_servo()

    def image_callback(self, msg):
        """图像回调：每来一帧就检测所有 ArUco 码、估计其 3D 位置并缓存，同时刷新 UI。

        【性能优化】不是每帧都跑 ArUco 检测（~30-80ms），而是每 N 帧检测一次。
        中间帧只做显示，大幅降低 CPU 占用，画面更流畅。
        """
        self._frame_count += 1
        do_detect = (self._frame_count % self._DETECT_EVERY_N == 0)

        # ROS Image 消息 → OpenCV BGR 图像；转换失败（如格式不符）直接丢弃该帧
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        # ---- 检测阶段：只有第 N 帧才跑完整的 ArUco 检测 ----
        if do_detect:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)

            current_markers = {}
            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

                for i in range(len(ids)):
                    m_id = ids[i][0]
                    tvec = tvecs[i][0]
                    c = corners[i][0]
                    u = int(np.mean(c[:, 0]))
                    v = int(np.mean(c[:, 1]))
                    current_markers[m_id] = {'tvec': tvec, 'pixel': (u, v)}

                    # 颜色：目标码=绿色，夹爪码=蓝色，其它=黄色
                    if m_id == self.target_marker_id:
                        color = (0, 255, 0)
                    elif m_id == self.GRIPPER_MARKER_ID:
                        color = (255, 0, 0)
                    else:
                        color = (0, 255, 255)

                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    cv2.putText(frame, f"ID:{m_id}", (u - 20, v - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 原子替换缓存
            self.latest_markers = current_markers

            # ---- 叠加状态文字 ----
            self._draw_status_overlay(frame)
            self._cached_detect_frame = frame.copy()

        else:
            # 跳帧：直接复用上一帧带检测框的画面，只更新状态文字即可
            if self._cached_detect_frame is not None:
                display_frame = self._cached_detect_frame.copy()
            else:
                display_frame = frame
            self._draw_status_overlay(display_frame)
            frame = display_frame  # 下面统一用 frame 显示

        # ---- 统一显示 ----
        cv2.imshow("Visual Servoing", cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1)

    def _draw_status_overlay(self, frame):
        """在画面上叠加状态信息文字。"""
        if self.pre_grasp_state == 'approaching':
            status_text = f"Approaching target ID:{self.target_marker_id}..."
            status_color = (255, 165, 0)  # 橙色 - 正在预抓取移动
        elif self.pre_grasp_state == 'pending_approach':
            status_text = f"Preparing approach to ID:{self.target_marker_id}..."
            status_color = (255, 165, 0)  # 橙色 - 准备发起预抓取
        elif self.pre_grasp_state == 'servoing' and self.is_servoing:
            status_text = f"Servoing: ON (Target ID:{self.target_marker_id})"
            status_color = (0, 255, 0)    # 绿色 - 视觉伺服进行中
        else:
            status_text = "Servoing: OFF (Click to select target)"
            status_color = (0, 0, 255)    # 红色 - 空闲

        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, status_color, 2)

    def servo_control_loop(self):
        """
        核心控制循环（定时器 0.3s 周期，运行在定时器线程）：
        - 阶段1: 检测到 pending_approach，发起预抓取绝对 IK 移动
        - 阶段2: 预抓取移动中 (approaching)，等待 IK 回调
        - 阶段3: 预抓取到位后 (servoing)，执行视觉伺服闭环微调
        """
        # ---- 【新增】阶段1：发起预抓取接近 ----
        if self.pre_grasp_state == 'pending_approach':
            self._start_approach()
            return

        # ---- 【新增】阶段2：预抓取移动中，等待绝对 IK 完成 ----
        if self.pre_grasp_state == 'approaching':
            return

        # ---- 阶段3：视觉伺服闭环 ----
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
            self.pre_grasp_state = 'idle'
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
        """增量 IK 服务的异步回调：一步增量执行完毕后被触发，负责"解锁"以允许下一步。"""
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
