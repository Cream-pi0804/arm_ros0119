# arm_ros0119

基于 **ROS 2 Humble** 的三自由度机械臂视觉抓取系统。

项目把"看（相机/ArUco 识别）—想（运动学解算）—动（串口下发电机脉冲）"整条链路打通，支持两种工作方式：

- **离散抓取**：相机识别 ArUco 目标，解析逆运动学，调度机械臂走到目标点完成抓取。
- **视觉伺服**：基于雅可比的增量逆运动学闭环，让机械臂实时朝目标码逼近。

此外还提供直线 / 圆弧插补与 G 代码轨迹执行，可用于简单的轨迹加工演示。

---

## 目录结构

```
arm_ros0119/
├── src/
│   ├── arm/                # 机械臂运动学与运动控制（Python 节点）
│   ├── arm_interfaces/     # 机械臂相关自定义 msg / srv 接口
│   ├── camera_tools/       # 相机采集、ArUco 识别、手眼标定、视觉伺服
│   └── camera_msgs/        # 相机相关自定义 msg 接口
├── biaoding.txt            # 相机标定结果（内参 / 畸变系数）记录
├── motion_demo.txt         # G 代码运动演示样例
└── README.md
```

> `build/`、`install/`、`log/` 为 colcon 构建产物，已在 `.gitignore` 中忽略。

---

## 软件包说明

### `arm` —— 运动学与运动控制

| 可执行节点 | 节点功能 |
| --- | --- |
| `solve_arm_ik` | 逆运动学服务（`calculate_ik`）。基于 DH 模型 + `scipy.optimize` 数值求解，把目标笛卡尔点解算为三个电机角度并发布。 |
| `vik` | 速度级 / 增量逆运动学服务（`calculate_incremental_ik`）。用雅可比矩阵将空间位移 `dx,dy,dz` 换算为电机角度增量，下发后阻塞等待下位机"到位"反馈。 |
| `zhengdh` | 正运动学发布节点。订阅 `/arm/Pointnow`，实时解算法兰盘绝对位姿并发布到 `/dh_robot_pose`（供手眼标定使用）。 |
| `motor_serial` | 串口通信节点。通过 `/dev/ttyUSB0` 与下位机（STM32）收发，下发电机脉冲、回传当前状态，支持断线自动重连。 |
| `grasp_scheduler_node` | 离散抓取调度。订阅 `/vision/target_Pose`，依次调用 IK、移动、夹爪开合，完成"抓取—搬运—放置"流程。 |
| `gcode_interpreter` | G 代码解释器。解析 G 代码（G01/G17/G18/G19 等），调用直线/圆弧插补服务。 |
| `linear_move` | 直线插补服务端（`/arm/linear_interpolation`）。 |
| `circular_move` | 圆弧插补服务端（`/arm/circular_interpolation`），通过起点/途经点/终点定义空间圆弧。 |
| `gcode` | G 代码轨迹规划节点（速度/加速度受限的轨迹采样）。 |

### `camera_tools` —— 视觉

| 可执行节点 | 节点功能 |
| --- | --- |
| `camera_pub` | 相机采集发布。通过 RTSP（默认 `rtsp://192.168.1.168:554/stream_0`）读取视频流，发布到 `raw_image`，采集与发布解耦。 |
| `aruco_detector` | ArUco 检测，估计标记在相机系下的 3D 位姿。 |
| `aruco_selector` | ArUco 目标选择与抓取，含完整 6D 位姿及姿态转换，结合手眼矩阵把目标变换到机械臂基座系。 |
| `roi_selector` | 鼠标框选目标区域，发布裁剪图。 |
| `shouyanbiaoding` | 眼在手外（Eye-to-Hand）手眼标定。采集机械臂位姿与标记位姿样本，求解 `T_base_camera`。 |
| `shijuesifu` | 视觉伺服抓取"上层大脑"。鼠标点选目标后，用比例控制器闭环调用增量 IK 服务驱动机械臂逼近目标。 |

### `arm_interfaces` —— 机械臂接口

- **msg**：`Jointangle`（三电机脉冲）、`Pointnow`（当前位置 + 到位标志）、`End`（夹爪值 + 到达时间）
- **srv**：`Targetpoint`（目标点求解）、`LinearInterpolation`、`CircularInterpolation`、`GCodeTrajectory`

### `camera_msgs` —— 相机接口

- **msg**：`Image`（自定义图像消息，预留）

---

## 系统架构（数据流）

```
                        ┌─────────────┐
  RTSP 摄像头 ──────────▶│  camera_pub │── /raw_image ─┐
                        └─────────────┘               │
                                                       ▼
                        ┌───────────────────────────────────────┐
                        │ aruco_selector / shijuesifu （视觉决策）│
                        └───────────────────────────────────────┘
                              │ /vision/target_Pose      │ calculate_incremental_ik
                              ▼                           ▼
                  ┌──────────────────────┐     ┌──────────────────┐
                  │ grasp_scheduler_node │     │     vik (增量IK)  │
                  └──────────────────────┘     └──────────────────┘
                              │ calculate_ik              │
                              ▼                           │
                       ┌──────────────┐                  │ /arm/Jointangle
                       │ solve_arm_ik │── /arm/Jointangle─┤
                       └──────────────┘                   ▼
                                              ┌──────────────────────┐
                                              │ motor_serial (串口)   │◀──▶ STM32 下位机
                                              └──────────────────────┘
                                                         │ /arm/Pointnow
                                                         ▼
                                                  ┌────────────┐
                                                  │  zhengdh   │── /dh_robot_pose
                                                  └────────────┘
```

---

## 环境要求

- Ubuntu 22.04 + **ROS 2 Humble**
- Python 3 依赖：`numpy`、`scipy`、`opencv-python`（含 `aruco`）、`pyserial`
- ROS 依赖：`rclpy`、`sensor_msgs`、`geometry_msgs`、`std_srvs`、`cv_bridge`

```bash
sudo apt install ros-humble-cv-bridge python3-opencv
pip3 install numpy scipy pyserial
```

---

## 构建

```bash
cd ~/arm_ros0119
colcon build --symlink-install
source install/setup.bash
```

> 每开一个新终端运行节点前，都需要先 `source install/setup.bash`。

---

## 运行

> 串口设备需要先授权：`sudo chmod 666 /dev/ttyUSB0`

**1. 启动底层运动链（建议每个节点单独开一个终端）**

```bash
ros2 run arm motor_serial          # 串口通信
ros2 run arm solve_arm_ik          # 逆运动学服务
ros2 run arm zhengdh               # 正运动学发布
```

**2. 启动相机与视觉**

```bash
ros2 run camera_tools camera_pub        # 相机采集
ros2 run camera_tools aruco_selector    # ArUco 识别与目标选择
```

**3a. 离散抓取**

```bash
ros2 run arm grasp_scheduler_node
```

**3b. 视觉伺服抓取**

```bash
ros2 run arm vik                        # 增量 IK 服务
ros2 run camera_tools shijuesifu        # 视觉伺服上层
```

**G 代码轨迹演示（可选）**

```bash
ros2 run arm linear_move
ros2 run arm circular_move
ros2 run arm gcode_interpreter
# 轨迹样例见 motion_demo.txt
```

---

## 相机标定

相机内参 / 畸变系数记录在 `biaoding.txt`，标定使用 ROS 标定工具：

```bash
ros2 run camera_calibration cameracalibrator \
  --size 7x8 --square 0.017 --no-service-check -p chessboard \
  --ros-args --remap image:=/raw_image
```

手眼标定（眼在手外）使用 `camera_tools/shouyanbiaoding`，标定得到的 `T_base_camera` 矩阵需回填到 `aruco_selector` / `shijuesifu` 中。

---

## 关键话题与服务

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `/raw_image` | `sensor_msgs/Image` | 相机原始图像 |
| `/arm/Jointangle` | `arm_interfaces/Jointangle` | 下发的三电机脉冲 |
| `/arm/Pointnow` | `arm_interfaces/Pointnow` | 下位机回传的当前位置与到位标志 |
| `/arm/endd` | `arm_interfaces/End` | 夹爪控制 |
| `/dh_robot_pose` | `geometry_msgs/Pose` | 正运动学解算的法兰盘位姿 |
| `/vision/target_Pose` | `geometry_msgs/Pose` | 视觉给出的目标位姿 |
| `calculate_ik` | `arm_interfaces/Targetpoint` | 逆运动学求解服务 |
| `calculate_incremental_ik` | `arm_interfaces/Targetpoint` | 增量逆运动学服务 |

---

## 说明

- 当前各软件包的 `package.xml` / `setup.py` 中 `description`、`license`、`maintainer` 多为默认占位值，正式发布前建议补全。
- 机械臂 DH 参数、脉冲—角度换算系数等硬编码在各节点内，更换硬件时需同步修改。
