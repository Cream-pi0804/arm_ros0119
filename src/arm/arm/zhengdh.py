import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Pose
from arm_interfaces.msg import Jointangle, Pointnow

class DHPosePublisher(Node):
    def __init__(self):
        super().__init__('dh_pose_publisher')
        
        # 订阅当前机械臂脉冲/位置反馈
        self.sub = self.create_subscription(
            Pointnow,
            '/arm/Pointnow',
            self.point_now_callback,
            10
        )
        
        # 发布经过 DH 正解算出来的法兰盘绝对位姿
        self.pub = self.create_publisher(Pose, '/dh_robot_pose', 10)
        
        # 将你在 IK 里的转换系数逆推出来（脉冲转弧度）
        self.PULSE_PER_DEGREE1 = 600 / 40.0
        self.PULSE_PER_DEGREE2 = 10000 / 60.0
        self.PULSE_PER_DEGREE3 = 2600 / 180.0

        self.get_logger().info('DH 正运动学实时发布节点已启动，正在监听 /arm/Pointnow ...')

    def _get_transform(self, alpha, a, d, theta):
        """标准的 DH 变换矩阵"""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
            [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), d*np.cos(alpha)],
            [0, 0, 0, 1]
        ])

    def forward_kinematics_full(self, qs):
        """计算末端法兰的完整 4x4 姿态矩阵"""
        q1, q2, q3 = qs
        dh_params = [
            (0, 0.135, 0.0913, q1),
            (-np.pi/2, 0.135, 0, -q2),
            (0, 0.08, 0, -q2 + 3.6585 * np.pi / 180),
            (0, 0.43881, 0, q3 - 3.0054),
            (-np.pi/2, 0.12286, 0.02214, 0)
        ]
        
        t_total = np.eye(4)
        for alpha, a, d, theta in dh_params:
            t_total = t_total @ self._get_transform(alpha, a, d, theta)
            
        return t_total

    def point_now_callback(self, msg):
        """
        每次收到底层反馈，都把它还原成弧度，通过 DH 算 4x4 矩阵，并打包成 Pose 发送
        【注意】：这里我假设你的 msg 属性叫做 motor_1/2/3。如果不同请自行修改。
        """
        try:
            # 1. 逆向转换：脉冲 -> 度 -> 弧度
            deg1 = float(msg.x) / self.PULSE_PER_DEGREE1
            deg2 = float(msg.y) / (2 * self.PULSE_PER_DEGREE2)
            deg3 = float(msg.z) / self.PULSE_PER_DEGREE3
            
            q1 = np.radians(deg1)
            q2 = np.radians(deg2)
            q3 = np.radians(deg3)
            
            # 2. 调用 DH 核心算法获取 4x4 矩阵
            T = self.forward_kinematics_full([q1, q2, q3])
            
            # 3. 提取平移(XYZ)和旋转(四元数)
            pos_x, pos_y, pos_z = T[0, 3], T[1, 3], T[2, 3]
            
            rotation_matrix = T[:3, :3]
            # 用 scipy 提取四元数 [qx, qy, qz, qw]
            qx, qy, qz, qw = R.from_matrix(rotation_matrix).as_quat()

            # 4. 组装 Pose 消息并发布
            pose_msg = Pose()
            pose_msg.position.x = float(pos_x)
            pose_msg.position.y = float(pos_y)
            pose_msg.position.z = float(pos_z)
            
            pose_msg.orientation.x = float(qx)
            pose_msg.orientation.y = float(qy)
            pose_msg.orientation.z = float(qz)
            pose_msg.orientation.w = float(qw)

            self.pub.publish(pose_msg)
            
        except Exception as e:
            self.get_logger().error(f"DH 解算失败: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DHPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()