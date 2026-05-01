import launch
import launch_ros

def generate_launch_description():

    motor_serial_node = launch_ros.actions.Node(
        package='arm',
        executable="motor_serial",
        output='log',
    )
    solve_arm_ik_serves = launch_ros.actions.Node(
        package='arm',
        executable="solve_arm_ik",
        output='log',
    )
    linear_move = launch_ros.actions.Node(
        package='arm',
        executable='linear_move',
        output='log',
    )
    circular_move = launch_ros.actions.Node(
        package='arm',
        executable='circular_move',
        output='log',
    )
    zhengdh = launch_ros.actions.Node(
        package='arm',
        executable='zhengdh',
        output='log',
    )
   # 合成启动描述并返回
    launch_description = launch.LaunchDescription([
        motor_serial_node,
        solve_arm_ik_serves,
        linear_move,
        circular_move,
        zhengdh
    ])
    return launch_description