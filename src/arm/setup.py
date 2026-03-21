from setuptools import find_packages, setup
from glob import glob
package_name = 'arm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name+'/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ni',
    maintainer_email='ni@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'solve_arm_ik=arm.solve_arm_ik:main',
            'motor_serial=arm.motor_serial:main',
            'gcode=arm.gcode:main',
            'linear_move=arm.linear_move:main',
            'circular_move=arm.circular_move:main',
            'gcode_interpreter=arm.gcode_interpreter:main',
        ],
    },
)
