from setuptools import find_packages, setup

package_name = 'camera_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ni',
    maintainer_email='3310056144@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                'camera_pub=camera_tools.camera_pub:main',
                'roi_selector=camera_tools.roi_selector:main',
                'aruco_detector=camera_tools.aruco_detector:main',
                'aruco_selector=camera_tools.aruco_selector:main',
                'shouyanbiaoding=camera_tools.shouyanbiaoding:main',
        ],
    },
)
