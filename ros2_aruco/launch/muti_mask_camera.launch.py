

from launch import LaunchDescription
from launch_ros.actions import Node

from launch import LaunchContext
from launch.actions import SetEnvironmentVariable



def generate_launch_description():
    ns_moblie_ur5 = "/moblie" 
    ns_mobile = "/mobile_obj" 
    ns_packing_area = "/packing" 
    ns_packing_obj = "/packing_obj" 
    return LaunchDescription([
        Node(
            package='ros2_aruco',
            node_executable='aruco_node',
            output="screen",
            node_namespace=ns_moblie_ur5,
            parameters=[{"camera_calibration_file": "file:///home/spragunr/.ros/camera_info/camera.yaml"}],
            remappings=[
                ('/camera/camera_info', '/camera/camera_info'),
                ('/camera/image', '/camera/image_raw')]
        ),
        Node(
            package="ros2_aruco",
            node_executable="aruco_node",
            node_namespace=ns_packing_area,
            parameters=[{"camera_calibration_file": "file:///home/spragunr/.ros/camera_info/camera.yaml"}],
            remappings=[
                ('/camera/camera_info', '/camera/camera_info'),
                ('/camera/image', '/camera/image_raw')]
        ),
    ])

    
