#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class FrontLidarCheck(Node):

    def __init__(self):
        super().__init__('front_lidar_check')

        self.subscription = self.create_subscription(
            LaserScan,
            '/j100_0001/sensors/lidar2d_0/scan',
            self.scan_callback,
            10
        )

    def scan_callback(self, msg):

        # 0 rad = LiDAR 정면
        front_index = round(
            (0.0 - msg.angle_min)
            / msg.angle_increment
        )

        front_distance = msg.ranges[front_index]

        print()
        print("=== LiDAR Front Distance ===")
        print(f"index    : {front_index}")
        print(f"distance : {front_distance:.3f} m")
        print()

        rclpy.shutdown()


def main():

    rclpy.init()

    node = FrontLidarCheck()

    rclpy.spin(node)


if __name__ == '__main__':
    main()
