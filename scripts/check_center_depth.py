#!/usr/bin/env python3

import math
import struct

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


TOPIC = '/j100_0001/sensors/camera_0/depth/image'


class CenterDepthCheck(Node):

    def __init__(self):
        super().__init__('center_depth_check')

        self.subscription = self.create_subscription(
            Image,
            TOPIC,
            self.depth_callback,
            10
        )

    def depth_callback(self, msg):

        if msg.encoding != '32FC1':
            print(f"Unexpected encoding: {msg.encoding}")
            rclpy.shutdown()
            return

        center_x = msg.width // 2
        center_y = msg.height // 2

        bytes_per_pixel = 4

        offset = (
            center_y * msg.step
            + center_x * bytes_per_pixel
        )

        endian = '>' if msg.is_bigendian else '<'

        depth = struct.unpack_from(
            endian + 'f',
            msg.data,
            offset
        )[0]

        print()
        print("=== RGB-D Center Depth ===")
        print(f"resolution : {msg.width} x {msg.height}")
        print(f"pixel      : ({center_x}, {center_y})")

        if math.isfinite(depth):
            print(f"distance   : {depth:.3f} m")
        else:
            print(f"distance   : {depth} (no valid depth)")

        print()

        rclpy.shutdown()


def main():

    rclpy.init()

    node = CenterDepthCheck()

    rclpy.spin(node)


if __name__ == '__main__':
    main()
