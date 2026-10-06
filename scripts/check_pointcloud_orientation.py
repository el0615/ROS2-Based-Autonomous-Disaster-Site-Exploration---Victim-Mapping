#!/usr/bin/env python3

import struct
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


TOPIC = '/j100_0001/sensors/camera_0/points'


class OrientationChecker(Node):

    def __init__(self):
        super().__init__('pointcloud_orientation_checker')

        self.sub = self.create_subscription(
            PointCloud2,
            TOPIC,
            self.callback,
            qos_profile_sensor_data
        )

        print("Waiting for PointCloud...")

    def read_xyz(self, msg, u, v):

        fields = {f.name: f.offset for f in msg.fields}

        base = v * msg.row_step + u * msg.point_step

        endian = '>' if msg.is_bigendian else '<'

        x = struct.unpack_from(
            endian + 'f',
            msg.data,
            base + fields['x']
        )[0]

        y = struct.unpack_from(
            endian + 'f',
            msg.data,
            base + fields['y']
        )[0]

        z = struct.unpack_from(
            endian + 'f',
            msg.data,
            base + fields['z']
        )[0]

        return x, y, z

    def callback(self, msg):

        cx = msg.width // 2
        cy = msg.height // 2

        offset = 100

        points = {
            'CENTER': (cx, cy),
            'LEFT':   (cx - offset, cy),
            'RIGHT':  (cx + offset, cy),
            'UP':     (cx, cy - offset),
            'DOWN':   (cx, cy + offset),
        }

        print()
        print("=== PointCloud Orientation Test ===")
        print("frame_id:", msg.header.frame_id)
        print()

        for name, (u, v) in points.items():

            x, y, z = self.read_xyz(msg, u, v)

            print(
                f"{name:6s} pixel=({u:3d},{v:3d}) "
                f"-> x={x:7.3f}, y={y:7.3f}, z={z:7.3f}"
            )

        print()
        rclpy.shutdown()


def main():

    rclpy.init()

    node = OrientationChecker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
