#!/usr/bin/env python3

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


TOPIC = '/j100_0001/sensors/camera_0/points'


class AxisChecker(Node):

    def __init__(self):
        super().__init__('pointcloud_axis_checker')

        self.sub = self.create_subscription(
            PointCloud2,
            TOPIC,
            self.callback,
            qos_profile_sensor_data
        )

        print("Waiting for PointCloud...")

    def callback(self, msg):

        offsets = {f.name: f.offset for f in msg.fields}

        print()
        print("=== PointCloud Info ===")
        print("frame_id :", msg.header.frame_id)
        print("width    :", msg.width)
        print("height   :", msg.height)
        print("point_step:", msg.point_step)

        data = np.frombuffer(msg.data, dtype=np.uint8)

        count = msg.width * msg.height

        xyz = {}

        for axis in ['x', 'y', 'z']:

            arr = np.ndarray(
                shape=(count,),
                dtype=np.float32,
                buffer=data,
                offset=offsets[axis],
                strides=(msg.point_step,)
            )

            arr = arr[np.isfinite(arr)]

            xyz[axis] = arr

            print()
            print(f"{axis.upper()} axis")
            print(f"  min    = {np.min(arr):.3f}")
            print(f"  max    = {np.max(arr):.3f}")
            print(f"  mean   = {np.mean(arr):.3f}")
            print(f"  median = {np.median(arr):.3f}")

        print()
        print("=== Center Point ===")

        center = (
            (msg.height // 2) * msg.width
            + (msg.width // 2)
        )

        for axis in ['x', 'y', 'z']:

            arr = np.ndarray(
                shape=(count,),
                dtype=np.float32,
                buffer=data,
                offset=offsets[axis],
                strides=(msg.point_step,)
            )

            print(f"{axis} = {arr[center]:.3f}")

        rclpy.shutdown()


def main():

    rclpy.init()

    node = AxisChecker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
