#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import PointCloud2


INPUT_TOPIC = '/j100_0001/sensors/camera_0/points'
OUTPUT_TOPIC = '/j100_0001/sensors/camera_0/points_corrected'


class PointCloudToOptical(Node):

    def __init__(self):
        super().__init__('pointcloud_to_optical')

        output_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=3
        )

        self.publisher = self.create_publisher(
            PointCloud2,
            OUTPUT_TOPIC,
            output_qos
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            INPUT_TOPIC,
            self.callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f'{INPUT_TOPIC} -> {OUTPUT_TOPIC}'
        )

    def callback(self, msg):

        offsets = {f.name: f.offset for f in msg.fields}

        if not all(axis in offsets for axis in ['x', 'y', 'z']):
            self.get_logger().error('x/y/z fields not found')
            return

        data = bytearray(msg.data)

        count = msg.width * msg.height

        endian = '>' if msg.is_bigendian else '<'
        dtype = np.dtype(endian + 'f4')

        x = np.ndarray(
            shape=(count,),
            dtype=dtype,
            buffer=data,
            offset=offsets['x'],
            strides=(msg.point_step,)
        )

        y = np.ndarray(
            shape=(count,),
            dtype=dtype,
            buffer=data,
            offset=offsets['y'],
            strides=(msg.point_step,)
        )

        z = np.ndarray(
            shape=(count,),
            dtype=dtype,
            buffer=data,
            offset=offsets['z'],
            strides=(msg.point_step,)
        )

        # 원본 보존
        old_x = x.copy()
        old_y = y.copy()
        old_z = z.copy()

        # camera_link:
        # X = Forward, Y = Left, Z = Up
        #
        # optical:
        # X = Right, Y = Down, Z = Forward
        #
        # link -> optical
        x[:] = -old_y
        y[:] = -old_z
        z[:] =  old_x

        output = PointCloud2()

        output.header.stamp = msg.header.stamp
        output.header.frame_id = 'camera_0_color_optical_frame'

        output.height = msg.height
        output.width = msg.width
        output.fields = msg.fields
        output.is_bigendian = msg.is_bigendian
        output.point_step = msg.point_step
        output.row_step = msg.row_step
        output.data = bytes(data)
        output.is_dense = msg.is_dense

        self.publisher.publish(output)


def main():
    rclpy.init()

    node = PointCloudToOptical()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

