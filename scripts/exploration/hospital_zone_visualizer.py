#!/usr/bin/env python3
"""Read-only hospital zone overlay for ROS 2 Humble / RViz.

New project code; publishes visualization_msgs/MarkerArray ONLY.
Does not publish /map, costmaps, poses, goals, velocities, or TF.
The original-map geometry and occupancy fingerprint must match before display.
Run --self-test without ROS to validate transforms, polygons, masks and metadata.

Primary interface references:
https://raw.githubusercontent.com/ros2/common_interfaces/humble/visualization_msgs/msg/Marker.msg
https://raw.githubusercontent.com/ros-navigation/navigation2/humble/nav2_map_server/src/map_io.cpp
"""
from __future__ import annotations

import argparse
import array
import copy
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def default_zones() -> Path:
    return Path(__file__).resolve().parents[2] / 'config/exploration/hospital_zones.json'


def pixel_to_map(u: float, v: float, info: dict) -> tuple[float, float]:
    """Pixel EDGES: top-left image -> lower-left map origin, with yaw."""
    ox, oy, yaw = info['origin']
    x = u * info['resolution']
    y = (info['height'] - v) * info['resolution']
    c, s = math.cos(yaw), math.sin(yaw)
    return ox + c*x - s*y, oy + s*x + c*y


def map_to_pixel(x: float, y: float, info: dict) -> tuple[float, float]:
    ox, oy, yaw = info['origin']
    dx, dy = x-ox, y-oy
    c, s = math.cos(yaw), math.sin(yaw)
    return ((c*dx+s*dy)/info['resolution'],
            info['height']-(-s*dx+c*dy)/info['resolution'])


def area(points: list) -> float:
    return abs(sum(p[0]*q[1]-q[0]*p[1]
                   for p, q in zip(points, points[1:]+points[:1]))) / 2


def load_zones(path: Path) -> tuple[dict, list]:
    data = json.loads(path.read_text(encoding='utf-8'))
    info = data['map']
    w, h = info['width'], info['height']
    if w < 1 or h < 1 or info['resolution'] <= 0:
        raise ValueError('Invalid map geometry')
    if len(info['origin']) != 3 or not all(math.isfinite(v) for v in info['origin']):
        raise ValueError('Invalid map origin')
    zones = data['zones']
    ids = [z['id'] for z in zones]
    labels = [z['mask_id'] for z in zones]
    if len(set(ids)) != len(ids) or len(set(labels)) != len(labels):
        raise ValueError('Duplicate region ID')
    for z in zones:
        p = z['polygon_pixel']
        if len(p) < 3 or area(p) <= 0:
            raise ValueError(f"Invalid polygon: {z['id']}")
        for u, v in p:
            if not (math.isfinite(u) and math.isfinite(v) and 0 <= u <= w and 0 <= v <= h):
                raise ValueError(f"Polygon outside reference image: {z['id']}")
        for p, q in zip(z['polygon_pixel'], z['polygon_map_m']):
            if math.dist(pixel_to_map(*p, info), q) > 2e-6:
                raise ValueError(f"Pixel/world coordinates disagree: {z['id']}")
    with gzip.open(path.parent / data['mask'], 'rt', encoding='utf-8') as f:
        mask = json.load(f)
    if (mask['width'], mask['height']) != (w, h):
        raise ValueError('Mask size mismatch')
    runs = mask['runs']
    counts = Counter()
    previous = (-1, 0)
    for row, left, right, region in runs:
        if not (0 <= row < h and 0 <= left < right <= w and region in labels):
            raise ValueError('Invalid mask run')
        if row < previous[0] or (row == previous[0] and left < previous[1]):
            raise ValueError('Overlapping or unsorted mask runs')
        previous = row, right
        counts[region] += right-left
    for z in zones:
        if counts[z['mask_id']] != z['known_free_cells']:
            raise ValueError(f"Mask count mismatch: {z['id']}")
    return data, runs


def verify_map(msg: Any, info: dict) -> tuple[bool, str]:
    m = msg.info
    if (m.width, m.height) != (info['width'], info['height']):
        return False, f'grid mismatch: {m.width}x{m.height}'
    if abs(m.resolution-info['resolution']) > 1e-7:
        return False, f'resolution mismatch: {m.resolution}'
    p, q = m.origin.position, m.origin.orientation
    if abs(p.x-info['origin'][0]) > 1e-4 or abs(p.y-info['origin'][1]) > 1e-4 or abs(p.z) > 1e-4:
        return False, 'map origin mismatch'
    norm = math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
    if norm < 1e-9 or abs(q.x) > 1e-5 or abs(q.y) > 1e-5:
        return False, 'invalid/nonplanar map quaternion'
    yaw = math.atan2(2*q.w*q.z, q.w*q.w-q.z*q.z)
    delta = yaw-info['origin'][2]
    if abs(math.atan2(math.sin(delta), math.cos(delta))) > 1e-5:
        return False, 'map rotation mismatch'
    if msg.header.frame_id.lstrip('/') != info['frame_id']:
        return False, f'frame mismatch: {msg.header.frame_id}'
    if len(msg.data) != info['width']*info['height']:
        return False, 'map data length mismatch'
    # ROS OccupancyGrid uses bottom row first; stored hash follows that order.
    digest = hashlib.sha256(array.array('b', msg.data).tobytes()).hexdigest()
    if digest != info['occupancy_ros_order_sha256']:
        return False, 'occupancy fingerprint mismatch (different map or YAML thresholds)'
    return True, 'reference map geometry and occupancy match'


def run(args: argparse.Namespace, ros_args: list[str], data: dict, runs: list) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from geometry_msgs.msg import Point
        from nav_msgs.msg import OccupancyGrid
        from visualization_msgs.msg import Marker, MarkerArray
    except ImportError as exc:
        print(f'ROS2 import error: {exc}\nRun source /opt/ros/humble/setup.bash and /usr/bin/python3.', file=sys.stderr)
        return 2

    if args.marker_topic.rstrip('/') == args.map_topic.rstrip('/'):
        raise ValueError('Marker topic must differ from map topic')
    info = data['map']
    per_zone = defaultdict(list)
    for row, left, right, region in runs:
        # Two CCW triangles per 1-cell-high rectangle. Never fill non-mask cells.
        for u, v in [(left,row),(left,row+1),(right,row+1),
                     (left,row),(right,row+1),(right,row)]:
            x, y = pixel_to_map(u, v, info)
            per_zone[region].append(Point(x=float(x), y=float(y), z=0.015))

    rclpy.init(args=ros_args)
    node = None
    class ZoneVisualizer(Node):
        def __init__(self) -> None:
            super().__init__('hospital_zone_visualizer')
            self.overlay_allowed = False
            self.sent_keys: set[tuple[str,int]] = set()
            self.last_check = ''
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.zone_pub = self.create_publisher(MarkerArray, args.marker_topic, qos)
            self.map_sub = self.create_subscription(OccupancyGrid, args.map_topic, self.on_map, qos)
            self.refresh_timer = self.create_timer(5.0, self.publish_overlay)
            self.get_logger().info(f'[READY] DISPLAY ONLY; waiting for {args.map_topic}; markers={args.marker_topic}')
            self.get_logger().info(f"[ZONES] rooms={data['room_count']} lobby/corridors={data['nonroom_count']}; no navigation commands")

        def on_map(self, msg: Any) -> None:
            try:
                ok, reason = verify_map(msg, info)
            except (TypeError, ValueError, OverflowError) as exc:
                ok, reason = False, f'invalid map: {exc}'
            if reason != self.last_check:
                self.last_check = reason
                if ok:
                    self.get_logger().info('[MAP_OK] '+reason)
                else:
                    self.get_logger().error('[MAP_MISMATCH] '+reason+'; no overlay until correct map arrives')
            self.overlay_allowed = ok
            if ok:
                self.publish_overlay()
            else:
                self.clear_overlay()

        def marker(self, ns: str, ident: int, kind: int, color: str, alpha: float) -> Any:
            m = Marker()
            m.header.frame_id = info['frame_id']
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = ns, ident, kind, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.r, m.color.g, m.color.b = [int(color[i:i+2],16)/255.0 for i in (1,3,5)]
            m.color.a = float(alpha)
            # Re-published every 5 simulated seconds; do not persist forever after exit.
            m.lifetime.sec = 12
            self.sent_keys.add((ns, ident))
            return m

        def publish_overlay(self) -> None:
            if not self.overlay_allowed:
                return
            msg = MarkerArray()
            for z in data['zones']:
                ident = z['mask_id']
                if not args.no_fill:
                    m = self.marker('hospital_zone_fills', ident, Marker.TRIANGLE_LIST, z['color'], args.fill_alpha)
                    m.points = per_zone[ident]
                    msg.markers.append(m)
                m = self.marker('hospital_zone_boundaries', ident, Marker.LINE_STRIP, z['color'], 1.0)
                m.scale.x = args.line_width
                pts = z['polygon_map_m']
                m.points = [Point(x=float(x),y=float(y),z=0.045) for x,y in pts+[pts[0]]]
                msg.markers.append(m)
                m = self.marker('hospital_zone_labels', ident, Marker.TEXT_VIEW_FACING, '#132536', 1.0)
                m.pose.position.x, m.pose.position.y = map(float,z['label_map_m'])
                m.pose.position.z = 0.18
                m.scale.z = args.text_height
                # Portable ASCII: RViz font installations may not contain Korean glyphs.
                m.text = z['id'].replace('CORRIDOR_', 'C-').replace('CONNECTOR_', 'LINK-').replace('_','-')
                if z['id'].startswith('CONNECTOR'):
                    m.scale.z *= 0.75
                msg.markers.append(m)
            if not args.hide_doors:
                m = self.marker('hospital_user_doors', 0, Marker.SPHERE_LIST, '#F43E46', 1.0)
                m.scale.x = m.scale.y = m.scale.z = 0.18
                m.points = [Point(x=float(d['map_xy_m'][0]),y=float(d['map_xy_m'][1]),z=0.08) for d in data['doors']]
                msg.markers.append(m)
            self.zone_pub.publish(msg)

        def clear_overlay(self) -> None:
            if not self.sent_keys:
                return
            msg = MarkerArray()
            for ns, ident in self.sent_keys:
                m = Marker()
                m.header.frame_id = info['frame_id']
                m.ns, m.id, m.action = ns, ident, Marker.DELETE
                msg.markers.append(m)
            self.zone_pub.publish(msg)
            self.sent_keys.clear()

    try:
        node = ZoneVisualizer()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.clear_overlay()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def self_test(data: dict, runs: list) -> int:
    info = data['map']
    for yaw in (0.0,0.4,-1.5,math.pi):
        test = copy.deepcopy(info);test['origin'][2] = yaw
        for p in [(0,0),(0,info['height']),(info['width'],0),(23.5,41.5)]:
            assert math.dist(p,map_to_pixel(*pixel_to_map(*p,test),test)) < 1e-9
    assert pixel_to_map(0,info['height'],info) == tuple(info['origin'][:2])
    assert len(data['zones']) == 27 and data['room_count'] == 20
    assert len(data['doors']) == 26
    ids = {z['id'] for z in data['zones']}
    assert {'CORRIDOR_E','R02A','R02B','R10A','R10B','R15_T','R13'} <= ids
    for row, left, right, _ in runs:
        assert not (197 <= row < 317 and left < 37), 'Exterior assigned to survey'
    print('[SELF-TEST PASS] 27 zones, 26 user marks; transforms, mask ranges/counts and entrance exclusion')
    print('No ROS2/RViz runtime or robot-motion testing was performed by this self-test.')
    return 0


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zones',type=Path,default=default_zones())
    parser.add_argument('--map-topic',default='/j100_0001/map')
    parser.add_argument('--marker-topic',default='/hospital_zones/markers')
    parser.add_argument('--fill-alpha',type=float,default=.22)
    parser.add_argument('--line-width',type=float,default=.045)
    parser.add_argument('--text-height',type=float,default=.45)
    parser.add_argument('--no-fill',action='store_true')
    parser.add_argument('--hide-doors',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args,ros_args=parser.parse_known_args()
    if not 0<=args.fill_alpha<=1 or args.line_width<=0 or args.text_height<=0:
        parser.error('alpha must be in [0,1] and sizes must be positive')
    data,runs=load_zones(args.zones.expanduser().resolve())
    if args.self_test:
        return self_test(data,runs)
    return run(args,ros_args,data,runs)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,KeyError) as exc:
        print(f'[ERROR] {exc}',file=sys.stderr)
        raise SystemExit(2)
