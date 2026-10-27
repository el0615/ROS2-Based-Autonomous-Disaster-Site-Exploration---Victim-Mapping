#!/usr/bin/env python3
"""Frontier detector for live ROS2 OccupancyGrid maps (visualization only).

DN-009 scope: detect free/unknown boundaries, cluster cells, visualize on RViz.
Does not choose goals, command velocities, or call Nav2 actions.

No third-party Python modules are required beyond ROS2 Humble runtime.
The pure-Python self-test runs even on computers without ROS2 installed.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class FrontierGroup:
    cells: tuple[int, ...]  # row-major OccupancyGrid indices, all observed free cells
    center_ix: float       # visualization only; NOT a navigation goal
    center_iy: float


def detect_frontiers(data: Sequence[int], width: int, height: int,
                     *, max_free_cost: int = 20, min_cells: int = 8,
                     resolution: float = 0.05,
                     min_length_m: float = 0.5,
                     support_radius_m: float = 0.15,
                     min_free_support: int = 9,
                     min_unknown_support: int = 10,
                     obstacle_clearance_m: float = 0.15,
                     stats: dict | None = None) -> tuple[list[FrontierGroup], int]:
    """Find frontier groups: free cells 4-adjacent to an in-bounds unknown cell.

    Use 8-connectivity to group frontier *candidate* cells. Reject groups with
    too few cells / physical length. Heuristically reject isolated laser ray
    cells using local free/unknown density and occupied-cell clearance.
    This is a visualization-only heuristic, NOT a guarantee of robot reachability.
    Returns (groups, raw free/unknown boundary count BEFORE density filtering).
    """
    if width <= 0 or height <= 0 or resolution <= 0 or len(data) != width * height:
        raise ValueError('invalid OccupancyGrid dimensions, resolution, or data length')
    if (min_cells < 1 or min_length_m < 0 or support_radius_m < 0
            or obstacle_clearance_m < 0 or min_free_support < 0
            or min_unknown_support < 0):
        raise ValueError('invalid frontier filtering thresholds')

    candidates: set[int] = set()
    for i, v in enumerate(data):
        if not (0 <= v <= max_free_cost):
            continue
        x = i % width
        y = i // width
        if ((x > 0 and data[i - 1] == -1)
                or (x + 1 < width and data[i + 1] == -1)
                or (y > 0 and data[i - width] == -1)
                or (y + 1 < height and data[i + width] == -1)):
            candidates.add(i)

    total_candidates = len(candidates)

    # A true open frontier normally has a 2D patch of known free space behind
    # it and unknown space ahead. A thin, 1-cell laser ray often does not.
    # Integral images give constant-time rectangular support counts per cell.
    radius = int(math.ceil(support_radius_m / resolution - 1e-12))
    stride = width + 1
    free_sum = [0] * ((height + 1) * stride)
    unknown_sum = [0] * ((height + 1) * stride)
    for y in range(height):
        fr = un = 0
        for x in range(width):
            v = data[y * width + x]
            fr += int(0 <= v <= max_free_cost)
            un += int(v == -1)
            dst = (y + 1) * stride + x + 1
            free_sum[dst] = free_sum[dst - stride] + fr
            unknown_sum[dst] = unknown_sum[dst - stride] + un

    def rectangle_count(table, x0, y0, x1, y1):
        return (table[y1 * stride + x1] - table[y0 * stride + x1]
                - table[y1 * stride + x0] + table[y0 * stride + x0])

    clearance_r = int(math.ceil(obstacle_clearance_m / resolution - 1e-12))
    filtered = set()
    for i in candidates:
        x, y = i % width, i // width
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if rectangle_count(free_sum, x0, y0, x1, y1) < min_free_support:
            continue
        if rectangle_count(unknown_sum, x0, y0, x1, y1) < min_unknown_support:
            continue
        # Test occupied cells (>=65), not Unknown cells. Frontier must touch
        # Unknown by definition; Unknown should not be treated as an obstacle.
        # Goal feasibility and footprint checking remain tasks for DN-010.
        has_obstacle = False
        for dy in range(-clearance_r, clearance_r + 1):
            if has_obstacle:
                break
            ny = y + dy
            if ny < 0 or ny >= height:
                continue
            for dx in range(-clearance_r, clearance_r + 1):
                if dx * dx + dy * dy > clearance_r * clearance_r:
                    continue
                nx = x + dx
                if 0 <= nx < width and data[ny * width + nx] >= 65:
                    has_obstacle = True
                    break
        if not has_obstacle:
            filtered.add(i)
    if stats is not None:
        stats['density_clearance_cells'] = len(filtered)
    candidates = filtered
    groups: list[FrontierGroup] = []
    # Avoid inventing a frontier outside the grid; only interior -1 neighbors count.
    while candidates:
        start = candidates.pop()
        q = deque([start])
        component = [start]
        while q:
            pos = q.popleft()
            x = pos % width
            y = pos // width
            for dy in (-1, 0, 1):
                ny = y + dy
                if ny < 0 or ny >= height:
                    continue
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    if nx < 0 or nx >= width:
                        continue
                    idx = ny * width + nx
                    if idx in candidates:
                        candidates.remove(idx)
                        component.append(idx)
                        q.append(idx)
        # Approximate physical extent by number of connected boundary cells.
        if len(component) < min_cells or len(component) * resolution < min_length_m:
            continue
        cx = sum((i % width) + 0.5 for i in component) / len(component)
        cy = sum((i // width) + 0.5 for i in component) / len(component)
        groups.append(FrontierGroup(tuple(component), cx, cy))

    groups.sort(key=lambda g: len(g.cells), reverse=True)
    return groups, total_candidates


def grid_to_world(gx: float, gy: float, origin_x: float, origin_y: float,
                  origin_yaw: float, resolution: float) -> tuple[float, float]:
    lx = gx * resolution
    ly = gy * resolution
    c, s = math.cos(origin_yaw), math.sin(origin_yaw)
    return origin_x + c * lx - s * ly, origin_y + s * lx + c * ly


def run_self_test() -> None:
    # A 3x3 visible patch within unknown space has an 8-cell frontier ring.
    w = h = 7
    grid = [-1] * (w * h)
    for y in range(2, 5):
        for x in range(2, 5):
            grid[y * w + x] = 0
    groups, raw = detect_frontiers(grid, w, h, min_cells=8, min_length_m=0.4)
    assert raw == 8, (raw, groups)
    assert len(groups) == 1 and len(groups[0].cells) == 8
    assert abs(groups[0].center_ix - 3.5) < 1e-9
    assert abs(groups[0].center_iy - 3.5) < 1e-9
    # Fully known map borders do not count as unknown space.
    groups, raw = detect_frontiers([0] * 49, w, h, min_cells=1, min_length_m=0)
    assert raw == 0 and groups == []
    # Occupied and unknown cells themselves must never be output as frontier.
    grid[2 * w + 2] = 100
    groups, _ = detect_frontiers(grid, w, h, min_cells=1, min_length_m=0)
    assert all(0 <= grid[i] <= 20 for g in groups for i in g.cells)
    # Check 90-degree non-axis-aligned map origin transform.
    wx, wy = grid_to_world(1, 0, 2, 3, math.pi / 2, .5)
    assert math.isclose(wx, 2.0, abs_tol=1e-9)
    assert math.isclose(wy, 3.5, abs_tol=1e-9)
    # Small frontier filter must suppress a one-cell boundary.
    tiny = [-1] * 9
    tiny[4] = 0
    groups, raw = detect_frontiers(tiny, 3, 3, min_cells=2, min_length_m=0)
    assert raw == 1 and not groups
    # A one-cell wide free ray into Unknown should be suppressed by density
    # whereas a broad free patch of the same length should survive.
    w, h = 45, 33
    ray = [-1] * (w * h)
    for x in range(5, 36):
        ray[16 * w + x] = 0
    ray_groups, ray_count = detect_frontiers(ray, w, h)
    assert ray_count > 20 and not ray_groups, (ray_count, ray_groups)
    patch = [-1] * (w * h)
    for y in range(10, 23):
        for x in range(5, 36):
            patch[y * w + x] = 0
    patch_groups, patch_count = detect_frontiers(patch, w, h)
    assert patch_count > 20 and patch_groups, (patch_count, patch_groups)
    # Obstacle clearance removes a free/unknown boundary beside a wall, while
    # leaving the opposite side of a broad patch discoverable.
    for y in range(11, 22):
        patch[y * w + 5] = 100
    with_obstacle, _ = detect_frontiers(patch, w, h)
    assert with_obstacle and all(0 <= patch[i] <= 20 for g in with_obstacle for i in g.cells)
    print('SELF-TEST PASS: free/unknown boundary, 8-connected clustering, '
          'ray suppression, broad-frontier retention, obstacle clearance, '
          'no border false positives, rotated map origin.')
    print('ROS2 subscriptions, TF, MarkerArray display and live SLAM are not tested here.')


def run_ros(args: argparse.Namespace) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
    from geometry_msgs.msg import Point
    from nav_msgs.msg import OccupancyGrid
    from visualization_msgs.msg import Marker, MarkerArray

    reliable_latched = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )

    class FrontierDetector(Node):
        def __init__(self):
            super().__init__('frontier_detector')
            self.publisher = self.create_publisher(MarkerArray, args.marker_topic, reliable_latched)
            self.subscription = self.create_subscription(
                OccupancyGrid, args.map_topic, self.on_map, reliable_latched)
            self.get_logger().info(
                f'[READY] SLAM map={args.map_topic}  RViz markers={args.marker_topic} '
                f'min_cells={args.min_cells} min_length_m={args.min_length_m:.2f} '
                f'free_support>={args.min_free_support} unknown_support>={args.min_unknown_support}; ' 'visualization ONLY (no automatic navigation)')

        def on_map(self, msg: OccupancyGrid):
            width = int(msg.info.width)
            height = int(msg.info.height)
            res = float(msg.info.resolution)
            if not msg.data or width <= 0 or height <= 0 or res <= 0:
                self.get_logger().warn('Empty OccupancyGrid; waiting for SLAM map.')
                return
            try:
                stats = {}
                groups, raw_count = detect_frontiers(
                    msg.data, width, height,
                    max_free_cost=args.max_free_cost,
                    min_cells=args.min_cells,
                    resolution=res,
                    min_length_m=args.min_length_m,
                    support_radius_m=args.support_radius_m,
                    min_free_support=args.min_free_support,
                    min_unknown_support=args.min_unknown_support,
                    obstacle_clearance_m=args.obstacle_clearance_m,
                    stats=stats)
            except ValueError as exc:
                self.get_logger().error(f'Invalid map: {exc}')
                return

            origin = msg.info.origin
            q = origin.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            ox, oy = origin.position.x, origin.position.y
            frame = msg.header.frame_id or 'map'
            stamp = msg.header.stamp

            def point_at(gx, gy, z=0.07):
                wx, wy = grid_to_world(gx, gy, ox, oy, yaw, res)
                point = Point()
                point.x, point.y, point.z = wx, wy, z
                return point

            array = MarkerArray()
            clear = Marker()
            clear.header.frame_id = frame
            clear.header.stamp = stamp
            clear.action = Marker.DELETEALL
            array.markers.append(clear)

            if groups:
                cells = Marker()
                cells.header.frame_id = frame
                cells.header.stamp = stamp
                cells.ns = 'frontier_cells'
                cells.id = 1
                cells.type = Marker.POINTS
                cells.action = Marker.ADD
                cells.pose.orientation.w = 1.0
                cells.scale.x = max(0.03, res * 0.85)
                cells.scale.y = max(0.03, res * 0.85)
                cells.color.r = 0.10
                cells.color.g = 0.95
                cells.color.b = 0.35
                cells.color.a = 0.95
                for group in groups:
                    for i in group.cells:
                        cells.points.append(point_at((i % width) + .5, (i // width) + .5))
                array.markers.append(cells)

                # Cluster centers are drawn for debugging. They are NOT valid Nav2 goals.
                centers = Marker()
                centers.header.frame_id = frame
                centers.header.stamp = stamp
                centers.ns = 'frontier_cluster_centers'
                centers.id = 2
                centers.type = Marker.SPHERE_LIST
                centers.action = Marker.ADD
                centers.pose.orientation.w = 1.0
                centers.scale.x = centers.scale.y = centers.scale.z = 0.15
                centers.color.r = 1.0
                centers.color.g = 0.85
                centers.color.b = 0.05
                centers.color.a = 1.0
                centers.points = [point_at(g.center_ix, g.center_iy, 0.12) for g in groups]
                array.markers.append(centers)

            self.publisher.publish(array)
            count_unknown = sum(v == -1 for v in msg.data)
            count_valid = sum(len(g.cells) for g in groups)
            self.get_logger().info(
                f'[FRONTIER] map={width}x{height} @ {res:.3f}m '
                f'unknown={count_unknown} candidate_cells={raw_count} '
                f'supported_cells={stats.get("density_clearance_cells", 0)} '
                f'valid_groups={len(groups)} valid_cells={count_valid}')

    rclpy.init()
    node = None
    try:
        node = FrontierDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(description='Visualize map frontiers; does not control the robot')
    ap.add_argument('--self-test', action='store_true', help='run pure-Python unit tests without ROS')
    ap.add_argument('--map-topic', default='/map')
    ap.add_argument('--marker-topic', default='/frontier_markers')
    ap.add_argument('--min-cells', type=int, default=8)
    ap.add_argument('--min-length-m', type=float, default=0.5)
    ap.add_argument('--max-free-cost', type=int, default=20)
    ap.add_argument('--support-radius-m', type=float, default=0.15,
                    help='local square support window radius, meters')
    ap.add_argument('--min-free-support', type=int, default=9,
                    help='minimum free cells inside local support window')
    ap.add_argument('--min-unknown-support', type=int, default=10,
                    help='minimum unknown cells inside local support window')
    ap.add_argument('--obstacle-clearance-m', type=float, default=0.15,
                    help='reject frontier points near occupied cells (meters)')
    args = ap.parse_args()
    if args.self_test:
        run_self_test()
    else:
        run_ros(args)


if __name__ == '__main__':
    main()
