#!/usr/bin/env python3
"""Frontier goal selection (visualization only) for ROS 2 Humble.

Uses sibling frontier_detector.py from DN-009. Selects a *candidate* pose
in known, clearance-checked free space connected to the robot. This is NOT
Nav2 planner/costmap validation. It sends NO NavigateToPose goals and NO Twist.

Topics (defaults):
    IN  /map (nav_msgs/OccupancyGrid)
    OUT /frontier_goal_markers (visualization_msgs/MarkerArray)
    OUT /frontier_selected_goal (geometry_msgs/PoseArray, 0 or 1 poses)

Self tests run without ROS (with the sibling DN-009 detector available).
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import heapq
import math
import sys
from typing import Sequence

from frontier_detector import detect_frontiers, grid_to_world

SQRT2 = math.sqrt(2.0)
NEIGHBORS = ((-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
             (-1,-1,SQRT2),(-1,1,SQRT2),(1,-1,SQRT2),(1,1,SQRT2))


@dataclass(frozen=True)
class Candidate:
    group_index: int
    goal_idx: int
    boundary_idx: int
    path_m: float
    frontier_length_m: float
    safety_m: float
    score: float


@dataclass
class Selection:
    chosen: Candidate | None
    candidates: list[Candidate]
    clusters: int
    raw_cells: int
    reachable_cells: int
    start_idx: int | None


def world_to_grid(wx: float, wy: float, ox: float, oy: float,
                  yaw: float, res: float) -> tuple[float, float]:
    dx, dy = wx - ox, wy - oy
    c, s = math.cos(yaw), math.sin(yaw)
    return (c*dx+s*dy)/res, (-s*dx+c*dy)/res


def build_safe_mask(data: Sequence[int], width: int, height: int,
                    radius_cells: int, free_max: int = 20):
    """Conservative 8-neighbor distance from unknown/occupied/nonfree cells.

    Free cells must have > radius_cells grid steps to a blocked cell or border;
    this is a conservative *map-grid* footprint screen, not a polygon/costmap
    collision test. Border is also treated as impassable.
    """
    n = width*height
    blocked = [not (0 <= v <= free_max) for v in data]
    d = [10**9] * n
    q = deque()
    for i, b in enumerate(blocked):
        if b:
            d[i] = 0
            q.append(i)
    while q:
        i = q.popleft()
        x, y = i%width, i//width
        nxt = d[i] + 1
        for dx,dy,_ in NEIGHBORS:
            xx,yy = x+dx,y+dy
            if 0 <= xx < width and 0 <= yy < height:
                j = yy*width+xx
                if d[j] > nxt:
                    d[j] = nxt
                    q.append(j)
    safe = [False]*n
    for i,b in enumerate(blocked):
        if b: continue
        x,y = i%width,i//width
        # Require a buffer both from blocked cells and map edges.
        edge = min(x+1, y+1, width-x, height-y)
        safe[i] = d[i] > radius_cells and edge > radius_cells
    return safe, d


def nearest_safe_start(safe: Sequence[bool], width: int, height: int,
                       gx: float, gy: float, search_radius_cells: int) -> int | None:
    ix,iy = int(math.floor(gx)),int(math.floor(gy))
    best = None
    best_d2 = float('inf')
    for dy in range(-search_radius_cells,search_radius_cells+1):
        y=iy+dy
        if not 0<=y<height: continue
        for dx in range(-search_radius_cells,search_radius_cells+1):
            x=ix+dx
            if not 0<=x<width: continue
            if dx*dx+dy*dy > search_radius_cells**2: continue
            j=y*width+x
            if safe[j]:
                dist2=((x+0.5)-gx)**2+((y+0.5)-gy)**2
                if dist2<best_d2:
                    best,best_d2=j,dist2
    return best


def path_distances(safe: Sequence[bool], w: int, h: int,
                   start: int, resolution: float) -> list[float]:
    n=w*h
    cost=[math.inf]*n
    cost[start]=0.0
    heap=[(0.0,start)]
    while heap:
        distance,i=heapq.heappop(heap)
        if distance>cost[i]+1e-8: continue
        x,y=i%w,i//w
        for dx,dy,step in NEIGHBORS:
            nx,ny=x+dx,y+dy
            if not(0<=nx<w and 0<=ny<h):continue
            j=ny*w+nx
            if not safe[j]: continue
            # Prevent cutting diagonally between two unsafe cells.
            if dx and dy and (not safe[y*w+nx] or not safe[ny*w+x]):continue
            nd=distance+step*resolution
            if nd+1e-8 < cost[j]:
                cost[j]=nd
                heapq.heappush(heap,(nd,j))
    return cost


def choose_goals(data: Sequence[int], w: int, h: int, res: float,
                 robot_grid: tuple[float,float], *,
                 clearance_m: float=0.35,
                 offset_min_m: float=0.35,
                 offset_max_m: float=1.0,
                 min_goal_distance_m: float=0.75,
                 group_gain_weight: float=1.5,
                 path_weight: float=1.0,
                 max_free_cost: int=20) -> Selection:
    if w<=0 or h<=0 or len(data)!=w*h or res<=0:
        raise ValueError('invalid OccupancyGrid')
    if not (0 < clearance_m <= offset_max_m and 0 <= offset_min_m < offset_max_m):
        raise ValueError('invalid clearance or offset settings')
    groups,raw=detect_frontiers(data,w,h,resolution=res,max_free_cost=max_free_cost)
    if not groups:
        return Selection(None,[],0,raw,0,None)

    radius_cells=math.ceil(clearance_m/res-1e-9)
    safe,obstacle_d=build_safe_mask(data,w,h,radius_cells,free_max=max_free_cost)
    start=nearest_safe_start(safe,w,h,*robot_grid,max(1,math.ceil(0.60/res)))
    if start is None:
        return Selection(None,[],len(groups),raw,0,None)
    distances=path_distances(safe,w,h,start,res)
    n_reachable=sum(math.isfinite(v) for v in distances)

    r0=math.ceil(offset_min_m/res)
    r1=math.ceil(offset_max_m/res)
    # Sparse sampled offset grid for real-time use. Bounded candidate search.
    offsets=[]
    for dy in range(-r1,r1+1):
        for dx in range(-r1,r1+1):
            q=dx*dx+dy*dy
            if r0*r0<=q<=r1*r1:
                offsets.append((dx,dy,q))
    offsets.sort(key=lambda p:p[2])

    candidates=[]
    for gi,g in enumerate(groups):
        # Use multiple representatives, not cluster centroid (may lie in Unknown).
        stride=max(1,len(g.cells)//50)
        best=None
        checked=set()
        for boundary in g.cells[::stride]:
            bx,by=boundary%w,boundary//w
            for dx,dy,q in offsets:
                x,y=bx+dx,by+dy
                if not (0<=x<w and 0<=y<h):continue
                j=y*w+x
                if j in checked: continue
                checked.add(j)
                if not safe[j] or not math.isfinite(distances[j]):continue
                if distances[j] < min_goal_distance_m:continue
                # Limit the radius of goal search; travel route is map-reachable.
                sep=math.hypot((x+0.5)-robot_grid[0],(y+0.5)-robot_grid[1])*res
                if sep < min_goal_distance_m:continue
                length=len(g.cells)*res  # heuristic size, not guaranteed information gain
                score=(group_gain_weight*math.log1p(length)
                       -path_weight*distances[j]
                       -0.35*abs(math.sqrt(q)*res-0.55))
                candidate=Candidate(gi,j,boundary,distances[j],length,
                                    max(0.0,(obstacle_d[j]-1)*res),score)
                if best is None or candidate.score>best.score:
                    best=candidate
        if best is not None:
            candidates.append(best)
    candidates.sort(key=lambda c:(-c.score,c.path_m,c.goal_idx))
    return Selection(candidates[0] if candidates else None,candidates,
                     len(groups),raw,n_reachable,start)


def self_test():
    w,h=80,70
    data=[-1]*(w*h)
    # Single known free island with unknown frontier around.
    for y in range(15,55):
        for x in range(10,60):
            data[y*w+x]=0
    # Additional disconnected free island is never an acceptable target.
    for y in range(8,16):
        for x in range(65,75):
            data[y*w+x]=0
    result=choose_goals(data,w,h,0.1,(30.5,35.5),clearance_m=.3,
                        offset_max_m=.9)
    assert result.chosen is not None, result
    assert result.reachable_cells>0
    i=result.chosen.goal_idx
    assert data[i]==0
    assert i%w<60, 'unreachable island selected'
    assert result.chosen.path_m>=0.75
    assert result.chosen.safety_m>=0.2
    # Zero-frontier map: outputs must clear, not retain old goal.
    none=choose_goals([0]*(w*h),w,h,.1,(30.5,35.5))
    assert none.chosen is None and not none.candidates
    # Fully unknown map has no navigable frontier.
    none=choose_goals([-1]*(w*h),w,h,.1,(30.5,35.5))
    assert none.chosen is None
    # Starting in unknown/no safe connected free area must not invent a goal.
    none=choose_goals(data,w,h,.1,(2.5,2.5),clearance_m=.3)
    assert none.chosen is None
    # Rotated OccupancyGrid origin roundtrip.
    ox,oy,theta=2.7,-1.4,math.pi/3
    gx,gy=28.5,24.5
    wx,wy=grid_to_world(gx,gy,ox,oy,theta,.05)
    rx,ry=world_to_grid(wx,wy,ox,oy,theta,.05)
    assert abs(gx-rx)<1e-8 and abs(gy-ry)<1e-8
    print('SELF-TEST PASS: frontier group candidates, mapped-free clearance, '
          'reachable component, goal offset, no frontier, unknown-only map, '
          'robot not on map, rotated grid origin.')
    print('ROS2 TF, RViz and Nav2 global/local costmap feasibility are NOT tested.')


def run_ros(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
    from rclpy.time import Time
    from nav_msgs.msg import OccupancyGrid
    from geometry_msgs.msg import Point,PoseArray,Pose
    from visualization_msgs.msg import Marker,MarkerArray
    from tf2_ros import Buffer, TransformListener, TransformException

    latched=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                       history=QoSHistoryPolicy.KEEP_LAST,depth=1)

    class Selector(Node):
        def __init__(self):
            super().__init__('frontier_goal_selector')
            self.map=None
            self.buffer=Buffer()
            self.listener=TransformListener(self.buffer,self)
            self.sub=self.create_subscription(OccupancyGrid,args.map_topic,self.map_cb,latched)
            self.markers=self.create_publisher(MarkerArray,args.marker_topic,latched)
            self.poses=self.create_publisher(PoseArray,args.goal_topic,latched)
            self.timer=self.create_timer(2.0,self.evaluate)
            self.get_logger().info(f'[READY] subscribe {args.map_topic}; marker={args.marker_topic} '
                                   f'candidate_pose_array={args.goal_topic}; frame={args.robot_frame}; '
                                   'visualization ONLY; NO Nav2 goals are sent')
        def map_cb(self,msg):
            self.map=msg
        def evaluate(self):
            msg=self.map
            if msg is None:
                self.get_logger().info('[WAIT_MAP] no SLAM OccupancyGrid yet')
                return
            w,h=int(msg.info.width),int(msg.info.height)
            res=float(msg.info.resolution)
            frame=msg.header.frame_id or 'map'
            if w*h!=len(msg.data) or w==0 or h==0 or res<=0:
                self.get_logger().warn('[MAP_INVALID] waiting for valid grid')
                self.clear(frame,msg.header.stamp)
                return
            try:
                tf=self.buffer.lookup_transform(frame,args.robot_frame,Time())
            except TransformException as exc:
                self.get_logger().warn(f'[WAIT_TF] {frame} <- {args.robot_frame}: {exc}')
                self.clear(frame,msg.header.stamp)
                return
            tr=tf.transform.translation
            q=msg.info.origin.orientation
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            ox,oy=msg.info.origin.position.x,msg.info.origin.position.y
            robot_grid=world_to_grid(tr.x,tr.y,ox,oy,yaw,res)
            try:
                selection=choose_goals(msg.data,w,h,res,robot_grid,
                                       clearance_m=args.clearance_m,
                                       offset_max_m=args.offset_max_m,
                                       min_goal_distance_m=args.min_goal_distance_m)
            except ValueError as exc:
                self.get_logger().error(f'[MAP_INVALID] {exc}')
                self.clear(frame,msg.header.stamp)
                return
            clear=Marker()
            clear.header.frame_id=frame
            clear.header.stamp=msg.header.stamp
            clear.action=Marker.DELETEALL
            markers=MarkerArray()
            markers.markers.append(clear)
            pa=PoseArray()
            pa.header.frame_id=frame
            pa.header.stamp=msg.header.stamp
            if selection.candidates:
                # Show one safely projected free-space candidate per reachable group.
                options=Marker()
                options.header=pa.header
                options.ns='reachable_frontier_candidates'
                options.id=1
                options.type=Marker.SPHERE_LIST
                options.action=Marker.ADD
                options.pose.orientation.w=1.0
                options.scale.x=options.scale.y=options.scale.z=.17
                options.color.r=.1
                options.color.g=.7
                options.color.b=1.0
                options.color.a=.9
                for c in selection.candidates:
                    p=Point()
                    p.x,p.y=grid_to_world(c.goal_idx%w+.5,c.goal_idx//w+.5,ox,oy,yaw,res)
                    p.z=.14
                    options.points.append(p)
                markers.markers.append(options)
            if selection.chosen:
                c=selection.chosen
                gx,gy=grid_to_world(c.goal_idx%w+.5,c.goal_idx//w+.5,ox,oy,yaw,res)
                fx,fy=grid_to_world(c.boundary_idx%w+.5,c.boundary_idx//w+.5,ox,oy,yaw,res)
                ang=math.atan2(fy-gy,fx-gx)
                pose=Pose()
                pose.position.x,pose.position.y,pose.position.z=gx,gy,0.0
                pose.orientation.z=math.sin(ang/2)
                pose.orientation.w=math.cos(ang/2)
                pa.poses.append(pose)
                arrow=Marker()
                arrow.header=pa.header
                arrow.ns='selected_frontier_goal'
                arrow.id=2
                arrow.type=Marker.ARROW
                arrow.action=Marker.ADD
                arrow.pose=pose
                arrow.pose.position.z=.15
                arrow.scale.x=.50
                arrow.scale.y=.13
                arrow.scale.z=.16
                arrow.color.r=1.0
                arrow.color.g=.25
                arrow.color.b=.15
                arrow.color.a=1.0
                markers.markers.append(arrow)
                label=Marker()
                label.header=pa.header
                label.ns='selected_frontier_goal'
                label.id=3
                label.type=Marker.TEXT_VIEW_FACING
                label.action=Marker.ADD
                label.pose.position.x=gx
                label.pose.position.y=gy
                label.pose.position.z=.70
                label.pose.orientation.w=1.0
                label.scale.z=.28
                label.color.r=label.color.g=label.color.b=1.0
                label.color.a=1.0
                label.text=f'SELECTED path~{c.path_m:.1f}m score={c.score:.1f}'
                markers.markers.append(label)
                self.get_logger().info(f'[SELECTED] frontier_groups={selection.clusters} '
                                       f'reachable_groups={len(selection.candidates)} '
                                       f'goal=({gx:.2f},{gy:.2f}) path_est={c.path_m:.2f}m '
                                       f'heuristic_score={c.score:.2f} '
                                       'NO navigation command')
            else:
                self.get_logger().info(f'[NO_GOAL] groups={selection.clusters} '
                                       f'reachable_groups={len(selection.candidates)} '
                                       f'reachable_cells={selection.reachable_cells}; '
                                       'no safe connected goal candidate')
            self.markers.publish(markers)
            self.poses.publish(pa)
        def clear(self,frame,stamp):
            ma=MarkerArray()
            m=Marker()
            m.action=Marker.DELETEALL
            m.header.frame_id=frame
            m.header.stamp=stamp
            ma.markers.append(m)
            self.markers.publish(ma)
            pa=PoseArray()
            pa.header.frame_id=frame
            pa.header.stamp=stamp
            self.poses.publish(pa)

    rclpy.init(args=sys.argv)
    node=Selector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


def main():
    ap=argparse.ArgumentParser(description='DN-010: select free-space candidate goals; no Nav2 commands')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--map-topic',default='/map')
    ap.add_argument('--marker-topic',default='/frontier_goal_markers')
    ap.add_argument('--goal-topic',default='/frontier_selected_goal')
    ap.add_argument('--robot-frame',default='base_link')
    ap.add_argument('--clearance-m',type=float,default=.35)
    ap.add_argument('--offset-max-m',type=float,default=1.0)
    ap.add_argument('--min-goal-distance-m',type=float,default=.75)
    # accept ROS2 remapping parameters after --ros-args; rclpy parses them
    args,other=ap.parse_known_args()
    if other and other[0]!='--ros-args':
        ap.error(f'unrecognized arguments: {other}')
    if args.self_test:
        self_test()
    else:
        run_ros(args)


if __name__=='__main__':
    main()
