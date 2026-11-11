#!/usr/bin/env python3
"""Read-only memory consumer -> Nav2 Humble KeepoutFilter mask + diagnostics.

No motion publishers/action clients. No original map/coverage modification.
Outside the point disks is UNKNOWN (-1), NOT free: the Humble keepout filter
must not turn unknown source map cells into free cells.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import fcntl
import json
import math
import os
import sys
import time


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--session',type=Path,required=True,
        help='Explicit mission output directory whose memory this server consumes')
    parser.add_argument('--mode',choices=('observe','apply'),default='observe')
    parser.add_argument('--namespace',default='/j100_0001')
    opts,ros_args=parser.parse_known_args(argv)
    root=opts.project.expanduser().resolve();src=root/'scripts/exploration/zone_exploration/src'
    sys.path[:0]=[str(src),str(root/'scripts/exploration')]
    from reroute_core import (VERSION,RerouteSettings,PointStore,context_for,
        geometry_dict,json_hash,rasterize_points)
    from hospital_coverage_core import load_bundle,planar_pose
    from hospital_zone_visualizer import verify_map
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from rclpy.duration import Duration
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from tf2_ros import Buffer,TransformListener
    from nav_msgs.msg import OccupancyGrid
    from nav2_msgs.msg import CostmapFilterInfo
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker,MarkerArray
    config=json.loads((root/'config/exploration/navigation/dn012_reroute_v1_4.json').read_text())
    settings=RerouteSettings(**config['detector']).validate()
    ns='/'+opts.namespace.strip('/')
    scene=os.environ.get('DN012_STATIC_SCENE_ID',config['scene_id'])
    bundle=load_bundle(root/'config/exploration/navigation/hospital_zones.json',
        root/'config/exploration/navigation/hospital_coverage_reference.json.gz')
    geometry=geometry_dict(bundle[1]);context=context_for(root,scene,ns)
    session=opts.session.expanduser()
    if not session.is_absolute():session=root/session
    if not session.is_dir():raise ValueError('SESSION_DIRECTORY_NOT_FOUND')
    store=PointStore(session.resolve()/'memory/reroute_points.json',
        context,geometry,settings.max_records)

    class MaskNode(Node):
        def __init__(self):
            super().__init__('dn012_reroute_mask')
            if not self.get_parameter('use_sim_time').value:
                raise ValueError('USE_SIM_TIME_REQUIRED')
            latched=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL)
            live=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE)
            self.info_pub=self.create_publisher(CostmapFilterInfo,ns+'/dn012_reroute/filter_info',latched)
            self.mask_pub=self.create_publisher(OccupancyGrid,ns+'/dn012_reroute/mask',latched)
            self.status_pub=self.create_publisher(String,ns+'/dn012_reroute/status',10)
            self.marker_pub=self.create_publisher(MarkerArray,ns+'/dn012_reroute/markers',latched)
            self.map_sub=self.create_subscription(OccupancyGrid,ns+'/map',self.on_map,latched)
            self.cost_sub=self.create_subscription(OccupancyGrid,ns+'/global_costmap/costmap',self.on_cost,live)
            self.tf=Buffer(cache_time=Duration(seconds=10.),node=self)
            self.listener=TransformListener(self.tf,self)
            self.map_ok=False;self.map_fault=None;self.map_info=None
            self.mask=None;self.diag={};self.last_revision=None;self.last_sent_stamp=0.
            self.active=();self.initial_ids={r['id'] for r in store.data['records'] if r.get('enabled') and r.get('mask_eligible')}
            self.global_cost=None;self.cost_stamp=0.;self.cost_wall=0.
            self.last_reason=None;self.tick_count=0
            info=CostmapFilterInfo();info.header.frame_id='map';info.type=0
            info.filter_mask_topic=ns+'/dn012_reroute/mask';info.base=0.;info.multiplier=1.
            self.info_pub.publish(info)
            self.timer=self.create_timer(.5,self.tick)
            self.get_logger().info(f'[REROUTE_MASK_SERVER] mode={opts.mode}; scene={scene}; file={store.path}')

        def on_map(self,msg):
            try:
                ok,reason=verify_map(msg,bundle[0]['map'])
                if not ok:raise ValueError(reason)
                if msg.header.frame_id.lstrip('/')!='map':raise ValueError('WRONG_MAP_FRAME')
                if abs(msg.info.origin.orientation.z)>1e-8:raise ValueError('ROTATED_MAP_UNSUPPORTED')
                self.map_info=msg.info;self.map_ok=True
            except Exception as exc:
                self.map_ok=False;self.map_fault=str(exc)

        def on_cost(self,msg):
            try:
                g=bundle[1]
                if (msg.header.frame_id.lstrip('/')!='map' or msg.info.width!=g.width or
                    msg.info.height!=g.height or abs(msg.info.resolution-g.resolution)>1e-8 or
                    abs(msg.info.origin.position.x-g.origin[0])>1e-6 or
                    abs(msg.info.origin.position.y-g.origin[1])>1e-6 or
                    abs(msg.info.origin.orientation.z)>1e-8):return
                self.global_cost=np.asarray(msg.data,dtype=np.int16).reshape(g.height,g.width)
                self.cost_stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
                self.cost_wall=time.monotonic()
            except (ValueError,AttributeError):return

        def pose(self):
            tf=self.tf.lookup_transform('map','base_link',Time(),timeout=Duration(seconds=0.))
            stamp=tf.header.stamp.sec+tf.header.stamp.nanosec*1e-9
            now=self.get_clock().now().nanoseconds*1e-9
            if not -.1<=now-stamp<=2.:raise ValueError('POSE_TF_STALE')
            return planar_pose(tf.transform,math.radians(5))

        def publish_markers(self):
            arr=MarkerArray();clear=Marker();clear.action=Marker.DELETEALL;arr.markers.append(clear)
            for k,r in enumerate(store.data['records']):
                m=Marker();m.header.frame_id='map';m.header.stamp=self.get_clock().now().to_msg()
                m.ns='reroute_candidate_points';m.id=k;m.type=Marker.SPHERE;m.action=Marker.ADD
                m.pose.orientation.w=1.;m.pose.position.x=float(r['point_xy'][0]);m.pose.position.y=float(r['point_xy'][1]);m.pose.position.z=.15
                m.scale.x=m.scale.y=.20;m.scale.z=.20
                m.color.r=1.;m.color.g=.45;m.color.b=.05;m.color.a=.9
                arr.markers.append(m)
            self.marker_pub.publish(arr)

        def tick(self):
            reason='OK';ack=None;now=self.get_clock().now().nanoseconds*1e-9
            try:
                if not self.map_ok or self.map_fault:raise ValueError(self.map_fault or 'WAIT_VERIFIED_REFERENCE_MAP')
                pose=self.pose();store.load()
                records=store.data['records'] if opts.mode=='apply' else []
                mask,diag=rasterize_points(geometry,records,pose,settings,self.active)
                self.mask=mask;self.diag=diag
                if diag['revision']!=self.last_revision:
                    msg=OccupancyGrid();msg.header.frame_id='map';msg.header.stamp=self.get_clock().now().to_msg()
                    msg.info=self.map_info;msg.data=mask.ravel().tolist()
                    self.mask_pub.publish(msg);self.last_sent_stamp=now
                    self.last_revision=diag['revision'];self.active=tuple(diag['active_ids'])
                    self.get_logger().info(f'[REROUTE_MASK_PUBLISHED] active={len(self.active)} cells={diag["occupied_cells"]}')
                self.publish_markers()
                if self.global_cost is not None and time.monotonic()-self.cost_wall<4. and -.1<=now-self.cost_stamp<=5.:
                    if self.cost_stamp+1e-6>=self.last_sent_stamp and np.all(self.global_cost[mask==100]>=99):
                        ack=diag['revision']
                if diag['active_overlap']:reason='CURRENT_POSE_OVERLAPS_ACTIVE_MASK'
                elif self.initial_ids.intersection(diag['pending_near_robot']):reason='LOADED_MEMORY_NEAR_ROBOT_INSPECT_FIRST'
                elif diag['pending_near_robot']:reason='WAIT_POINT_DISTANCE_BEFORE_MASK_INSERTION'
                elif self.info_pub.get_subscription_count()<2 or self.mask_pub.get_subscription_count()<2:
                    reason='WAIT_BOTH_NAV2_KEEPOUT_SUBSCRIBERS'
                elif ack!=diag['revision']:reason='WAIT_GLOBAL_MEMORY_COSTMAP_ACK'
            except Exception as exc:
                reason=type(exc).__name__+':'+str(exc)
            status=dict(schema='dn012_reroute_mask_status_v1',version=VERSION,
                mode=opts.mode,context_hash=store.context_hash,memory_file=str(store.path.resolve()),
                ready=reason=='OK',reason=reason,
                ros_time_s=now,revision=self.last_revision,global_applied_revision=ack,
                memory_records=len(store.data['records']),
                active_ids=list(self.active),pending_near_robot=self.diag.get('pending_near_robot',[]),
                active_overlap=self.diag.get('active_overlap',[]),
                info_subscribers=self.info_pub.get_subscription_count(),mask_subscribers=self.mask_pub.get_subscription_count())
            m=String();m.data=json.dumps(status,allow_nan=False);self.status_pub.publish(m)
            if reason!=self.last_reason:
                self.get_logger().info('[REROUTE_MASK_STATUS] '+reason);self.last_reason=reason

    lock_key=hashlib.sha256((os.environ.get('ROS_DOMAIN_ID','0')+'|'+ns).encode()).hexdigest()[:16]
    server_lock=open('/tmp/dn012_reroute_mask_'+lock_key+'.lock','a')
    try:fcntl.flock(server_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        server_lock.close()
        raise RuntimeError('ANOTHER_REROUTE_MASK_SERVER_IS_RUNNING')
    rclpy.init(args=ros_args);node=None
    try:
        node=MaskNode();rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        server_lock.close()
    return 0

if __name__=='__main__':
    raise SystemExit(main())
