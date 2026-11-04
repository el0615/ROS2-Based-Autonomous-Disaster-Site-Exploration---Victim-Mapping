#!/usr/bin/env python3
"""Independent Zone-Based Exploration v1 for the existing 27-zone J100 project.

ENTRY proposals are internal transit goals independent of predicted gain. Local
observation catalogs are computed while navigating, cached, then reranked with
fresh measured Coverage. Each executable goal requires Navfn action success,
current map and costmap, minimal path/endpoint checks, zone-internal goal with
clear goal footprint, fresh stop/coverage and one owned Nav2 action.
The extra full-path STRICT_SWEPT check is deliberately NOT used: Nav2 owns
path following, Costmap and dynamic collision prediction. A Navfn path is not
a guarantee of safe motion. Supervised Gazebo pilot only.
"""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid
import numpy as np
from hospital_coverage_core import Geometry, load_bundle, planar_pose
from hospital_zone_visualizer import verify_map
from zb_io import geometry_matches, stamp_s, atomic_json
from zb_reference_planning import (Settings, SensorModel, array_fingerprint,
    current_input_error, validate_coverage, oriented_footprint_clear)
from zb_reference_sequence import (StopEvidence,
    validate_nav_path, safe_early_stop, json_candidate)
from zone_policy import ZoneMission, room_priority, angle_diff
from costmap_diagnostics import snapshot_costmap_freshness
from zone_planner import ZonePlanner, refresh as refresh_candidate, is_inside, arrival_diagnostic


ACTIVE = {1, 2, 3}
STATUS = {0:'UNKNOWN',1:'ACCEPTED',2:'EXECUTING',3:'CANCELING',
          4:'SUCCEEDED',5:'CANCELED',6:'ABORTED'}


def parser():
    root=Path(os.environ['ZB_PROJECT_ROOT'])
    p=argparse.ArgumentParser(description='Zone-first exploration. Default DRY RUN; --execute --supervised may move Gazebo J100.')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--supervised',action='store_true',help='Explicit acknowledgment for supervised Gazebo navigation')
    p.add_argument('--max-goals',type=int,default=6,help='Includes both entry and observation Goals; NOT number of rooms')
    p.add_argument('--zone-ids',nargs='+',help='Optional subset; default all approved zones, no hardcoded whitelist')
    p.add_argument('--threshold',type=float,default=.90)
    p.add_argument('--max-views-per-zone',type=int,default=4)
    p.add_argument('--zone-policy',choices=['rooms-first','nearest'],default='rooms-first')
    p.add_argument('--entry-candidates',type=int,default=8)
    p.add_argument('--entry-inset-m',type=float,default=.45)
    p.add_argument('--entry-arrival-buffer-m',type=float,default=.20,help='Prefer deeper ENTRY targets (meters); no safety guard change')
    p.add_argument('--max-entry-arrival-retries',type=int,default=1,help='Bounded extra ENTRY goal after real pose/footprint outside zone')
    p.add_argument('--max-candidates',type=int,default=15,help='Local observation positions, not global zone count')
    p.add_argument('--heading-samples',type=int,default=4)
    p.add_argument('--planning-budget-s',type=float,default=15.)
    p.add_argument('--visibility-backend',choices=['auto','python','c'],default='auto')
    p.add_argument('--max-revisits',type=int,default=1)
    p.add_argument('--revisit-cooldown-s',type=float,default=30.)
    p.add_argument('--continue-on-nav-abort',action='store_true',help='Opt-in: only after terminal ABORTED, fresh stop/coverage and clear local footprint; otherwise stop')
    p.add_argument('--max-plan-attempts',type=int,default=60)
    p.add_argument('--namespace',default='/j100_0001')
    p.add_argument('--zones',type=Path,default=root/'config/exploration/hospital_zones.json')
    p.add_argument('--reference',type=Path,default=root/'config/exploration/hospital_coverage_reference.json.gz')
    p.add_argument('--map-topic',default='/j100_0001/map')
    p.add_argument('--coverage-grid-topic',default='/hospital_coverage/grid')
    p.add_argument('--coverage-status-topic',default='/hospital_coverage/status')
    p.add_argument('--costmap-topic',default='/j100_0001/global_costmap/costmap')
    p.add_argument('--local-costmap-topic',default='/j100_0001/local_costmap/costmap')
    p.add_argument('--scan-topic',default='/j100_0001/sensors/lidar2d_0/scan')
    p.add_argument('--odom-topic',default='/j100_0001/platform/odom/filtered')
    p.add_argument('--robot-frame',default='base_link')
    p.add_argument('--planner-id',default='GridBased')
    p.add_argument('--entrance-nav-only',action='store_true',help='Reuse approved planner-only exterior entrance overlay, never alter Coverage')
    p.add_argument('--entrance-overlay-file',type=Path,default=root/'config/exploration/entrance_nav_only_overlay_v1.json')
    p.add_argument('--output-dir',type=Path,default=root/'logs/hospital_zone_mission')
    return p


def make_node_type(args,bundle,cfg,output):
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from rclpy.duration import Duration
    from rclpy.action import ActionClient
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy,qos_profile_sensor_data
    from nav_msgs.msg import OccupancyGrid, Odometry
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from action_msgs.msg import GoalStatusArray
    from action_msgs.srv import CancelGoal
    from unique_identifier_msgs.msg import UUID
    from nav2_msgs.action import ComputePathToPose, NavigateToPose
    from geometry_msgs.msg import Point, PoseStamped
    from visualization_msgs.msg import Marker,MarkerArray
    from tf2_ros import Buffer,TransformListener,TransformException
    data,geom,labels,known_free,signature=bundle
    scope=[z for z in data['zones'] if args.zone_ids is None or z['id'] in args.zone_ids]
    ns='/'+args.namespace.strip('/') if args.namespace.strip('/') else ''

    class HospitalSequence(Node):
        def __init__(self):
            super().__init__('hospital_zone_exploration')
            self.book=ZoneMission(scope,cfg.threshold,args.max_views_per_zone,
                                  max_revisits=args.max_revisits,
                                  revisit_cooldown_s=args.revisit_cooldown_s)
            self.zones_by_id={z['id']:z for z in scope}
            self.zone_planner=ZonePlanner(bundle,cfg,entry_candidates=args.entry_candidates,
                entry_inset=args.entry_inset_m,policy=args.zone_policy,
                local_candidates=args.max_candidates,heading_samples=args.heading_samples,
                budget_s=args.planning_budget_s,backend=args.visibility_backend,
                entrance_overlay=args.entrance_overlay_file if args.entrance_nav_only else None,
                entry_arrival_buffer=args.entry_arrival_buffer_m)
            self.batch_zone=None;self.batch_kind=None
            self.prefetch_for=None;self.phase='SELECT_ZONE'
            self.options_context=None
            self.stop_evidence=StopEvidence()
            self._stop_confirmed_logged_goal_id=None  # one STOP_CONFIRMED per unique NavigateToPose Goal
            self.reference_ok=False;self.fault=None;self.session_id=None;self.coverage_epoch=0
            self.coverage_reports=deque(maxlen=12);self.coverage_grids=deque(maxlen=12);self.matched=None
            self.costmap_entry=None;self.local_entry=None;self.scan_entry=None;self.odom_entry=None
            # Diagnostic-only counters; no change to the safety freshness predicate or QoS.
            self.costmap_monitor=dict(received_count=0,accepted_count=0,
                invalid_geometry_count=0,invalid_values_count=0,
                last_receive_wall=None,last_valid_receive_wall=None,
                last_header_stamp_s=None,last_interarrival_wall_s=None,max_interarrival_wall_s=0.0)
            self.costmap_stale_episode=False
            self.last_clock=None;self.clock_advanced_wall=time.monotonic()
            self.log_times={};self.state='WAIT_INPUT';self.last_notice=None
            self.stop_requested=None;self.stop_started=None;self.done=False;self.exit_code=0
            self.pool=ThreadPoolExecutor(max_workers=1);self.worker=None;self.job_context=None
            self.options=[];self.current=None;self.current_path=[];self.current_path_length=None
            self.validated_plan_mode=None
            self.plan_send=None;self.plan_handle=None;self.plan_result=None;self.plan_started=0.
            self.nav_send=None;self.nav_handle=None;self.nav_result=None;self.nav_uuid=None
            self.nav_started=0.;self.cancel_future=None;self.cancel_reason=None;self.cancel_started=None
            self.nav_terminal=None;self.post_result_ros=None;self.post_result_scans=None;self.post_result_wall=None
            self.before_fraction=None;self.before_overall=None;self.owned_ids=set();self.nav_statuses={}
            self.sent_goals=0;self.plan_attempts=0;self.succeeded_goals=0;self.coverage_cancels=0
            self.plan_snapshot=None;self.bad_since=None;self.next_report=0.;self.boot_wall=time.monotonic()
            self.tf_buffer=Buffer(cache_time=Duration(seconds=30.),node=self)
            self.tf_listener=TransformListener(self.tf_buffer,self)
            latched=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
            live=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
            action_qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.map_sub=self.create_subscription(OccupancyGrid,args.map_topic,self.on_map,latched)
            self.grid_sub=self.create_subscription(OccupancyGrid,args.coverage_grid_topic,self.on_grid,latched)
            self.coverage_sub=self.create_subscription(String,args.coverage_status_topic,self.on_status,latched)
            self.cost_sub=self.create_subscription(OccupancyGrid,args.costmap_topic,self.on_costmap,live)
            self.local_sub=self.create_subscription(OccupancyGrid,args.local_costmap_topic,self.on_local_costmap,live)
            self.scan_sub=self.create_subscription(LaserScan,args.scan_topic,self.on_scan,qos_profile_sensor_data)
            self.odom_sub=self.create_subscription(Odometry,args.odom_topic,self.on_odom,qos_profile_sensor_data)
            self.action_status_sub=self.create_subscription(GoalStatusArray,ns+'/navigate_to_pose/_action/status',self.on_action_status,action_qos)
            self.marker_pub=self.create_publisher(MarkerArray,'/hospital_zone_mission/markers',latched)
            self.status_pub=self.create_publisher(String,'/hospital_zone_mission/status',latched)
            self.stop_service=self.create_service(Trigger,'/hospital_zone_mission/stop',self.on_stop_service)
            self.plan_client=ActionClient(self,ComputePathToPose,ns+'/compute_path_to_pose')
            self.nav_client=ActionClient(self,NavigateToPose,ns+'/navigate_to_pose')
            self.exact_cancel_client=self.create_client(CancelGoal,ns+'/navigate_to_pose/_action/cancel_goal')
            self.events_handle=(output/'events.jsonl').open('x',encoding='utf-8')
            self.event('READY',implementation='zone_based_nav2_delegated_v2_4_start_bridge',mode='EXECUTE' if args.execute else 'DRY_RUN',max_goals=args.max_goals,
                       zones=len(scope),threshold=cfg.threshold,early_cancel=False)
            self.info('mode','[READY] '+('--execute: ROBOT MAY MOVE' if args.execute else 'DRY RUN: no NavigateToPose')+
                      f'; {len(scope)} zones; max_goals={args.max_goals}; early_cancel={False}',0)
            self.info('policy','[POLICY] zone-first ENTRY then local OBSERVE; measured coverage only; one owned Goal; Navfn path + safe zone-internal endpoint; Nav2 handles collision avoidance; abort stops by default',0)
            self.info('arrival_policy',f'[ENTRY_ARRIVAL_V2_3] interior_buffer={args.entry_arrival_buffer_m:.2f}m; bounded_retry={args.max_entry_arrival_retries}; strict measured footprint unchanged',0)
            self.info('nav2_delegate','[NAV2_DELEGATION_V2] no custom full-path STRICT_SWEPT; Navfn path sanity and zone-internal goal checks remain; Nav2 Costmap/RPP execute',0)
            self.info('start_bridge_policy','[START_BRIDGE_V2_4] conservative bridge used for ranking only; no bridge -> Navfn endpoint probe; local-start footprint, Navfn and target checks unchanged',0)
            self.info('costmap_diag_policy','[COSTMAP_DIAG_V2_2] receive/header/age diagnostics only; 5s ROS / 15s wall / 2s cancel gating UNCHANGED',0)
            self.info('zone_policy',f'[ZONE_POLICY] {args.zone_policy}; threshold={cfg.threshold}; entry_inset={args.entry_inset_m}m; local_candidates={args.max_candidates}; no permanent unreachable claims',0)
            self.info('logs',f'[LOG] {output}',0)

        def ros_s(self):return self.get_clock().now().nanoseconds*1e-9

        def info(self,key,text,period=5.):
            wall=time.monotonic()
            if wall-self.log_times.get(key,-math.inf)>=period:
                self.get_logger().info(text);self.log_times[key]=wall

        def warn(self,key,text,period=5.):
            wall=time.monotonic()
            if wall-self.log_times.get(key,-math.inf)>=period:
                self.get_logger().warning(text);self.log_times[key]=wall

        def event(self,event,**fields):
            row=dict(event=event,ros_time_s=self.ros_s(),wall_utc=datetime.now(timezone.utc).isoformat(),**fields)
            self.events_handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.events_handle.flush()

        def transition(self,state):
            if state!=self.state:
                self.state=state;self.info('state',f'[STATE] {state}; sent={self.sent_goals}/{args.max_goals}',0)
                self.event('STATE',state=state)

        def on_map(self,msg):
            try:ok,reason=verify_map(msg,data['map'])
            except (ValueError,TypeError,AttributeError,OverflowError) as exc:ok,reason=False,str(exc)
            if self.reference_ok and not ok:self.fault='REFERENCE_MAP_CHANGED'
            if ok and not self.reference_ok:self.info('map','[MAP_OK] approved map fingerprint matches',0)
            self.reference_ok=bool(ok) and self.fault is None
            if not ok:self.warn('map_bad','[MAP_MISMATCH] '+reason)

        def on_status(self,msg):
            try:
                s=json.loads(msg.data)
                if s.get('schema')!='hospital_room_coverage_v1' or s.get('zone_bundle_signature')!=signature:
                    raise ValueError('Coverage schema/bundle mismatch')
                if not isinstance(s.get('session_id'),str) or not s['session_id']:raise ValueError('Invalid session')
                stamp=float(s['ros_time_s'])
                if not math.isfinite(stamp):raise ValueError('Nonfinite timestamp')
            except (ValueError,KeyError,TypeError) as exc:
                self.matched=None;self.coverage_reports.clear();self.coverage_grids.clear()
                self.warn('coverage_bad',f'[BAD_COVERAGE] {exc}');return
            if self.session_id is not None and s['session_id']!=self.session_id:
                self.fault='COVERAGE_SESSION_CHANGED_RESTART_REQUIRED';return
            if self.session_id is None:
                self.session_id=s['session_id'];self.coverage_epoch+=1
                self.coverage_grids.clear();self.coverage_reports.clear()
                self.info('session',f'[COVERAGE_SESSION] {self.session_id}; existing measurements preserved',0)
            self.coverage_reports.append((stamp,time.monotonic(),s));self.match_coverage()

        def on_grid(self,msg):
            if not geometry_matches(msg,data['map']):
                self.matched=None;self.warn('grid_bad','[COVERAGE_GEOMETRY_MISMATCH]');return
            vals=np.asarray(msg.data,dtype=np.int16).reshape(labels.shape).copy()
            self.coverage_grids.append((stamp_s(msg.header.stamp),time.monotonic(),vals));self.match_coverage()

        def match_coverage(self):
            for stamp,wall,s in reversed(self.coverage_reports):
                for gs,gw,grid in reversed(self.coverage_grids):
                    if abs(stamp-gs)>.5:continue
                    try:validate_coverage(s,grid,labels,signature,data['zones'])
                    except (ValueError,KeyError,TypeError):continue
                    if self.matched is None or stamp>=self.matched['stamp']:
                        self.matched=dict(stamp=stamp,wall=min(wall,gw),status=s,observed=grid==100)
                    return

        def on_costmap(self,msg):
            # Observability only: count deliveries and invalid payloads without changing
            # the existing costmap acceptance or failure behavior.
            receive_wall=time.monotonic()
            monitor=self.costmap_monitor
            prev=monitor['last_receive_wall']
            if prev is not None:
                gap=receive_wall-prev
                monitor['last_interarrival_wall_s']=gap
                monitor['max_interarrival_wall_s']=max(monitor['max_interarrival_wall_s'],gap)
            monitor['last_receive_wall']=receive_wall
            monitor['received_count']+=1
            monitor['last_header_stamp_s']=stamp_s(msg.header.stamp)
            if not geometry_matches(msg,data['map']):
                monitor['invalid_geometry_count']+=1
                self.costmap_entry=None;return
            cost=np.asarray(msg.data,dtype=np.int16).reshape(labels.shape).copy()
            if not np.all((cost>=-1)&(cost<=100)):
                monitor['invalid_values_count']+=1
                self.costmap_entry=None;return
            key=array_fingerprint(np.where(cost<0,3,np.where(cost==100,2,np.where(cost==99,1,0))).astype(np.uint8))
            self.costmap_entry=dict(stamp=stamp_s(msg.header.stamp),wall=receive_wall,values=cost,key=key)
            monitor['accepted_count']+=1
            monitor['last_valid_receive_wall']=receive_wall

        def costmap_diagnostic(self,now,wall):
            return snapshot_costmap_freshness(self.costmap_entry,self.costmap_monitor,now,wall)

        def on_stale_global_costmap(self,now,wall):
            # One event per stale episode, not one event per 50ms ROS polling tick.
            if self.costmap_stale_episode:return
            self.costmap_stale_episode=True
            d=self.costmap_diagnostic(now,wall)
            self.event('GLOBAL_COSTMAP_STALE_DIAG',**d)
            self.warn('costmap_stale_diag',
                '[GLOBAL_COSTMAP_STALE_DIAG] '+
                f'cause={d["cause"]} ros_age={d["ros_age_s"]}s wall_age={d["wall_age_s"]}s '+
                f'rx={d["received_count"]} accepted={d["accepted_count"]} '+
                f'last_gap={d["last_interarrival_wall_s"]}s max_gap={d["max_interarrival_wall_s"]}s',0)

        def on_fresh_global_costmap(self,now,wall):
            if not self.costmap_stale_episode:return
            self.costmap_stale_episode=False
            d=self.costmap_diagnostic(now,wall)
            self.event('GLOBAL_COSTMAP_RECOVERED_DIAG',**d)
            self.info('costmap_recovered_diag',
                '[GLOBAL_COSTMAP_RECOVERED_DIAG] '+
                f'ros_age={d["ros_age_s"]}s wall_age={d["wall_age_s"]}s received={d["received_count"]}',0)

        def on_local_costmap(self,msg):
            try:
                m=msg.info;w=int(m.width);h=int(m.height);r=float(m.resolution)
                if w<1 or h<1 or r<=0 or w*h!=len(msg.data):raise ValueError('Invalid dimensions')
                q=m.origin.orientation;yaw=math.atan2(2*q.w*q.z,q.w*q.w-q.z*q.z)
                if abs(q.x)>1e-5 or abs(q.y)>1e-5 or abs(q.w*q.w+q.z*q.z-1)>1e-3:raise ValueError('Nonplanar origin')
                g=Geometry(w,h,r,(float(m.origin.position.x),float(m.origin.position.y),yaw))
                v=np.asarray(msg.data,dtype=np.int16).reshape(h,w).copy()
                if not np.all((v>=-1)&(v<=100)):raise ValueError('Invalid data')
                self.local_entry=dict(geom=g,values=v,frame=msg.header.frame_id.lstrip('/'),
                                      stamp=stamp_s(msg.header.stamp),wall=time.monotonic())
            except (ValueError,TypeError,AttributeError):self.local_entry=None

        def on_scan(self,msg):
            try:
                model=SensorModel(float(msg.angle_min),float(msg.angle_increment),len(msg.ranges),float(msg.range_min),float(msg.range_max))
                model.validate();frame=msg.header.frame_id.lstrip('/')
                if not frame:raise ValueError('Empty frame')
                self.scan_entry=dict(model=model,frame=frame,stamp=stamp_s(msg.header.stamp),wall=time.monotonic())
            except (ValueError,TypeError):self.scan_entry=None

        def on_odom(self,msg):
            t=stamp_s(msg.header.stamp);v=msg.twist.twist.linear;w=msg.twist.twist.angular
            vals=(float(v.x),float(v.y),float(v.z),float(w.z));wall=time.monotonic()
            if not all(math.isfinite(x) for x in (t,*vals)):
                self.odom_entry=None;self.stop_evidence.reset();return
            self.odom_entry=dict(stamp=t,wall=wall,velocity=vals)
            self.stop_evidence.update(t,wall,*vals)

        def on_action_status(self,msg):
            # Snapshot, not accumulation: old active entries must not survive a new array.
            self.nav_statuses={bytes(x.goal_info.goal_id.uuid).hex():int(x.status) for x in msg.status_list}

        def on_stop_service(self,request,response):
            self.request_stop('USER_STOP_SERVICE')
            response.success=True;response.message='Stop requested for owned Goal only; verify terminal result and physical stop.'
            return response

        def latest_pose(self,target,source,check_age=True):
            t=self.tf_buffer.lookup_transform(target,source,Time(),timeout=Duration(seconds=0.0))
            p=planar_pose(t.transform,math.radians(5))
            age=self.ros_s()-stamp_s(t.header.stamp)
            if check_age and not -1.5<=age<=2.:raise ValueError(f'TF age={age:.2f}s')
            return p

        def inputs(self):
            now=self.ros_s();wall=time.monotonic()
            if self.fault:return self.fault,None
            if not self.reference_ok:return 'WAIT_MAP',None
            if wall-self.clock_advanced_wall>5.:return 'WAIT_CLOCK_PAUSED',None
            if self.matched is None:return 'WAIT_MATCHED_COVERAGE',None
            if self.coverage_reports:
                err=current_input_error(self.coverage_reports[-1][2],now,wall-self.coverage_reports[-1][1])
                if err:return err,None
            err=current_input_error(self.matched['status'],now,wall-self.matched['wall'])
            if err:return err,None
            cost=self.costmap_entry;scan=self.scan_entry;od=self.odom_entry
            if cost is None:return 'WAIT_GLOBAL_COSTMAP',None
            # Exactly the original 5s ROS / 15s wall freshness policy.
            if not -.1<=now-cost['stamp']<=5. or wall-cost['wall']>15.:
                self.on_stale_global_costmap(now,wall)
                return 'STALE_GLOBAL_COSTMAP',None
            self.on_fresh_global_costmap(now,wall)
            if scan is None:return 'WAIT_SCAN_GEOMETRY',None
            if not -.1<=now-scan['stamp']<=3. or wall-scan['wall']>10.:return 'STALE_SCAN_GEOMETRY',None
            if od is None:return 'WAIT_ODOMETRY',None
            if not -.1<=now-od['stamp']<=2. or wall-od['wall']>3.:return 'STALE_ODOMETRY',None
            try:
                pose=self.latest_pose('map',args.robot_frame)
                ext=self.latest_pose(args.robot_frame,scan['frame'],False)
                m=scan['model'];s=self.matched['status']
                sensor=SensorModel(m.angle_min,m.angle_increment,m.beams,m.range_min,
                                    min(m.range_max,float(s['max_observation_range_m'])),*ext)
                sensor.validate()
            except (TransformException,ValueError,KeyError) as exc:
                self.warn('tf',f'[WAIT_TF_OR_SENSOR] {exc}');return 'WAIT_TF',None
            return None,dict(pose=pose,sensor=sensor,cost=cost,status=s,
                             observed=self.matched['observed'],epoch=self.coverage_epoch)

        def foreign_active(self):
            return [gid for gid,st in self.nav_statuses.items() if st in ACTIVE and gid not in self.owned_ids]

        def request_stop(self,reason):
            if self.done:return
            if self.stop_requested is None:
                self.stop_requested=reason;self.stop_started=time.monotonic()
                self.zone_planner.stop()
                self.warn('stop',f'[STOP_REQUESTED] {reason}; no additional goals',0)
                self.event('STOP_REQUESTED',reason=reason)
                if reason=='DATA_LOSS_STALE_GLOBAL_COSTMAP':
                    diag=self.costmap_diagnostic(self.ros_s(),time.monotonic())
                    diag['bad_duration_wall_s']=None if self.bad_since is None else round(max(0.,time.monotonic()-self.bad_since),4)
                    diag['active_goal_id']=self.nav_uuid
                    diag['active_zone_id']=None if self.current is None else self.current.get('zone_id')
                    self.event('GLOBAL_COSTMAP_CANCEL_DIAG',**diag)
                    self.warn('costmap_cancel_diag','[GLOBAL_COSTMAP_CANCEL_DIAG] '+
                        f'cause={diag["cause"]} ros_age={diag["ros_age_s"]}s '+
                        f'wall_age={diag["wall_age_s"]}s rx={diag["received_count"]} '+
                        f'bad_duration={diag["bad_duration_wall_s"]}s',0)
            if self.nav_uuid is not None and self.nav_terminal is None:
                self.request_cancel(reason)
            elif self.plan_handle is not None:
                try:self.plan_handle.cancel_goal_async()
                except Exception:pass

        def request_cancel(self,reason):
            if self.cancel_reason is not None:return
            if self.nav_uuid is None:return
            self.cancel_reason=reason;self.cancel_started=time.monotonic()
            gid=self.nav_uuid
            self.event('CANCEL_REQUESTED',reason=reason,goal_id=gid)
            self.warn('cancel',f'[CANCEL_REQUESTED] own Goal={gid}; reason={reason}',0)
            if self.nav_handle is not None:
                self.cancel_future=self.nav_handle.cancel_goal_async()
            elif self.exact_cancel_client.service_is_ready():
                req=CancelGoal.Request();req.goal_info.goal_id.uuid=list(bytes.fromhex(gid))
                req.goal_info.stamp.sec=0;req.goal_info.stamp.nanosec=0
                self.cancel_future=self.exact_cancel_client.call_async(req)
            self.transition('CANCELING')

        def make_pose(self,c):
            p=PoseStamped();p.header.frame_id='map'
            # Time zero asks Nav2 for the current transform, not a stale planning snapshot.
            p.pose.position.x=float(c['x']);p.pose.position.y=float(c['y'])
            p.pose.orientation.z=float(math.sin(c['yaw']/2));p.pose.orientation.w=float(math.cos(c['yaw']/2))
            return p

        def plan_request(self,c,inp):
            self.book.activate(c['zone_id'])
            self.phase='PLAN_ENTRY' if c['goal_kind']=='ENTRY' else 'PLAN_LOCAL'
            self.current=c;self.current_path=[];self.current_path_length=None
            self.validated_plan_mode=None
            goal=ComputePathToPose.Goal();goal.goal=self.make_pose(c);goal.planner_id=args.planner_id;goal.use_start=False
            self.plan_send=self.plan_client.send_goal_async(goal);self.plan_handle=None;self.plan_result=None
            self.plan_started=time.monotonic();self.plan_attempts+=1
            self.plan_snapshot=dict(epoch=inp['epoch'],pose=inp['pose'],cost_key=inp['cost']['key'])
            self.transition('PLAN_PENDING')
            self.info('plan',f'[PLAN_REQUESTED] zone={c["zone_id"]} xy=({c["x"]:.2f},{c["y"]:.2f}) yaw={c["yaw"]:.3f}; GridBased; no motion',0)
            self.event('PLAN_REQUESTED',candidate=json_candidate(c))

        def reject_candidate(self,reason):
            self.warn('reject',f'[PLAN_REJECTED] {reason}; try another sampled view, not unreachable-room proof',0)
            if reason in ('POSE_OR_SESSION_CHANGED','COSTMAP_CHANGED_REPLAN','INPUTS_NOT_FRESH','GOAL_NO_LONGER_NEEDED'):
                self.current=None;self.options=[];self.batch_zone=None;self.options_context=None
            if self.current:
                self.book.rejected.append(json_candidate(self.current));self.event('PLAN_REJECTED',reason=reason,candidate=json_candidate(self.current))
            self.current=None;self.plan_send=None;self.plan_handle=None;self.plan_result=None
            self.validated_plan_mode=None
            self.transition('SELECTING')

        def nav_request(self,inp):
            if not args.execute:raise RuntimeError('Dry-run navigation invariant')
            if not args.supervised or self.validated_plan_mode!='NAVFN_GOAL_VALIDATED':
                raise RuntimeError('EXECUTION_REQUIRES_SUPERVISION_AND_NAVFN_GOAL_VALIDATION')
            if self.sent_goals>=args.max_goals:raise RuntimeError('Goal bound invariant')
            if self.nav_send is not None or self.nav_handle is not None:raise RuntimeError('Overlapping goal invariant')
            c=self.current;goal=NavigateToPose.Goal();goal.pose=self.make_pose(c);goal.behavior_tree=''
            uid=uuid.uuid4();self.nav_uuid=uid.hex;self.owned_ids.add(uid.hex)
            u=UUID();u.uuid=list(uid.bytes)
            self.before_fraction=self.book.records[c['zone_id']].measured
            self.before_overall=float(inp['status']['overall_observed_percent'])
            self.nav_terminal=None;self.cancel_reason=None;self.cancel_started=None;self.cancel_future=None
            self.nav_send=self.nav_client.send_goal_async(goal,feedback_callback=self.on_feedback,goal_uuid=u)
            self.nav_started=time.monotonic();self.sent_goals+=1
            self.book.activate(c['zone_id'])
            self.phase='ENTERING_ZONE' if c['goal_kind']=='ENTRY' else 'OBSERVING_ZONE'
            self.prefetch_for=None
            self.transition('NAV_PENDING')
            self.event('NAV_REQUESTED',goal_id=uid.hex,number=self.sent_goals,candidate=json_candidate(c),before_percent=100*self.before_fraction)
            self.info('nav_req',f'[NAV_REQUESTED] {self.sent_goals}/{args.max_goals} zone={c["zone_id"]}; selected yaw preserved; one active goal',0)

        def on_feedback(self,msg):
            if self.nav_uuid is None or bytes(msg.goal_id.uuid).hex()!=self.nav_uuid:return
            f=msg.feedback
            self.info('feedback',f'[NAV_FEEDBACK] {self.sent_goals}/{args.max_goals} remaining={f.distance_remaining:.2f}m recoveries={f.number_of_recoveries}')
            # Recovery alone is intentionally NEVER a cancellation trigger.

        def poll_actions(self,inp):
            if self.cancel_future is not None and self.cancel_future.done():
                f=self.cancel_future;self.cancel_future=None
                try:
                    reply=f.result();matched=any(bytes(g.goal_id.uuid).hex()==self.nav_uuid for g in reply.goals_canceling)
                    ok=int(reply.return_code)==0 and matched
                    self.info('cancel_ack',f'[CANCEL_RESPONSE] code={reply.return_code} own_goal_matched={matched}; terminal result still required',0)
                    self.event('CANCEL_RESPONSE',code=int(reply.return_code),own_goal_matched=matched)
                except Exception as exc:self.warn('cancel_err',f'[CANCEL_RESPONSE_ERROR] {exc}',0)
            if self.nav_send is not None and self.nav_send.done():
                f=self.nav_send;self.nav_send=None
                try:h=f.result()
                except Exception as exc:
                    self.request_stop('NAV_HANDSHAKE_ERROR');self.warn('nav_exc',str(exc));return
                if not h.accepted:
                    self.nav_uuid=None;self.request_stop('NAV_REJECTED');return
                self.nav_handle=h;self.nav_result=h.get_result_async()
                self.info('accept',f'[NAV_ACCEPTED] {self.sent_goals}/{args.max_goals}; goal locked',0)
                if self.cancel_reason is not None:
                    # A stop may precede the goal acknowledgment. Reissue exact owned cancellation now.
                    self.cancel_future=h.cancel_goal_async()
                elif self.stop_requested:self.request_cancel(self.stop_requested)
                else:self.transition('NAV_RUNNING')
            if self.nav_result is not None and self.nav_result.done():
                f=self.nav_result;self.nav_result=None
                try:r=f.result();self.nav_terminal=int(r.status)
                except Exception:
                    self.request_stop('NAV_RESULT_ERROR');return
                name=STATUS.get(self.nav_terminal,str(self.nav_terminal))
                self.info('result',f'[NAV_RESULT] {self.sent_goals}/{args.max_goals} {name}',0)
                self.event('NAV_RESULT',status=name,goal_id=self.nav_uuid,cancel_reason=self.cancel_reason)
                self.post_result_ros=self.ros_s();self.post_result_wall=time.monotonic()
                self.post_result_scans=int(self.matched['status']['evidence_scans']) if self.matched else -1
                self.stop_evidence.reset(after=self.post_result_ros)
                self.transition('WAIT_STOP_AND_COVERAGE')
                if self.nav_terminal==6 and not args.continue_on_nav_abort:self.request_stop('NAV_ABORTED_INSPECT_BEFORE_RETRY')
                elif self.nav_terminal==5 and self.cancel_reason!='COVERAGE_SUFFICIENT':
                    self.request_stop(self.stop_requested or 'EXTERNAL_OR_SAFETY_CANCEL')
                elif self.nav_terminal not in (4,5,6):self.request_stop('UNEXPECTED_NAV_RESULT')
            # Handle only the preflight action below (never during an owned navigation).
            if self.plan_send is not None and self.plan_send.done():
                f=self.plan_send;self.plan_send=None
                try:h=f.result()
                except Exception as exc:self.request_stop('PLAN_HANDSHAKE_ERROR');self.warn('plan_error',str(exc),0);return
                if not h.accepted:self.reject_candidate('PLAN_REJECTED_BY_NAVFN');return
                self.plan_handle=h;self.plan_result=h.get_result_async()
                if self.stop_requested:
                    try:h.cancel_goal_async()
                    except Exception:pass
            if self.plan_result is not None and self.plan_result.done():
                f=self.plan_result;self.plan_result=None;self.plan_handle=None
                if self.stop_requested:return
                try:
                    result=f.result()
                    if int(result.status)!=4:raise ValueError('NAVFN_'+STATUS.get(int(result.status),'UNKNOWN'))
                    if inp is None:raise ValueError('INPUTS_NOT_FRESH')
                    if (inp['epoch']!=self.plan_snapshot['epoch'] or math.dist(inp['pose'][:2],self.plan_snapshot['pose'][:2])>.15
                            or angle_diff(inp['pose'][2],self.plan_snapshot['pose'][2])>.15):
                        raise ValueError('POSE_OR_SESSION_CHANGED')
                    if inp['cost']['key']!=self.plan_snapshot['cost_key']:raise ValueError('COSTMAP_CHANGED_REPLAN')
                    path=result.result.path
                    points=[(p.pose.position.x,p.pose.position.y) for p in path.poses]
                    if any(p.header.frame_id.lstrip('/') not in ('','map') for p in path.poses):raise ValueError('MIXED_PATH_FRAME')
                    length=validate_nav_path(points,path.header.frame_id,self.current,inp['pose'],geom,known_free,inp['cost']['values'])
                    self.current_path=points;self.current_path_length=length
                    free=known_free & (inp['cost']['values']>=0)&(inp['cost']['values']<100)
                    if not oriented_footprint_clear(free,geom,(self.current['x'],self.current['y'],self.current['yaw']),cfg):
                        raise ValueError('GOAL_FOOTPRINT_BLOCKED')
                    # Deliberately DO NOT recheck the full Navfn polyline with
                    # STRICT_SWEPT. Nav2's Costmaps/RPP handle dynamic collision
                    # prediction during execution. This is not a safety guarantee.
                    # Keep the approved zone mask/goal-footprint invariants.
                    z=self.zones_by_id[self.current['zone_id']]
                    if not is_inside(z,(self.current['x'],self.current['y'],self.current['yaw']),bundle,cfg):
                        raise ValueError('GOAL_NOT_INSIDE_TARGET_ZONE')
                    self.event('NAVFN_GOAL_CHECK_OK',goal_kind=self.current['goal_kind'],
                               path_length_m=length)
                    c=refresh_candidate(self.current,inp['observed'],geom,cfg)
                    if c is None or self.book.records[c['zone_id']].state in ('THRESHOLD_PENDING','OBSERVATION_SUFFICIENT'):
                        raise ValueError('GOAL_NO_LONGER_NEEDED')
                    self.current=c
                except Exception as exc:self.reject_candidate(str(exc));return
                validation_mode='NAVFN_GOAL_VALIDATED'
                self.info('plan_ok',f'[PLAN_OK] zone={self.current["zone_id"]} poses={len(self.current_path)} navfn_length={length:.2f}m; yaw={self.current["yaw"]:.3f}; validation={validation_mode}',0)
                self.event('PLAN_OK',length_m=length,poses=len(self.current_path),
                           validation_mode=validation_mode,execution_authorized=args.execute,
                           candidate=json_candidate(self.current))
                self.validated_plan_mode=validation_mode
                self.transition('READY_TO_SEND')

        def early_stop_area_ok(self,inp):
            local=self.local_entry;now=self.ros_s();wall=time.monotonic()
            if local is None or not -.1<=now-local['stamp']<=2. or wall-local['wall']>5:return False
            if not safe_early_stop(known_free,inp['cost']['values'],geom,inp['pose'],cfg):return False
            try:p=self.latest_pose(local['frame'],args.robot_frame)
            except (TransformException,ValueError):return False
            return safe_early_stop(local['values']>=0,local['values'],local['geom'],p,cfg)

        def finish(self,state,code=0):
            if self.done:return
            self.exit_code=code;self.transition(state);self.done=True
            self.event('SUMMARY',state=state,sent=self.sent_goals,succeeded=self.succeeded_goals,
                       coverage_cancels=self.coverage_cancels,zones=self.book.summary())
            self.info('summary',f'[SUMMARY] state={state}; goals_sent={self.sent_goals}; succeeded={self.succeeded_goals}; coverage_canceled={self.coverage_cancels}; not proof of full hospital search',0)
            self.publish_report(force=True)

        def after_result(self,inp):
            now=self.ros_s();wall=time.monotonic()
            if self.post_result_ros is None:return
            if not self.stop_evidence.ready(now,wall):
                self.info('stop_wait','[WAIT_STOP] need fresh low-speed odometry after terminal result')
                if now-self.post_result_ros>20 or wall-self.post_result_wall>30:self.finish('STOP_UNCONFIRMED_MANUAL_CHECK',2)
                return
            # Preserve the fresh stop check on EVERY poll. Only the informational
            # message is deduplicated; subsequent stale/invalid odometry still blocks
            # progress through the ready() check above.
            if self.nav_uuid is not None and self._stop_confirmed_logged_goal_id!=self.nav_uuid:
                self.info('stopped','[STOP_CONFIRMED] fresh odometry below 0.03m/s and 0.05rad/s for >=1 simulation second',0)
                self._stop_confirmed_logged_goal_id=self.nav_uuid
            if self.stop_requested:
                self.finish(self.stop_requested,0 if self.stop_requested.startswith('USER_') else 2);return
            if inp is None:
                if now-self.post_result_ros>20 or wall-self.post_result_wall>30:
                    self.finish('POST_RESULT_INPUT_TIMEOUT',2)
                return
            s=inp['status']
            if int(s['evidence_scans'])<=self.post_result_scans or float(s['last_evidence_scan_end_s'])<=self.post_result_ros:
                self.info('coverage_wait','[WAIT_COVERAGE] require a NEW post-result observation')
                if now-self.post_result_ros>20:self.finish('POST_RESULT_COVERAGE_TIMEOUT',2)
                return
            zone_id=self.current['zone_id']
            r=self.book.records[zone_id]
            if r.state=='THRESHOLD_PENDING':
                if now-self.post_result_ros>20:self.finish('POST_RESULT_THRESHOLD_TIMEOUT',2)
                return
            if self.nav_terminal==6:
                if not args.continue_on_nav_abort or not self.local_start_clear(inp):
                    self.finish('NAV_ABORTED_INSPECT_BEFORE_RETRY',2);return
                self.book.defer(zone_id,'NAV_ABORTED_AFTER_CONFIRMED_STOP')
                self.event('ZONE_DEFERRED',zone=zone_id,reason='NAV_ABORTED_AFTER_CONFIRMED_STOP')
            else:
                record=self.book.record_visit(self.current,self.before_fraction,r.measured,
                                               STATUS[self.nav_terminal],self.cancel_reason)
                if self.nav_terminal==4:self.succeeded_goals+=1
                elif self.nav_terminal==5:self.coverage_cancels+=1
                self.event('COVERAGE_GAIN',**record)
                self.info('gain',f'[COVERAGE_GAIN] zone={zone_id} {100*self.before_fraction:.2f}% -> {100*r.measured:.2f}% ({record["gain_pp"]:+.2f}pp)',0)
                inside=is_inside(self.zones_by_id[zone_id],inp['pose'],bundle,cfg)
                if inside:
                    self.book.mark_entered(zone_id)
                    self.event('ENTRY_CONFIRMED',zone=zone_id,pose=list(inp['pose']),goal_kind=self.current['goal_kind'])
                    self.info('entry',f'[ENTRY_CONFIRMED] {zone_id}; measured pose/footprint inside target mask',0)
                elif not self.book.is_complete(zone_id):
                    diag=arrival_diagnostic(self.zones_by_id[zone_id],
                        (self.current['x'],self.current['y'],self.current['yaw']),inp['pose'],bundle,cfg)
                    diag['goal_kind']=self.current['goal_kind']
                    self.event('ENTRY_ARRIVAL_MISMATCH',**diag)
                    self.info('arrival_mismatch',f'[ENTRY_ARRIVAL_MISMATCH] zone={zone_id} '
                        f'goal_to_actual_xy={diag["xy_error_m"]:.3f}m '
                        f'center_inside={diag["actual"]["center_inside"]} '
                        f'footprint_inside={diag["actual"]["footprint_inside"]} '
                        f'mask_depth={diag["actual"]["signed_mask_depth_m"]} ',0)
                    # Only retry a completed ENTRY, with fresh post-result
                    # Coverage + stopped odometry + a current clear local footprint.
                    # Replanning picks a DIFFERENT interior goal and requires
                    # a NEW successful Navfn preflight. No move without validation.
                    st=self.book.visit_state[zone_id]
                    eligible_retry=(self.current['goal_kind']=='ENTRY' and self.nav_terminal==4
                        and st.entry_arrival_retries < args.max_entry_arrival_retries
                        and self.sent_goals < args.max_goals and self.local_start_clear(inp))
                    if eligible_retry:
                        st.entry_arrival_retries+=1
                        self.book.activate(zone_id)
                        self.event('ENTRY_ALTERNATE_RETRY',zone=zone_id,attempt=st.entry_arrival_retries,
                                   max_attempts=args.max_entry_arrival_retries,reason='MEASURED_OUTSIDE_ZONE')
                        self.info('entry_retry',f'[ENTRY_ALTERNATE_RETRY] {zone_id} '
                            f'attempt={st.entry_arrival_retries}/{args.max_entry_arrival_retries}; '
                            'new goal will require fresh Navfn preflight',0)
                    else:
                        reason='TERMINAL_POSE_OUTSIDE_ZONE_REVIEW_TOLERANCE'
                        self.book.defer(zone_id,reason)
                        self.event('ZONE_DEFERRED',zone=zone_id,reason=reason,
                            retry_attempts=st.entry_arrival_retries)
                        self.info('entry_deferred',f'[ZONE_DEFERRED] {zone_id} '
                            'entry arrival mismatch; alternate exhausted or unsafe',0)
            self.nav_send=None;self.nav_handle=None;self.nav_result=None;self.nav_uuid=None
            self.nav_terminal=None;self.cancel_reason=None;self.cancel_future=None;self.post_result_ros=None
            self.current=None;self.options=[];self.current_path=[];self.batch_zone=None
            self.options_context=None;self.phase='SELECT_ZONE'
            if self.sent_goals>=args.max_goals:self.finish('GOAL_LIMIT_REACHED');return
            self.info('next','[NEXT_ZONE_STEP] reuse local catalog or select next incomplete zone',0)
            self.transition('SELECTING')

        def local_start_clear(self,inp):
            local=self.local_entry;now=self.ros_s();wall=time.monotonic()
            if local is None or not -.1<=now-local['stamp']<=2. or wall-local['wall']>5.:
                return False
            try:
                p=self.latest_pose(local['frame'],args.robot_frame)
                i,j=map(math.floor,local['geom'].world_to_grid_float(*p[:2]))
                return (local['geom'].contains(i,j) and 0<=local['values'][j,i]<99
                    and oriented_footprint_clear((local['values']>=0)&(local['values']<100),local['geom'],p,cfg))
            except (TransformException,ValueError):return False

        def snapshot(self,inp):
            snap=dict(inp);snap['observed']=inp['observed'].copy()
            snap['cost']=dict(inp['cost'],values=inp['cost']['values'].copy())
            return snap

        def submit_job(self,kind,inp,zones):
            if self.worker is not None:raise RuntimeError('Overlapping planner workers')
            self.job_context=dict(kind=kind,epoch=inp['epoch'],cost_key=inp['cost']['key'],
                pose=inp['pose'],zone=zones[0]['id'] if len(zones)==1 else None)
            self.worker=self.pool.submit(self.zone_planner.run,kind,self.snapshot(inp),zones,
                list(self.book.visits)+list(self.book.rejected))
            self.event('PLANNER_JOB',kind=kind,zones=[z['id'] for z in zones])
            if kind!='PREFETCH':self.transition('GENERATING_ENTRY' if kind=='ENTRY' else 'GENERATING_LOCAL_VIEWS')

        def collect_job(self,inp):
            if self.worker is None or not self.worker.done():return False
            worker=self.worker;context=self.job_context;self.worker=None;self.job_context=None
            try:batch=worker.result()
            except TimeoutError:
                self.event('PLANNING_BUDGET',kind=context['kind'],zone=context['zone'])
                if context['kind']=='PREFETCH':return True
                if context['zone']:
                    self.book.defer(context['zone'],'PLANNING_BUDGET_PARTIAL')
                    self.batch_zone=None;self.options=[]
                else:self.request_stop('ENTRY_PLANNING_BUDGET_EXCEEDED')
                return True
            except Exception as exc:
                self.warn('planner_error',f'[PLANNER_ERROR] {type(exc).__name__}: {exc}',0)
                self.request_stop('PLANNER_ERROR');return True
            self.event('PLANNER_TIMING',**{k:v for k,v in batch.items() if k not in ('candidates','entrance')})
            if batch.get('start_route_mode') == 'NAVFN_START_PROBE_NO_CONSERVATIVE_BRIDGE':
                self.event('START_BRIDGE_NAVFN_DELEGATED',kind=context['kind'],
                           pose=list(context['pose']),reason='NO_SHORT_STRICT_GRID_BRIDGE',
                           navfn_preflight_required=True,local_start_check_required=True)
                self.warn('start_bridge','[START_BRIDGE_NAVFN_DELEGATED] short conservative bridge unavailable; '
                          'safe endpoints only; Navfn + local start + goal footprint checks still mandatory',0)
            self.info('timing',f'[PLANNER_DONE] kind={context["kind"]} elapsed={batch["elapsed_s"]:.3f}s cache_hit={batch.get("cache_hit",False)}',0)
            if context['kind']=='PREFETCH':return True  # Never becomes executable without fresh reranking.
            if (inp is None or inp['epoch']!=context['epoch'] or inp['cost']['key']!=context['cost_key']
                    or math.dist(inp['pose'][:2],context['pose'][:2])>.15
                    or angle_diff(inp['pose'][2],context['pose'][2])>.15):
                self.info('stale','[DISCARD_STALE_PLAN] new snapshot required',0)
                self.batch_zone=None;self.options=[];return True
            for zid,reason in batch.get('failures',{}).items():
                if not self.book.is_complete(zid):
                    self.book.defer(zid,reason);self.event('ZONE_DEFERRED',zone=zid,reason=reason)
            self.options=batch['candidates'];self.batch_zone=batch['zone_id'];self.batch_kind=batch['kind']
            self.options_context=context
            if self.batch_zone and self.options:
                self.book.activate(self.batch_zone)
                self.info('select_zone',f'[ZONE_SELECTED] {self.batch_zone}; phase={self.batch_kind}; candidates={len(self.options)}',0)
                self.event('ZONE_SELECTED',zone=self.batch_zone,phase=self.batch_kind)
            elif self.batch_zone:
                self.book.defer(self.batch_zone,'NO_USEFUL_INTERNAL_VIEW_THIS_REVISION')
                self.event('ZONE_DEFERRED',zone=self.batch_zone,reason='NO_USEFUL_INTERNAL_VIEW_THIS_REVISION')
                self.batch_zone=None
            self.transition('SELECTING');return True

        def prefetch_during_navigation(self,inp):
            if inp is None or self.stop_requested or self.current is None:return
            if self.worker is not None:
                self.collect_job(inp);return
            zid=self.current['zone_id']
            if self.book.is_complete(zid):return
            key=self.nav_uuid  # one prefetch per Goal; changed input is rebuilt at handoff
            if self.prefetch_for==key:return
            self.prefetch_for=key
            self.submit_job('PREFETCH',inp,[self.zones_by_id[zid]])

        def select_zone_goal(self,inp):
            now=self.ros_s();wall=time.monotonic()
            if self.book.all_complete():self.finish('ALL_REQUESTED_ZONES_OBSERVED');return
            if self.sent_goals>=args.max_goals:self.finish('GOAL_LIMIT_REACHED');return
            if self.plan_attempts>=args.max_plan_attempts:self.finish('PLAN_ATTEMPT_LIMIT_PARTIAL');return
            if not self.plan_client.server_is_ready() or (args.execute and not self.nav_client.server_is_ready()):
                self.info('servers','[WAIT_NAV2_ACTION_SERVERS]');return
            if not self.stop_evidence.ready(now,wall):
                self.info('stop','[WAIT_STOP] fresh stationary odometry required before new Goal');return
            if wall-self.boot_wall<2.:return
            if self.worker is not None:
                if not self.worker.done():self.publish_report();return
                self.collect_job(inp)
                if self.stop_requested:return
            for zid in self.book.reopen_changed():
                self.event('ZONE_REOPENED',zone=zid,reason='CHANGED_COSTMAP_OR_OTHER_ZONE_PROGRESS')
            if self.options_context is not None:
                c=self.options_context
                if (inp['epoch']!=c['epoch'] or inp['cost']['key']!=c['cost_key']
                        or math.dist(inp['pose'][:2],c['pose'][:2])>.15
                        or angle_diff(inp['pose'][2],c['pose'][2])>.15):
                    self.options=[];self.batch_zone=None;self.options_context=None
            while self.options:
                c=self.options.pop(0)
                if self.book.records[c['zone_id']].state in ('OBSERVATION_SUFFICIENT','THRESHOLD_PENDING','PARTIAL_DEFERRED'):continue
                c=refresh_candidate(c,inp['observed'],geom,cfg)
                if c is None:continue
                self.event('ZONE_GOAL',candidate=json_candidate(c))
                self.info('chosen',f'[ZONE_GOAL] kind={c["goal_kind"]} zone={c["zone_id"]} xy=({c["x"]:.2f},{c["y"]:.2f}) yaw={c["yaw"]:.3f} predicted_new={c["new_area_m2"]:.2f}m2',0)
                self.plan_request(c,inp);self.publish_markers();return
            if self.batch_zone:
                if self.book.records[self.batch_zone].state not in ('OBSERVATION_SUFFICIENT','THRESHOLD_PENDING'):
                    reason='ENTRY_PREFLIGHT_EXHAUSTED_PARTIAL' if self.batch_kind=='ENTRY' else 'LOCAL_PREFLIGHT_EXHAUSTED_PARTIAL'
                    self.book.defer(self.batch_zone,reason)
                    self.event('ZONE_DEFERRED',zone=self.batch_zone,reason=reason)
                self.batch_zone=None;self.options_context=None
            eligible=self.book.eligible()
            if not eligible:
                if self.book.pending():self.info('threshold','[WAIT_THRESHOLD_CONFIRMATION]');return
                self.finish('PARTIAL_DEFERRED_REMAIN');return
            active=self.book.active_zone
            if active is None:
                # A robot already inside an incomplete room need not leave and re-enter.
                for z in sorted(eligible,key=lambda z:(room_priority(z,args.zone_policy),z['id'])):
                    if (room_priority(z,args.zone_policy)==0 or not any(room_priority(q,args.zone_policy)==0 for q in eligible)) and is_inside(z,inp['pose'],bundle,cfg):
                        self.book.activate(z['id']);self.book.mark_entered(z['id']);active=z['id'];break
            if active and any(z['id']==active for z in eligible):
                z=self.zones_by_id[active]
                if is_inside(z,inp['pose'],bundle,cfg):
                    self.phase='OBSERVE_ZONE';self.submit_job('OBSERVE',inp,[z]);return
                self.phase='ENTER_ZONE';self.submit_job('ENTRY',inp,[z]);return
            self.phase='SELECT_ZONE';self.submit_job('ENTRY',inp,eligible)

        def poll(self):
            if self.done:return
            now=self.ros_s();wall=time.monotonic()
            if self.last_clock is None or now>self.last_clock:self.clock_advanced_wall=wall
            if self.last_clock is not None and now<self.last_clock-.1:self.fault='CLOCK_RESET_RESTART_REQUIRED'
            self.last_clock=now
            error,inp=self.inputs()
            if inp is not None:
                self.book.context(inp['cost']['key'],wall)
                try:
                    for zid in self.book.update(inp['status']):
                        self.info('complete_'+zid,f'[ZONE_COMPLETE] {zid} measured={self.book.records[zid].measured*100:.1f}%; skip extra views',0)
                        self.event('ZONE_COMPLETE',zone=zid,measured_percent=self.book.records[zid].measured*100)
                except ValueError as exc:self.fault=str(exc);error=self.fault;inp=None
            self.poll_actions(inp)
            if self.done:return
            if self.foreign_active():
                self.request_stop('OTHER_NAVIGATION_GOAL_ACTIVE')
            nav_inflight=self.nav_uuid is not None and self.nav_terminal is None
            if nav_inflight and error:
                if self.bad_since is None:self.bad_since=wall
                if self.fault or wall-self.bad_since>=2.:self.request_stop('DATA_LOSS_'+error)
            else:self.bad_since=None
            if nav_inflight and self.nav_send is not None and wall-self.nav_started>15:
                self.request_stop('NAV_ACCEPTANCE_TIMEOUT')
            if nav_inflight and self.cancel_started is not None and wall-self.cancel_started>15:
                self.finish('CANCEL_UNCONFIRMED_MANUAL_STOP_REQUIRED',2);return
            if self.nav_terminal is not None:
                self.after_result(inp);self.publish_report();return
            if self.stop_requested:
                if not nav_inflight:
                    self.finish(self.stop_requested,0 if self.stop_requested.startswith('USER_') else 2)
                self.publish_report();return
            if nav_inflight:
                self.prefetch_during_navigation(inp)
                if inp is not None and self.book.is_complete(self.current['zone_id']):
                    if False and self.early_stop_area_ok(inp):
                        self.request_cancel('COVERAGE_SUFFICIENT')
                    else:
                        self.info('coverage_hold','[COVERAGE_REACHED] skip future views; finish current Goal (early cancellation disabled or stopping-area gate not passed)')
                self.publish_report();return
            if error:
                self.info(error,f'[{error}] waiting; no goals sent')
                if self.fault:self.request_stop(self.fault)
                self.publish_report();return
            # A handshake/path response timeout is a communication fault, NOT unreachable-room evidence.
            if self.plan_send is not None or self.plan_handle is not None:
                if wall-self.plan_started>15:self.request_stop('PLAN_RESPONSE_TIMEOUT')
                self.publish_report();return
            if self.state=='READY_TO_SEND':
                if not args.execute:
                    self.publish_markers();self.finish('DRY_RUN_PASS');return
                if not self.stop_evidence.ready(now,wall):
                    self.info('pre_stop','[WAIT_STOP] before new NavigateToPose');return
                if (math.dist(inp['pose'][:2],self.plan_snapshot['pose'][:2])>.15
                        or angle_diff(inp['pose'][2],self.plan_snapshot['pose'][2])>.15
                        or inp['cost']['key']!=self.plan_snapshot['cost_key']):
                    self.options=[];self.current=None;self.batch_zone=None;self.validated_plan_mode=None
                    self.transition('SELECTING');return
                if self.book.records[self.current['zone_id']].state in ('THRESHOLD_PENDING','OBSERVATION_SUFFICIENT'):
                    self.options=[];self.current=None;self.batch_zone=None;self.validated_plan_mode=None
                    self.transition('SELECTING');return
                if not self.local_start_clear(inp):
                    self.request_stop('LOCAL_START_NOT_CLEAR_OR_STALE');return
                self.nav_request(inp);self.publish_report();return
            self.select_zone_goal(inp)
            self.publish_report()

        def publish_markers(self):
            if not rclpy.ok():return
            arr=MarkerArray();clear=Marker();clear.header.frame_id='map';clear.action=Marker.DELETEALL;arr.markers.append(clear)
            c=self.current
            if c:
                def marker(ns,ident,kind,color):
                    m=Marker();m.header.frame_id='map';m.header.stamp=self.get_clock().now().to_msg()
                    m.ns=ns;m.id=ident;m.type=kind;m.action=Marker.ADD;m.pose.orientation.w=1.
                    m.color.r,m.color.g,m.color.b,m.color.a=color;return m
                m=marker('owned_view_goal',0,Marker.ARROW,(1.,.3,.05,1.))
                m.pose=self.make_pose(c).pose;m.pose.position.z=.15
                m.scale.x=.65;m.scale.y=.12;m.scale.z=.12;arr.markers.append(m)
                m=marker('target_zone',0,Marker.TEXT_VIEW_FACING,(.05,.1,.15,1.))
                m.pose.position.x=float(c['x']);m.pose.position.y=float(c['y']);m.pose.position.z=.9;m.scale.z=.30
                m.text=f'{c["zone_id"]}\n{self.state} | goal {self.sent_goals}/{args.max_goals}'
                arr.markers.append(m)
                if self.current_path:
                    m=marker('navfn_preflight_path',0,Marker.LINE_STRIP,(.2,.3,1.,.85));m.scale.x=.04
                    m.points=[Point(x=float(x),y=float(y),z=.12) for x,y in self.current_path];arr.markers.append(m)
                inds=np.asarray(c['new_indices'],dtype=int)
                if len(inds):
                    m=marker('PREDICTED_not_observed',0,Marker.CUBE_LIST,(1.,.8,.05,.6))
                    m.scale.x=m.scale.y=geom.resolution*.8;m.scale.z=.01
                    for v in inds[::max(1,math.ceil(len(inds)/10000))]:
                        j,i=divmod(int(v),geom.width);x,y=geom.cell_to_world(i,j)
                        m.points.append(Point(x=float(x),y=float(y),z=.10))
                    arr.markers.append(m)
            self.marker_pub.publish(arr)

        def publish_report(self,force=False):
            wall=time.monotonic()
            if not force and wall<self.next_report:return
            self.next_report=wall+2.
            report=dict(schema='hospital_zone_mission_v1',implementation='zone_based_nav2_delegated_v2_4_start_bridge',state=self.state,phase=self.phase,
                        active_zone=self.book.active_zone,visibility=self.zone_planner.visibility.summary(),
                        execute=args.execute,goal_limit=args.max_goals,sent=self.sent_goals,
                        nav_succeeded=self.succeeded_goals,coverage_canceled=self.coverage_cancels,
                        plan_attempts=self.plan_attempts,coverage_session=self.session_id,
                        zone_bundle_signature=signature,threshold=cfg.threshold,
                        early_cancel=False,current=json_candidate(self.current),
                        nav2_path_length_m=self.current_path_length,owned_goal_id=self.nav_uuid,
                        stop_reason=self.stop_requested,cancel_reason=self.cancel_reason,
                        ros_time_s=self.ros_s(),zones=self.book.summary(),visits=self.book.visits,
                        disclaimer='Measured 2D spatial coverage, not victim-search completion; bounded trial, no guaranteed reachability')
            m=String();m.data=json.dumps(report,ensure_ascii=False,allow_nan=False)
            if rclpy.ok():self.status_pub.publish(m)
            atomic_json(output/'latest_status.json',report)
            if self.done:atomic_json(output/'summary.json',report)
            self.publish_markers()

        def close(self):
            self.zone_planner.stop()
            if self.worker:self.worker.cancel()
            self.pool.shutdown(wait=True,cancel_futures=True)
            self.publish_report(force=True)
            self.events_handle.close()

    return HospitalSequence


def main():
    p=parser();args,ros_args=p.parse_known_args()
    if ros_args and ros_args[0]!='--ros-args':p.error('Unknown arguments: '+' '.join(ros_args))
    if args.execute and not args.supervised:p.error('--execute requires --supervised for this Gazebo pilot')
    for val,lo,hi,name in [(args.max_goals,1,100,'max-goals'),(args.max_views_per_zone,1,20,'max-views-per-zone'),
                           (args.max_candidates,1,300,'max-candidates'),(args.heading_samples,1,16,'heading-samples'),
                           (args.entry_candidates,1,20,'entry-candidates'),(args.max_plan_attempts,1,300,'max-plan-attempts'),
                           (args.max_revisits,0,5,'max-revisits'),
                           (args.max_entry_arrival_retries,0,2,'max-entry-arrival-retries')]:
        if not lo<=val<=hi:p.error('Invalid '+name)
    if not 0<=args.entry_arrival_buffer_m<=1.0:
        p.error('Invalid entry-arrival-buffer-m')
    if not .1<=args.entry_inset_m<=2. or not 1<=args.planning_budget_s<=120 or not 0<=args.revisit_cooldown_s<=600:
        p.error('Invalid geometry/time budget')
    cfg=Settings(threshold=args.threshold,max_candidates=args.max_candidates,
                 heading_samples=args.heading_samples);cfg.validate()
    bundle=load_bundle(args.zones.expanduser(),args.reference.expanduser())
    ids={z['id'] for z in bundle[0]['zones']}
    if args.zone_ids and (len(set(args.zone_ids))!=len(args.zone_ids) or not set(args.zone_ids)<=ids):
        p.error('Invalid/duplicate zone IDs. Available: '+', '.join(sorted(ids)))
    import fcntl
    lockname=hashlib.sha256((os.environ.get('ROS_DOMAIN_ID','0')+'|'+args.namespace).encode()).hexdigest()[:16]
    lock=open('/tmp/hospital_observation_sequence_'+lockname+'.lock','a')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('Another hospital_observation_sequence instance is already running')
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from rclpy.executors import ExternalShutdownException
    output=args.output_dir.expanduser().resolve()/datetime.now().strftime('session_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'run_manifest.json',dict(
        version='DN011_ZoneBased_Exploration_v2.3',command=list(sys.argv),
        python=sys.version,numpy=np.__version__,
        options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        zone_bundle_signature=bundle[4],
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path(__file__).resolve().parent.glob('*.py')},
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (args.zones,args.reference) if p.is_file()}))
    rclpy.init(args=ros_args,signal_handler_options=SignalHandlerOptions.NO)
    node=None;interrupts=[0]
    def interrupted(signum,frame):interrupts[0]+=1
    old_int=signal.signal(signal.SIGINT,interrupted);old_term=signal.signal(signal.SIGTERM,interrupted)
    try:
        node=make_node_type(args,bundle,cfg,output)()
        if not node.get_parameter('use_sim_time').value:
            raise ValueError('Gazebo trial requires --ros-args -p use_sim_time:=true')
        while rclpy.ok() and not node.done:
            if interrupts[0]:node.request_stop('USER_INTERRUPT')
            if interrupts[0]>=2:
                print('[FORCED_EXIT] Nav2 may still be running. Use RViz Cancel and verify physical stop.',file=sys.stderr)
                node.exit_code=2;break
            try:
                rclpy.spin_once(node,timeout_sec=.05);node.poll()
            except Exception as exc:
                node.warn('fatal',f'[RUNTIME_ERROR] {type(exc).__name__}: {exc}',0)
                node.request_stop('RUNTIME_ERROR')
                # Keep spinning to receive cancellation/result instead of shutting ROS down immediately.
                deadline=time.monotonic()+15
                while rclpy.ok() and not node.done and time.monotonic()<deadline and interrupts[0]<2:
                    try:
                        rclpy.spin_once(node,timeout_sec=.05);node.poll_actions(None)
                        if node.nav_terminal is not None:node.after_result(None)
                        elif node.nav_uuid is None:node.finish('RUNTIME_ERROR',2)
                    except Exception:time.sleep(.05)
                if not node.done:node.finish('RUNTIME_ERROR_STOP_UNCONFIRMED',2)
        return node.exit_code
    except ExternalShutdownException:
        print('[ROS_SHUTDOWN] Could not confirm stop. Check Nav2 and Gazebo.',file=sys.stderr);return 2
    finally:
        signal.signal(signal.SIGINT,old_int);signal.signal(signal.SIGTERM,old_term)
        if node is not None:
            try:node.close()
            finally:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        lock.close()
        print(f'[SAVED] {output}')
        print(('Dry Run: no NavigateToPose was requested by this node.' if not args.execute
               else 'Execute mode may have commanded navigation; independently verify physical stop.')
              + ' Stopped odometry is not an emergency-stop guarantee.')


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,ImportError,KeyError) as exc:
        print(f'[ERROR] {exc}',file=sys.stderr);raise SystemExit(2)
