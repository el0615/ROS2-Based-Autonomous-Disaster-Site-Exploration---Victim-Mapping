#!/usr/bin/env python3
"""Single-room observation planning for ROS2 Humble. DISPLAY / DIAGNOSTICS ONLY.

Reads approved zones, user's actual coverage, current robot/sensor TF and Nav2
full global costmap. Publishes only /observation_planner/markers and /status.
No NavigateToPose, no cancellation, no cmd_vel, no /map or TF output.
New implementation inspired by IPA visibility-based sensor placement and NBV;
not an installed/copied IPA or ETH planner. See docs/observation_planner_provenance.md.
"""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from hospital_coverage_core import load_bundle, planar_pose
from hospital_zone_visualizer import verify_map
from observation_planning_core import (
    Settings,SensorModel,ThresholdTracker,array_fingerprint,build_plan,rerank_plan,
    reportable,validate_coverage,current_input_error,
)


from exploration_io import stamp_s, geometry_matches, atomic_json


def build_node_type(args,bundle,cfg,output):
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from rclpy.duration import Duration
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy,qos_profile_sensor_data
    from nav_msgs.msg import OccupancyGrid
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker,MarkerArray
    from tf2_ros import Buffer,TransformListener,TransformException
    data,geom,labels,known_free,signature=bundle
    zone=next((z for z in data['zones'] if z['id']==args.zone),None)
    if zone is None: raise ValueError(f'Unknown --zone {args.zone}; available: '+', '.join(z['id'] for z in data['zones']))

    class ObservationPlanner(Node):
        def __init__(self):
            super().__init__('observation_goal_planner')
            self.reference_ok=False;self.fault=None;self.last_map_reason=None
            self.coverage_reports=deque(maxlen=12);self.coverage_grids=deque(maxlen=12)
            self.costmap_entry=None;self.scan_entry=None;self.last_scan_signature=None
            self.matched=None;self.session_id=None;self.coverage_epoch=0
            self.threshold_tracker=ThresholdTracker(cfg.threshold)
            self.last_selection=None;self.cached_plan=None;self.cached_key=None
            self.future=None;self.job_info=None;self.pool=ThreadPoolExecutor(max_workers=1)
            self.last_clock=None;self.clock_advanced_wall=time.monotonic()
            self.next_plan_wall=0.0;self.next_publish_wall=0.0;self.next_save_wall=0.0
            self.log_times={};self.last_state='STARTING';self.last_report=None
            self.tf_buffer=Buffer(cache_time=Duration(seconds=30.0),node=self)
            self.tf_listener=TransformListener(self.tf_buffer,self)
            latched=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
            # BEST_EFFORT / VOLATILE is compatible with live RELIABLE or BEST_EFFORT costmaps.
            live_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
            self.map_subscription=self.create_subscription(OccupancyGrid,args.map_topic,self.on_map,latched)
            self.coverage_grid_subscription=self.create_subscription(OccupancyGrid,args.coverage_grid_topic,self.on_coverage_grid,latched)
            self.coverage_status_subscription=self.create_subscription(String,args.coverage_status_topic,self.on_coverage_status,latched)
            self.costmap_subscription=self.create_subscription(OccupancyGrid,args.costmap_topic,self.on_costmap,live_qos)
            self.scan_subscription=self.create_subscription(LaserScan,args.scan_topic,self.on_scan,qos_profile_sensor_data)
            self.marker_publisher=self.create_publisher(MarkerArray,'/observation_planner/markers',latched)
            self.status_publisher=self.create_publisher(String,'/observation_planner/status',latched)
            self.get_logger().info(f'[READY] DISPLAY ONLY zone={zone["id"]}; no Nav2 goals/cancel/cmd_vel/map/TF output')
            self.get_logger().info(f'[POLICY] measured target={cfg.threshold*100:.1f}%; predicted gain never changes actual coverage')
            self.get_logger().info('[INPUTS] approved mask + LIVE coverage + base/laser TF + full global costmap + scan geometry')
            self.get_logger().info(f'[OUTPUT] /observation_planner/markers /observation_planner/status; logs={output}')

        def info(self,key,text,seconds=5.0):
            now=time.monotonic()
            if now-self.log_times.get(key,-math.inf)>=seconds:
                self.get_logger().info(text)
                self.log_times[key]=now

        def warn(self,key,text,seconds=5.0):
            now=time.monotonic()
            if now-self.log_times.get(key,-math.inf)>=seconds:
                self.get_logger().warning(text)
                self.log_times[key]=now

        def on_map(self,msg):
            try:ok,reason=verify_map(msg,data['map'])
            except (ValueError,TypeError,AttributeError,OverflowError) as exc:ok,reason=False,str(exc)
            if reason!=self.last_map_reason:
                self.last_map_reason=reason
                if ok:self.get_logger().info('[MAP_OK] approved reference fingerprint matches')
                else:self.get_logger().error('[MAP_MISMATCH] '+reason)
            if self.reference_ok and not ok:self.fault='REFERENCE_MAP_CHANGED_RESTART_REQUIRED'
            self.reference_ok=bool(ok) and self.fault is None
            if not self.reference_ok:self.clear_markers()

        def on_coverage_status(self,msg):
            try:
                status=json.loads(msg.data)
                if status.get('schema')!='hospital_room_coverage_v1' or not isinstance(status.get('session_id'),str):
                    raise ValueError('schema/session missing')
                if status.get('zone_bundle_signature')!=signature:raise ValueError('zone signature mismatch')
                ros=float(status['ros_time_s'])
                if not math.isfinite(ros):raise ValueError('nonfinite time')
            except (ValueError,TypeError,KeyError) as exc:
                self.matched=None;self.coverage_reports.clear();self.coverage_grids.clear()
                self.warn('bad_status',f'[BAD_COVERAGE_STATUS] {exc}');return
            if status['session_id']!=self.session_id:
                self.session_id=status['session_id'];self.coverage_epoch+=1
                self.coverage_grids.clear();self.coverage_reports.clear();self.matched=None
                self.cached_plan=None;self.cached_key=None;self.last_selection=None
                self.clear_markers()
                self.get_logger().info(f'[COVERAGE_SESSION] {self.session_id}; previous selection cleared')
            self.coverage_reports.append((ros,time.monotonic(),status))
            self.match_coverage()

        def on_coverage_grid(self,msg):
            if not geometry_matches(msg,data['map']):
                self.matched=None;self.warn('bad_grid','[COVERAGE_GEOMETRY_MISMATCH] no selection');return
            vals=np.asarray(msg.data,dtype=np.int16).reshape(labels.shape).copy()
            self.coverage_grids.append((stamp_s(msg.header.stamp),time.monotonic(),vals))
            self.match_coverage()

        def match_coverage(self):
            for sstamp,swall,status in reversed(self.coverage_reports):
                for gstamp,gwall,grid in reversed(self.coverage_grids):
                    if abs(sstamp-gstamp)>0.5:continue
                    try:validate_coverage(status,grid,labels,signature,data['zones'])
                    except (ValueError,KeyError,TypeError):continue
                    if self.matched is None or sstamp>=self.matched['stamp']:
                        self.matched=dict(stamp=sstamp,wall=min(swall,gwall),status=status,observed=grid==100)
                    return

        def on_costmap(self,msg):
            if not geometry_matches(msg,data['map']):
                self.costmap_entry=None
                self.warn('bad_costmap','[COSTMAP_GEOMETRY_MISMATCH] full global costmap must match the reference grid/frame; no automatic resampling')
                return
            cost=np.asarray(msg.data,dtype=np.int16).reshape(labels.shape).copy()
            if not np.all((cost>=-1)&(cost<=100)):
                self.costmap_entry=None;return
            # Only changes in unknown/lethal/inscribed affect this geometry model.
            key=array_fingerprint(np.where(cost<0,3,np.where(cost==100,2,np.where(cost==99,1,0))).astype(np.uint8))
            self.costmap_entry=dict(stamp=stamp_s(msg.header.stamp),wall=time.monotonic(),values=cost,key=key)

        def on_scan(self,msg):
            try:
                model=SensorModel(float(msg.angle_min),float(msg.angle_increment),len(msg.ranges),float(msg.range_min),float(msg.range_max))
                model.validate()
                frame=msg.header.frame_id.lstrip('/')
                if not frame:raise ValueError('empty scan frame')
            except (ValueError,TypeError) as exc:
                self.scan_entry=None;self.warn('scan_model',f'[INVALID_SCAN_MODEL] {exc}');return
            self.scan_entry=dict(stamp=stamp_s(msg.header.stamp),wall=time.monotonic(),model=model,frame=frame)

        def latest_pose(self,target,source,check_age=True):
            # Zero-time lookup is intentional for a NEW hypothetical view, not for back-projecting old scans.
            t=self.tf_buffer.lookup_transform(target,source,Time(),timeout=Duration(seconds=0.0))
            p=planar_pose(t.transform,math.radians(5.0))
            age=self.get_clock().now().nanoseconds*1e-9-stamp_s(t.header.stamp)
            if check_age and (age>2.0 or age < -1.5):raise ValueError(f'TF age {age:.2f}s')
            return p

        def inputs(self):
            now=self.get_clock().now().nanoseconds*1e-9;wall=time.monotonic()
            if self.fault:return self.fault,None
            if not self.reference_ok:return 'WAIT_MAP',None
            if wall-self.clock_advanced_wall>5:return 'WAIT_CLOCK_PAUSED',None
            if self.matched is None:return 'WAIT_MATCHED_COVERAGE',None
            s=self.matched['status']
            # A newer HALTED/STALE report must not be hidden by a previous matched LIVE pair.
            if self.coverage_reports:
                latest=self.coverage_reports[-1][2]
                bad=current_input_error(latest,now,wall-self.coverage_reports[-1][1])
                if bad:return bad,None
            bad=current_input_error(s,now,wall-self.matched['wall'])
            if bad:return bad,None
            cost=self.costmap_entry;scan=self.scan_entry
            if cost is None:return 'WAIT_GLOBAL_COSTMAP',None
            if now-cost['stamp']>5 or now-cost['stamp']<-.1 or wall-cost['wall']>15:
                return 'STALE_GLOBAL_COSTMAP',None
            if scan is None:return 'WAIT_SCAN_GEOMETRY',None
            if now-scan['stamp']>3 or now-scan['stamp']<-.1 or wall-scan['wall']>10:
                return 'STALE_SCAN_GEOMETRY',None
            try:
                pose=self.latest_pose('map',args.robot_frame)
                extrinsic=self.latest_pose(args.robot_frame,scan['frame'],False)
            except (TransformException,ValueError) as exc:
                self.warn('tf',f'[WAIT_TF] {exc}; initialize AMCL and inspect remapped TF; no fallback pose')
                return 'WAIT_TF',None
            original=scan['model']
            sensor=SensorModel(original.angle_min,original.angle_increment,original.beams,
                    original.range_min,min(original.range_max,float(s['max_observation_range_m'])),*extrinsic)
            try:sensor.validate()
            except ValueError as exc:return 'INVALID_SENSOR_MODEL',None
            observed=self.matched['observed'];target=labels==zone['mask_id']
            fraction=float(np.count_nonzero(observed & target)/np.count_nonzero(target))
            return None,dict(pose=pose,sensor=sensor,cost=cost,observed=observed,
                    fraction=fraction,status=s,epoch=self.coverage_epoch)

        def clear_markers(self):
            a=MarkerArray();m=Marker();m.header.frame_id='map';m.action=Marker.DELETEALL
            a.markers.append(m);self.marker_publisher.publish(a)

        def publish_status(self,state,more=None):
            report=dict(schema='room_observation_plan_v1',mode='DISPLAY_ONLY_NO_NAVIGATION',
                        state=state,zone_id=zone['id'],coverage_session=self.session_id,
                        ros_time_s=self.get_clock().now().nanoseconds*1e-9,
                        actual_threshold=cfg.threshold,zone_bundle_signature=signature,
                        nav2_plan_checked=False,navigation_commands_sent=False,
                        source='New Python implementation; IPA/NBV concepts only')
            if more:report.update(more)
            self.last_report=report;msg=String();msg.data=json.dumps(report,ensure_ascii=False,allow_nan=False)
            self.status_publisher.publish(msg)
            if state!=self.last_state:
                self.get_logger().info(f'[STATE] {state}')
                self.last_state=state
            if time.monotonic()>=self.next_save_wall:
                atomic_json(output/'latest_plan.json',report);self.next_save_wall=time.monotonic()+3

        def publish_plan(self,plan,inputs,threshold_state):
            stamp=self.get_clock().now().to_msg();arr=MarkerArray()
            clear=Marker();clear.header.frame_id='map';clear.action=Marker.DELETEALL;arr.markers.append(clear)
            def marker(ns,ident,kind,color):
                m=Marker();m.header.frame_id='map';m.header.stamp=stamp;m.ns=ns;m.id=ident
                m.type=kind;m.action=Marker.ADD;m.pose.orientation.w=1.;m.lifetime.sec=8
                m.color.r,m.color.g,m.color.b,m.color.a=color
                return m
            def points(indices,z):
                ids=np.asarray(indices,dtype=np.int32)
                if ids.size>args.max_display_cells:ids=ids[::math.ceil(ids.size/args.max_display_cells)]
                out=[]
                for idx in ids:
                    j,i=divmod(int(idx),geom.width);x,y=geom.cell_to_world(i,j)
                    out.append(Point(x=float(x),y=float(y),z=float(z)))
                return out
            if plan is not None:
                m=marker('candidate_views',0,Marker.SPHERE_LIST,(.05,.65,1.,.85))
                m.scale.x=m.scale.y=m.scale.z=.16
                for c in plan['candidates']:
                    m.points.append(Point(x=float(c['x']),y=float(c['y']),z=.13))
                arr.markers.append(m)
                chosen=plan['selected']
                if chosen is not None:
                    m=marker('selected_view',0,Marker.ARROW,(1.,.2,.08,1.))
                    m.pose.position.x=float(chosen['x']);m.pose.position.y=float(chosen['y']);m.pose.position.z=.20
                    m.pose.orientation.z=math.sin(chosen['yaw']/2);m.pose.orientation.w=math.cos(chosen['yaw']/2)
                    m.scale.x=.65;m.scale.y=.12;m.scale.z=.12;arr.markers.append(m)
                    m=marker('predicted_new_cells_NOT_OBSERVED',0,Marker.CUBE_LIST,(1.,.68,.04,.50))
                    m.scale.x=m.scale.y=geom.resolution*.9;m.scale.z=.01
                    m.points=points(chosen['new_indices'],.11);arr.markers.append(m)
                    m=marker('grid_path_estimate_NOT_NAV2',0,Marker.LINE_STRIP,(.95,.25,.1,.8))
                    m.scale.x=.035
                    for i,j in chosen['path_cells']:
                        x,y=geom.cell_to_world(i,j);m.points.append(Point(x=float(x),y=float(y),z=.12))
                    arr.markers.append(m)
                    self.last_selection=chosen['id']
            m=marker('remaining_target_cells',0,Marker.CUBE_LIST,(.65,.35,.9,.25))
            m.scale.x=m.scale.y=geom.resolution*.85;m.scale.z=.008
            m.points=points(np.flatnonzero((labels==zone['mask_id'])&~inputs['observed']),.08);arr.markers.append(m)
            m=marker('target_status',0,Marker.TEXT_VIEW_FACING,(.12,.08,.2,1.))
            m.scale.z=.34;m.pose.position.x=float(zone['label_map_m'][0]);m.pose.position.y=float(zone['label_map_m'][1]);m.pose.position.z=.65
            m.text=f'{zone["id"]} measured={inputs["fraction"]*100:.1f}% / {cfg.threshold*100:.0f}%\n{threshold_state}\nDISPLAY ONLY'
            arr.markers.append(m);self.marker_publisher.publish(arr)

        def poll(self):
            now=self.get_clock().now().nanoseconds;wall=time.monotonic()
            if self.last_clock is None or now>self.last_clock:self.clock_advanced_wall=wall
            if self.last_clock is not None and now<self.last_clock-100_000_000:
                self.fault='CLOCK_RESET_RESTART_REQUIRED';self.cached_plan=None;self.clear_markers()
            self.last_clock=now
            error,inp=self.inputs()
            if error:
                if wall>=self.next_publish_wall:
                    self.clear_markers();self.publish_status(error)
                    self.info(error,f'[{error}] waiting; not a failed room, no navigation command')
                    self.next_publish_wall=wall+1
                if self.future and self.future.done():
                    try:self.future.result()
                    except Exception as exc:self.warn('job_rejected',f'[PLAN_JOB_DISCARDED] {type(exc).__name__}: {exc}')
                    self.future=None;self.job_info=None
                return
            threshold_state=self.threshold_tracker.update(inp['status'],inp['fraction'])
            if threshold_state in ('OBSERVATION_SUFFICIENT','THRESHOLD_PENDING'):
                if wall>=self.next_publish_wall:
                    self.publish_plan(None,inp,threshold_state)
                    self.publish_status(threshold_state,dict(measured_coverage_percent=100*inp['fraction'],
                            interpretation='Measured room target only, not victim search, cancellation or switching'))
                    self.next_publish_wall=wall+1
                if self.future and self.future.done():
                    try:self.future.result()
                    except Exception as exc:self.warn('threshold_discard',f'[PLAN_JOB_DISCARDED] {type(exc).__name__}: {exc}')
                    self.future=None;self.job_info=None
                self.cached_plan=None;self.cached_key=None;return
            model_key=(inp['epoch'],inp['cost']['key'],tuple(asdict(inp['sensor']).values()))
            if self.future and self.future.done():
                context=self.job_info;self.job_info=None;future=self.future;self.future=None
                try:plan=future.result()
                except Exception as exc:
                    self.warn('planning_error',f'[PLANNING_ERROR] {type(exc).__name__}: {exc}')
                    self.next_plan_wall=wall+3;self.cached_plan=None;return
                moved=math.dist(inp['pose'][:2],context['pose'][:2])
                if context['key']==model_key and moved<=.15:
                    self.cached_plan=plan;self.cached_key=model_key
                    self.info('plan_ready',f'[CANDIDATES] sampled={plan.get("sampled_candidates",0)} reachable={plan.get("reachable_sampled_candidates",0)} with_visibility={len(plan["candidates"])}',0)
                else:self.info('discard','[REPLAN] pose/session/costmap changed during computation; result discarded')
            if self.cached_plan is not None and (self.cached_key!=model_key or math.dist(inp['pose'][:2],self.cached_plan['robot_pose'][:2])>.15):
                self.cached_plan=None;self.clear_markers()
            if self.cached_plan is None and self.future is None and wall>=self.next_plan_wall:
                self.job_info=dict(key=model_key,pose=inp['pose'],started=wall)
                # Immutable copies keep callback updates independent of worker evaluation.
                self.future=self.pool.submit(build_plan,geom,labels,known_free,zone,inp['observed'].copy(),
                                             inp['pose'],inp['sensor'],cfg,inp['cost']['values'].copy())
                self.next_plan_wall=wall+3
            if wall<self.next_publish_wall:return
            self.next_publish_wall=wall+1
            if self.cached_plan is None:
                self.clear_markers();self.publish_status('PLANNING');return
            plan=rerank_plan(self.cached_plan,inp['observed'],geom,cfg,self.last_selection)
            # Partial-FoV orientation is re-optimized when sufficiently more cells become observed.
            previous_observed=self.cached_plan.get('_observed_count_at_build')
            count=int(inp['observed'].sum())
            if previous_observed is None:self.cached_plan['_observed_count_at_build']=count
            if previous_observed is not None and count-previous_observed>100 and not inp['sensor'].full_circle:
                self.cached_plan=None
            self.publish_plan(plan,inp,threshold_state)
            selected=plan['selected']
            extra=reportable(plan);extra['measured_coverage_percent']=100*inp['fraction']
            extra['prediction_warning']='Potential ray visibility, not measured gain or a safe Nav2 goal; finite return behavior, dynamic change and uncertainty remain'
            self.publish_status(plan['status'],extra)
            if selected:
                self.info('selection',f'[SELECTED_VIEW] zone={zone["id"]} xy=({selected["x"]:.2f},{selected["y"]:.2f}) '
                    f'measured={inp["fraction"]*100:.1f}% predicted_new={selected["new_area_m2"]:.2f}m2 '
                    f'(+{selected["potential_gain_pp"]:.1f}pp potential) grid_path={selected["path_est_m"]:.2f}m '
                    'NAV2_NOT_CHECKED; NO navigation command')
            else:self.info('no_candidate',f'[{plan["status"]}] {plan["reason"]}; no threshold completion assumed')

        def close(self):
            self.pool.shutdown(wait=True,cancel_futures=True)
            if self.last_report is not None:atomic_json(output/'latest_plan.json',self.last_report)
            if rclpy.ok():self.clear_markers()

    return ObservationPlanner


def parser():
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zone',default='R14_T')
    p.add_argument('--zones',type=Path,default=root/'config/exploration/hospital_zones.json')
    p.add_argument('--reference',type=Path,default=root/'config/exploration/hospital_coverage_reference.json.gz')
    p.add_argument('--map-topic',default='/j100_0001/map')
    p.add_argument('--coverage-grid-topic',default='/hospital_coverage/grid')
    p.add_argument('--coverage-status-topic',default='/hospital_coverage/status')
    p.add_argument('--costmap-topic',default='/j100_0001/global_costmap/costmap')
    p.add_argument('--scan-topic',default='/j100_0001/sensors/lidar2d_0/scan')
    p.add_argument('--robot-frame',default='base_link')
    p.add_argument('--threshold',type=float,default=.90)
    p.add_argument('--candidate-spacing-m',type=float,default=.60)
    p.add_argument('--outside-room-m',type=float,default=3.)
    p.add_argument('--safety-margin-m',type=float,default=.04)
    p.add_argument('--max-candidates',type=int,default=100)
    p.add_argument('--max-display-cells',type=int,default=15000)
    p.add_argument('--output-dir',type=Path,default=root/'logs/observation_planner')
    p.add_argument('--self-test',action='store_true')
    return p


def main():
    p=parser();args,ros_args=p.parse_known_args()
    if ros_args and ros_args[0]!='--ros-args':p.error('Unknown arguments: '+' '.join(ros_args))
    cfg=Settings(spacing_m=args.candidate_spacing_m,outside_room_m=args.outside_room_m,
                 safety_margin_m=args.safety_margin_m,max_candidates=args.max_candidates,threshold=args.threshold)
    cfg.validate()
    if args.max_display_cells<1:p.error('max-display-cells must be positive')
    bundle=load_bundle(args.zones.expanduser(),args.reference.expanduser())
    if args.zone not in [z['id'] for z in bundle[0]['zones']]:p.error('Unknown target zone '+args.zone)
    if args.self_test:
        import runpy
        tests=Path(__file__).resolve().parents[2]/'tests/test_observation_planning.py'
        if not tests.is_file():p.error('Extract the full ZIP including tests/')
        scope=runpy.run_path(str(tests),run_name='observation_planner_selftest')
        scope['run_tests'](bundle)
        print('[SELF-TEST PASS] logic + approved R14_T bundle; NOT actual ROS2/DDS/Gazebo validation')
        return 0
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
    except ImportError as exc:
        print(f'[ERROR] {exc}; source /opt/ros/humble/setup.bash and use /usr/bin/python3',file=sys.stderr);return 2
    output=args.output_dir.expanduser().resolve()/datetime.now().strftime('session_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    rclpy.init(args=ros_args);node=None
    try:
        cls=build_node_type(args,bundle,cfg,output);node=cls()
        if not node.get_parameter('use_sim_time').value:
            raise ValueError('Gazebo trial requires --ros-args -p use_sim_time:=true')
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.10)
            node.poll()
    except (KeyboardInterrupt,ExternalShutdownException):pass
    finally:
        if node is not None:
            try:node.close()
            finally:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        print(f'[SAVED] {output}\nDisplay-only planner stopped. No motion or cancellation commands were sent.')
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError,ImportError) as exc:
        print(f'[ERROR] {exc}',file=sys.stderr);raise SystemExit(2)
