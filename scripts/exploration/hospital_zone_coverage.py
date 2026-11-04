#!/usr/bin/env python3
"""Read-only live LiDAR coverage for the approved 27 hospital zones (ROS2 Humble).

Does NOT send goals/cancels/velocities, publish TF, or modify Nav2 or /map.
Independent /hospital_coverage/* topics do NOT feed the old 18-sector explorer.
Each execution starts a new observation session; previous data is not restored.

API sources:
https://raw.githubusercontent.com/ros2/common_interfaces/humble/sensor_msgs/msg/LaserScan.msg
https://raw.githubusercontent.com/ros2/geometry2/humble/tf2_ros_py/tf2_ros/buffer.py
https://raw.githubusercontent.com/ros2/rclpy/humble/rclpy/rclpy/node.py
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timezone
import gzip
import io
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np

from hospital_coverage_core import Coverage, load_bundle, planar_pose
from hospital_zone_visualizer import verify_map


def atomic_write(path: Path, content: bytes) -> None:
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec)*1_000_000_000+int(stamp.nanosec)


def build_node_type(args, data, geom, labels, known_free, signature, output_dir):
    """ROS bindings are lazy so core tests can run on systems without ROS."""
    from rclpy.node import Node
    from rclpy.time import Time
    from rclpy.duration import Duration
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from nav_msgs.msg import OccupancyGrid
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from tf2_ros import Buffer, TransformListener, TransformException

    class HospitalCoverageNode(Node):
        def __init__(self):
            super().__init__('hospital_zone_coverage')
            self.coverage=Coverage(geom,labels,known_free,data['zones'],args.threshold)
            self.reference_ok=False
            self.map_message=None
            self.fault=None
            self.last_map_reason=None
            self.pending_scans=deque()
            self.counters=Counter()
            self.last_enqueue_stamp=None
            self.last_received_stamp=None
            self.last_clock_ns=None
            self.last_scan_report={}
            self.last_evidence_end_s=None
            self.next_process=0.0
            self.next_publish=0.0
            self.next_save=0.0
            self.last_report=None
            self.notices={}
            self.reported_thresholds=set()
            self.session_start_wall=datetime.now(timezone.utc).isoformat()
            self.tf_buffer=Buffer(cache_time=Duration(seconds=30.0))
            self.tf_listener=TransformListener(self.tf_buffer,self)
            latched=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.map_subscription=self.create_subscription(OccupancyGrid,args.map_topic,self.on_map,latched)
            self.scan_subscription=self.create_subscription(LaserScan,args.scan_topic,self.on_scan,qos_profile_sensor_data)
            self.grid_publisher=self.create_publisher(OccupancyGrid,'/hospital_coverage/grid',latched)
            self.status_publisher=self.create_publisher(String,'/hospital_coverage/status',latched)
            self.marker_publisher=self.create_publisher(MarkerArray,'/hospital_coverage/markers',latched)
            self.get_logger().info('[READY] READ-ONLY room coverage; no goal/cancel/cmd_vel/map/TF output')
            self.get_logger().info(f'[ZONES] rooms={data["room_count"]} lobby/corridors={data["nonroom_count"]}; '
                                   f'denominator={self.coverage.total_cells} reference-white survey cells')
            self.get_logger().info(f'[TOPICS] map={args.map_topic}; scan={args.scan_topic}; output=/hospital_coverage/*')
            self.get_logger().info(f'[SESSION] NEW, starts at 0%; output={output_dir}')
            self.get_logger().info(f'[POLICY] threshold={args.threshold*100:.1f}% DISPLAY ONLY; '
                                   f'range_cap={args.max_range_m:.1f}m; infinite_ranges={args.use_infinite_ranges}')

        def notice(self,key,text,level='info',interval=5.0):
            now=time.monotonic()
            if now-self.notices.get(key,-math.inf)>=interval:
                if level == 'warning':
                    self.get_logger().warning(text)
                elif level == 'error':
                    self.get_logger().error(text)
                else:
                    self.get_logger().info(text)
                self.notices[key]=now

        def on_map(self,msg):
            try:
                ok,reason=verify_map(msg,data['map'])
            except (ValueError,TypeError,OverflowError,AttributeError) as exc:
                ok,reason=False,str(exc)
            if reason!=self.last_map_reason:
                self.last_map_reason=reason
                if ok:
                    self.get_logger().info('[MAP_OK] geometry, frame and occupancy fingerprint match')
                else:
                    self.get_logger().error('[MAP_MISMATCH] '+reason+'; scans not integrated')
            if not ok:
                if self.reference_ok:
                    self.fault='REFERENCE_MAP_CHANGED_RESTART_REQUIRED'
                self.reference_ok=False
                self.pending_scans.clear()
                return
            if self.fault:
                return
            self.map_message=msg
            self.reference_ok=True
            self.next_publish=0.0

        def on_scan(self,msg):
            self.counters['received_scans']+=1
            if not self.reference_ok or self.fault:
                return
            ns=stamp_ns(msg.header.stamp)
            if ns<=0 or not msg.header.frame_id or not msg.ranges:
                self.counters['invalid_messages']+=1
                return
            if self.last_received_stamp is not None and ns<=self.last_received_stamp:
                self.counters['out_of_order_or_duplicate']+=1
                return
            self.last_received_stamp=ns
            if self.last_enqueue_stamp is not None and ns-self.last_enqueue_stamp<args.scan_period_s*1e9:
                self.counters['throttled_scans']+=1
                return
            if len(msg.ranges)>20000 or not math.isfinite(msg.time_increment) or msg.time_increment<0:
                self.counters['invalid_messages']+=1
                return
            end_ns=ns+round((len(msg.ranges)-1)*msg.time_increment*1e9)
            if (end_ns-ns)*1e-9>args.max_scan_duration_s:
                self.counters['scan_duration_rejected']+=1
                return
            self.pending_scans.append((msg,ns,end_ns))
            self.last_enqueue_stamp=ns
            self.counters['queued_scans']+=1
            while len(self.pending_scans)>64:
                self.pending_scans.popleft()
                self.counters['queue_overflow_drops']+=1

        def process_pending(self):
            if not self.reference_ok or self.fault:
                return
            now_s=self.get_clock().now().nanoseconds*1e-9
            work_start=time.monotonic()
            processed_this_tick=0
            while self.pending_scans and processed_this_tick<2:
                if time.monotonic()-work_start>.10:
                    break
                msg,start_ns,end_ns=self.pending_scans[0]
                frame=msg.header.frame_id.lstrip('/')
                start=Time.from_msg(msg.header.stamp)
                end=start+Duration(nanoseconds=end_ns-start_ns)
                ready=self.tf_buffer.can_transform('map',frame,start,timeout=Duration(seconds=0.0))
                if ready and end_ns!=start_ns:
                    ready=self.tf_buffer.can_transform('map',frame,end,timeout=Duration(seconds=0.0))
                if not ready:
                    if now_s-start_ns*1e-9>args.tf_wait_s:
                        self.pending_scans.popleft()
                        self.counters['tf_timeout_drops']+=1
                        self.notice('tf_drop',f'[DROP_SCAN_TF] timestamped TF unavailable for >{args.tf_wait_s:g} simulation seconds', 'warning')
                        continue
                    self.notice('tf_wait',f'[WAIT_TF_SYNC] map<-{frame}; queued={len(self.pending_scans)} '
                                f'processed={self.coverage.scan_count}; waiting for scan-time TF', 'warning')
                    break
                try:
                    first=self.tf_buffer.lookup_transform('map',frame,start,timeout=Duration(seconds=0.0))
                    last=first if end_ns==start_ns else self.tf_buffer.lookup_transform('map',frame,end,timeout=Duration(seconds=0.0))
                    p0=planar_pose(first.transform,math.radians(args.max_tilt_deg))
                    p1=planar_pose(last.transform,math.radians(args.max_tilt_deg))
                except TransformException:
                    break
                except ValueError as exc:
                    self.pending_scans.popleft()
                    self.counters['tf_pose_rejected']+=1
                    self.notice('bad_pose',f'[SKIP_POSE] {exc}', 'warning')
                    continue
                self.pending_scans.popleft()
                try:
                    report=self.coverage.integrate(msg.ranges,float(msg.angle_min),float(msg.angle_increment),
                            float(msg.range_min),float(msg.range_max),p0,p1,start_ns*1e-9,
                            float(msg.time_increment),args.max_range_m,args.ray_stride,args.use_infinite_ranges)
                except ValueError as exc:
                    self.counters['invalid_messages']+=1
                    self.notice('bad_scan',f'[SKIP_SCAN] {exc}', 'warning')
                    continue
                self.last_scan_report=report
                if report['processed']:
                    processed_this_tick+=1
                    for k in ('new_cells','finite_beams','no_return_beams','invalid_beams','skipped_infinite','sensor_blocked_beams','used_beams'):
                        self.counters[k]+=report[k]
                    if report['used_beams']:
                        self.last_evidence_end_s=end_ns*1e-9
                    else:
                        self.notice('no_rays','[NO_VALID_RAYS] timestamped scan received but no usable rays; inspect ranges/sensor pose', 'warning')
                    for rid in sorted(self.coverage.threshold_confirmed-self.reported_thresholds):
                        z=next(z for z in data['zones'] if z['mask_id']==rid)
                        self.get_logger().info(f'[COVERAGE_THRESHOLD] {z["id"]} reached display threshold; NO cancellation/completion decision')
                        self.reported_thresholds.add(rid)
                else:
                    self.counters['overlapping_scans']+=1

        def make_report(self):
            now_s=self.get_clock().now().nanoseconds*1e-9
            age=None if self.last_evidence_end_s is None else max(0.0,now_s-self.last_evidence_end_s)
            health=('HALTED' if self.fault else 'WAIT_MAP' if not self.reference_ok
                    else 'WAIT_SCAN' if self.last_evidence_end_s is None else 'STALE' if age>args.stale_after_s else 'LIVE')
            stats=self.coverage.statistics()
            return dict(schema='hospital_room_coverage_v1',mode='OBSERVE_ONLY_NO_NAVIGATION',
                        session_id=output_dir.name,session_start_wall_utc=self.session_start_wall,
                        wall_utc=datetime.now(timezone.utc).isoformat(),ros_time_s=now_s,
                        map_validated=self.reference_ok,data_health=health,fault=self.fault,
                        last_evidence_scan_end_s=self.last_evidence_end_s,scan_age_sim_s=age,
                        scans_processed=self.coverage.scan_count,evidence_scans=self.coverage.evidence_scan_count,
                        pending_scans=len(self.pending_scans),diagnostics=dict(self.counters),
                        last_scan=self.last_scan_report,
                        reference_map=data['map'],zone_bundle_signature=signature,
                        zone_count=len(stats),reference_free_survey_cells=self.coverage.total_cells,
                        observed_cells=self.coverage.observed_total,
                        overall_observed_percent=round(100*self.coverage.observed_total/self.coverage.total_cells,4),
                        threshold_fraction=args.threshold,threshold_policy='2 scans separated by >=1 sim second; display only',
                        max_observation_range_m=args.max_range_m,infinite_ranges_used=args.use_infinite_ranges,
                        denominator='approved zone masks intersected with original PGM white=254; gray/exterior excluded',
                        semantics='Ever traversed by accepted lidar rays this session. Not structural verification, current occupancy, or victim-search completion.',
                        zones=stats)

        def publish(self):
            report=self.make_report()
            self.last_report=report
            status=String(); status.data=json.dumps(report,ensure_ascii=False,allow_nan=False)
            self.status_publisher.publish(status)
            if not self.reference_ok or self.fault:
                self.notice('not_ready',f'[{report["data_health"]}] {self.fault or "waiting for correct prior map"}')
                if self.fault:
                    self.clear_markers()
                return
            stamp=self.get_clock().now().to_msg()
            grid=OccupancyGrid(); grid.header.frame_id='map'; grid.header.stamp=stamp
            grid.info=self.map_message.info
            grid.data=self.coverage.grid_values().ravel().tolist()
            self.grid_publisher.publish(grid)
            markers=MarkerArray()
            def marker(ns,ident,typ):
                m=Marker();m.header.frame_id='map';m.header.stamp=stamp
                m.ns=ns;m.id=int(ident);m.type=typ;m.action=Marker.ADD
                m.pose.orientation.w=1.0;m.lifetime.sec=8
                return m
            if not args.no_points:
                m=marker('observed_reference_cells',0,Marker.CUBE_LIST)
                m.scale.x=m.scale.y=geom.resolution*.9;m.scale.z=.008
                m.color.r=.05;m.color.g=.85;m.color.b=.25;m.color.a=.68
                indices=np.flatnonzero(self.coverage.observed.ravel())
                if len(indices)>args.max_marker_cells:
                    indices=indices[::math.ceil(len(indices)/args.max_marker_cells)]
                for index in indices:
                    j,i=divmod(int(index),geom.width);x,y=geom.cell_to_world(i,j)
                    m.points.append(Point(x=float(x),y=float(y),z=.07))
                markers.markers.append(m)
            for z,s in zip(data['zones'],report['zones']):
                m=marker('zone_coverage_labels',z['mask_id'],Marker.TEXT_VIEW_FACING)
                m.pose.position.x=float(z['label_map_m'][0]);m.pose.position.y=float(z['label_map_m'][1])
                m.pose.position.z=.3;m.scale.z=args.text_height
                if z['id'].startswith('CONNECTOR'):
                    m.scale.z*=.72
                m.color.r=.05;m.color.g=.12;m.color.b=.20;m.color.a=1.0
                ident=z['id'].replace('CORRIDOR_','C-').replace('CONNECTOR_','LINK-').replace('_','-')
                suffix=' *' if s['observation_state']=='THRESHOLD_REACHED' else ''
                m.text=f'{ident}\n{s["coverage_percent"]:.1f}%{suffix}'
                markers.markers.append(m)
            self.marker_publisher.publish(markers)
            self.notice('coverage',f'[COVERAGE] health={report["data_health"]} scans={report["scans_processed"]} '
                        f'overall={report["overall_observed_percent"]:.2f}% '
                        f'({report["observed_cells"]}/{report["reference_free_survey_cells"]} cells) '
                        f'queued={len(self.pending_scans)} tf_dropped={self.counters["tf_timeout_drops"]}')
            active=sorted([s for s in report['zones'] if s['observed_cells']],key=lambda s:s['observed_cells'],reverse=True)[:8]
            if active:
                self.notice('zone_values','[ROOMS] '+' | '.join(f'{s["id"]}={s["coverage_percent"]:.1f}%' for s in active))
            elif self.last_evidence_end_s is None:
                self.notice('no_scan','[WAIT_SCAN] no valid scan-time TF/rays yet; prior map alone never counts as observed')
            if report['data_health']=='STALE':
                self.notice('stale','[STALE_OBSERVATIONS] retained historical coverage; fresh observation unavailable', 'warning')

        def save(self):
            report=self.make_report()
            atomic_write(output_dir/'coverage_status.json',json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False).encode())
            table=io.StringIO(newline='')
            fields=list(report['zones'][0])
            writer=csv.DictWriter(table,fieldnames=fields);writer.writeheader();writer.writerows(report['zones'])
            atomic_write(output_dir/'zone_coverage.csv',table.getvalue().encode('utf-8-sig'))
            snapshot=dict(schema='observed_mask_ros_row_runs_v1',width=geom.width,height=geom.height,
                          row_order='ROS bottom-up',zone_bundle_signature=signature,
                          observed_cells=self.coverage.observed_total,runs=self.coverage.observed_runs(),
                          resume_supported=False)
            atomic_write(output_dir/'observed_mask.json.gz',gzip.compress(json.dumps(snapshot,separators=(',',':')).encode(),mtime=0))

        def clear_markers(self):
            arr=MarkerArray()
            for ns,ids in [('observed_reference_cells',[0]),('zone_coverage_labels',[z['mask_id'] for z in data['zones']])]:
                for ident in ids:
                    m=Marker();m.header.frame_id='map';m.ns=ns;m.id=int(ident);m.action=Marker.DELETE
                    arr.markers.append(m)
            self.marker_publisher.publish(arr)

        def poll(self):
            now_ns=self.get_clock().now().nanoseconds
            if self.last_clock_ns is not None and now_ns<self.last_clock_ns-100_000_000 and not self.fault:
                self.fault='SIMULATION_CLOCK_MOVED_BACKWARD_RESTART_REQUIRED'
                self.pending_scans.clear()
                self.tf_buffer.clear()
                self.get_logger().error('[CLOCK_RESET] measurement halted; restart for a new mission')
            self.last_clock_ns=now_ns
            wall=time.monotonic()
            if wall>=self.next_process:
                self.next_process=wall+.1
                self.process_pending()
            if wall>=self.next_publish:
                self.next_publish=wall+args.publish_seconds
                self.publish()
            if wall>=self.next_save:
                self.next_save=wall+5.0
                self.save()

    return HospitalCoverageNode


def parser():
    root=Path(__file__).resolve().parents[2]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zones',type=Path,default=root/'config/exploration/hospital_zones.json')
    ap.add_argument('--reference',type=Path,default=root/'config/exploration/hospital_coverage_reference.json.gz')
    ap.add_argument('--map-topic',default='/j100_0001/map')
    ap.add_argument('--scan-topic',default='/j100_0001/sensors/lidar2d_0/scan')
    ap.add_argument('--output-dir',type=Path,default=root/'logs/hospital_coverage')
    ap.add_argument('--threshold',type=float,default=.75)
    ap.add_argument('--max-range-m',type=float,default=8.0)
    ap.add_argument('--scan-period-s',type=float,default=.5)
    ap.add_argument('--ray-stride',type=int,default=1)
    ap.add_argument('--use-infinite-ranges',action='store_true',help='Opt-in only after confirming sensor +inf means no hit')
    ap.add_argument('--tf-wait-s',type=float,default=8.0,help='TF waiting limit in ROS/simulation time, NOT mission duration')
    ap.add_argument('--stale-after-s',type=float,default=8.0)
    ap.add_argument('--max-scan-duration-s',type=float,default=.5)
    ap.add_argument('--max-tilt-deg',type=float,default=5.0)
    ap.add_argument('--publish-seconds',type=float,default=2.0,help='Wall-time diagnostics/visualization period')
    ap.add_argument('--max-marker-cells',type=int,default=10000)
    ap.add_argument('--text-height',type=float,default=.42)
    ap.add_argument('--no-points',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    return ap


def main():
    ap=parser();args,ros_args=ap.parse_known_args()
    if ros_args and ros_args[0]!='--ros-args':
        ap.error('Unrecognized arguments: '+' '.join(ros_args))
    if not 0<args.threshold<=1 or args.ray_stride<1 or args.max_marker_cells<1:
        ap.error('Invalid threshold/stride/marker count')
    for name in ('max_range_m','scan_period_s','tf_wait_s','stale_after_s','max_scan_duration_s','max_tilt_deg','publish_seconds','text_height'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0:
            ap.error(f'{name} must be positive and finite')
    args.zones=args.zones.expanduser().resolve();args.reference=args.reference.expanduser().resolve()
    data,geom,labels,known_free,signature=load_bundle(args.zones,args.reference)
    if args.self_test:
        tests=Path(__file__).resolve().parents[2]/'tests/test_hospital_coverage.py'
        if not tests.is_file():
            raise ValueError(f'Test file missing: {tests}; extract the full setup ZIP')
        import runpy
        scope=runpy.run_path(str(tests),run_name='hospital_coverage_selftest')
        scope['run_tests'](data,geom,labels,known_free)
        print('[SELF-TEST PASS] approved 27-zone masks and coverage logic; NOT live ROS2/Gazebo validation')
        return 0
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
    except ImportError as exc:
        print(f'[ERROR] {exc}; source /opt/ros/humble/setup.bash, then use /usr/bin/python3',file=sys.stderr)
        return 2
    output=args.output_dir.expanduser().resolve()/datetime.now().strftime('session_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    rclpy.init(args=ros_args)
    node=None
    try:
        cls=build_node_type(args,data,geom,labels,known_free,signature,output)
        node=cls()
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.1)
            node.poll()
    except (KeyboardInterrupt,ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.save()
                if rclpy.ok():
                    node.clear_markers()
            finally:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(f'[SAVED] {output}\nRead-only monitor stopped. No navigation commands were sent.')
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,KeyError,ImportError) as exc:
        print(f'[ERROR] {exc}',file=sys.stderr)
        raise SystemExit(2)
