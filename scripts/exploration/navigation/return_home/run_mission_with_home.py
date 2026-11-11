#!/usr/bin/env python3
"""DN-011 C6 / Bedless v2.4: 96% zone exploration followed by optional return-home.

External supervised coordinator: no edits to the tested mission manager, map,
Coverage node or Nav2 settings. Requires ROS2 Humble / Gazebo simulation.

Home is the initial stable map->base_link TF pose, NOT the Gazebo spawn arguments.
Return can occur after a verified FULL or explicitly PARTIAL terminal mission.
Return safety is separately checked; an exploration PARTIAL result stays PARTIAL.
All unknown / timeout / canceled conditions fail closed. Does not certify safety.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from return_home_policy import (
    check_mission_complete, evaluate_return_eligibility, pose_distance, parse_cli_param_output, parse_saved_session_path
)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException

PROJECT_NAME = 'ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping'
STOP_FLAG = False


def on_interrupt(_signal, _frame):
    global STOP_FLAG
    STOP_FLAG = True
    print('\n[RTH] Interrupt requested; stop mission / cancel owned return goal.', flush=True)


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def yaw_quat(q):
    return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))


def transform_pose(t):
    q = t.transform.rotation
    return {'x': float(t.transform.translation.x), 'y': float(t.transform.translation.y),
            'yaw': yaw_quat(q), 'qx': float(q.x), 'qy': float(q.y),
            'qz': float(q.z), 'qw': float(q.w)}


def json_save(path: Path, data: dict):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def angular_diff(a, b):
    return abs(math.atan2(math.sin(a-b), math.cos(a-b)))


class StopNow(Exception):
    pass


class HomeSupervisor(Node):
    def __init__(self):
        super().__init__('dn011_return_home_supervisor',
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.buf = Buffer(cache_time=Duration(seconds=15.0), node=self)
        self.listener = TransformListener(self.buf, self)
        self.odom_samples = deque(maxlen=60)
        self.map_msg = None
        self.cost_msg = None
        self.cost_wall = 0.0
        self.goal_status_seen = False
        self.goal_status_wall = 0.0
        self.active_nav_ids = set()
        self.recoveries = 0
        self.last_clock = None
        transient = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        volatile = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                              durability=DurabilityPolicy.VOLATILE)
        status_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/j100_0001/map',
                                 lambda msg: setattr(self, 'map_msg', msg), transient)
        self.create_subscription(OccupancyGrid, '/j100_0001/global_costmap/costmap',
                                 self.on_costmap, volatile)
        self.create_subscription(Odometry, '/j100_0001/platform/odom/filtered',
                                 self.on_odom, volatile)
        self.create_subscription(GoalStatusArray, '/j100_0001/navigate_to_pose/_action/status',
                                 self.on_status, status_qos)
        self.plan_client = ActionClient(self, ComputePathToPose, '/j100_0001/compute_path_to_pose')
        self.nav_client = ActionClient(self, NavigateToPose, '/j100_0001/navigate_to_pose')

    def on_costmap(self, msg):
        self.cost_msg = msg
        self.cost_wall = time.monotonic()

    def on_odom(self, msg):
        v = msg.twist.twist
        self.odom_samples.append((time.monotonic(), float(v.linear.x),
                                  float(v.linear.y), float(v.angular.z)))

    def on_status(self, msg):
        self.goal_status_seen = True
        self.goal_status_wall = time.monotonic()
        self.active_nav_ids = {bytes(s.goal_info.goal_id.uuid).hex() for s in msg.status_list
                               if s.status in (GoalStatus.STATUS_ACCEPTED,
                                               GoalStatus.STATUS_EXECUTING,
                                               GoalStatus.STATUS_CANCELING)}

    def tick(self, timeout=0.1, ignore_stop=False):
        if STOP_FLAG and not ignore_stop:
            raise StopNow('USER_INTERRUPT')
        rclpy.spin_once(self, timeout_sec=timeout)
        stamp = self.get_clock().now().nanoseconds
        if stamp > 1_000_000_000:
            if self.last_clock is not None and stamp < self.last_clock - 1_000_000_000:
                raise RuntimeError('ROS_CLOCK_RESET; home pose/session no longer valid')
            self.last_clock = stamp

    def wait_for(self, condition, seconds, description):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            self.tick()
            if condition():
                return
        raise RuntimeError('TIMEOUT_'+description)

    def await_future(self, future, seconds, description, *, ignore_stop=False):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            self.tick(ignore_stop=ignore_stop)
            if future.done():
                return future.result()
        raise RuntimeError('TIMEOUT_'+description)

    def tf_pose(self):
        try:
            tf = self.buf.lookup_transform('map', 'base_link', Time())
            # Do not lock a stale transform from a previous simulation/session.
            now_ns = self.get_clock().now().nanoseconds
            stamp_ns = tf.header.stamp.sec * 1_000_000_000 + tf.header.stamp.nanosec
            if stamp_ns <= 0 or now_ns <= 0 or abs(now_ns-stamp_ns) > 3_000_000_000:
                return None
            pose = transform_pose(tf)
            if not all(math.isfinite(v) for v in pose.values()):
                return None
            return pose
        except TransformException:
            return None

    def stable_home(self):
        self.wait_for(lambda: self.map_msg is not None and self.cost_msg is not None
                      and self.tf_pose() is not None
                      and self.get_clock().now().nanoseconds > 1_000_000_000,
                      35.0, 'MAP_COSTMAP_TF')
        samples = []
        deadline = time.monotonic()+8.0
        last = 0.0
        while time.monotonic() < deadline:
            self.tick()
            if time.monotonic()-last >= 0.3:
                pose = self.tf_pose()
                if pose:
                    samples.append(pose)
                    last = time.monotonic()
                if len(samples) >= 5:
                    break
        if len(samples) < 5:
            raise RuntimeError('INITIAL_MAP_POSE_NOT_STABLE')
        home = samples[-1]
        if any(pose_distance(home, p)[0] > 0.08 or angular_diff(home['yaw'],p['yaw']) > 0.12
               for p in samples):
            raise RuntimeError('INITIAL_MAP_POSE_MOVING_OR_UNSTABLE')
        self.wait_for(self.stationary, 10, 'INITIAL_ODOM_STOP')
        if not self.footprint_clear(home):
            raise RuntimeError('INITIAL_HOME_FOOTPRINT_NOT_FREE; do not save an unsafe home')
        return home

    def stationary(self):
        t = time.monotonic()
        samples = [v for v in self.odom_samples if t-v[0] <= 1.5]
        return (len(samples) >= 3 and samples[-1][0]-samples[0][0] >= 0.30
                and all(math.hypot(v[1],v[2]) <= 0.03 and abs(v[3]) <= 0.05
                        for v in samples))

    @staticmethod
    def grid_value(msg, x, y):
        if msg is None:
            return None
        m = msg.info
        if m.resolution <= 0 or m.width <= 0 or m.height <= 0 or len(msg.data)!=m.width*m.height:
            return None
        q = m.origin.orientation
        origin_yaw = yaw_quat(q)
        dx,dy = x-m.origin.position.x, y-m.origin.position.y
        c,s = math.cos(origin_yaw), math.sin(origin_yaw)
        i = math.floor((c*dx+s*dy)/m.resolution)
        j = math.floor((-s*dx+c*dy)/m.resolution)
        if i<0 or j<0 or i>=m.width or j>=m.height:
            return None
        return int(msg.data[j*m.width+i])

    def fresh_costmap(self):
        return self.cost_msg is not None and time.monotonic()-self.cost_wall < 2.5

    def footprint_clear(self, pose):
        # Conservative endpoint sampling of J100 rectangular footprint + 0.04m margin.
        # Does NOT certify the swept robot footprint along a dynamic Nav2 trajectory.
        if self.map_msg is None or not self.fresh_costmap():
            return False
        a,b = 0.229+0.04, 0.208+0.04
        c,s = math.cos(pose['yaw']), math.sin(pose['yaw'])
        step = 0.025
        n_x = math.ceil(2*a/step); n_y = math.ceil(2*b/step)
        for k in range(n_x+1):
            lx = -a + k*2*a/n_x
            for l in range(n_y+1):
                ly = -b + l*2*b/n_y
                x = pose['x'] + c*lx-s*ly
                y = pose['y'] + s*lx+c*ly
                stat = self.grid_value(self.map_msg, x, y)
                cost = self.grid_value(self.cost_msg, x, y)
                if stat != 0 or cost is None or cost < 0 or cost >= 100:
                    return False
        return True

    def parameters(self, node, names):
        """Read C6 settings using ROS2 CLI in an isolated process.

        ROS2 Humble live testing showed this supervisor's own asynchronous
        get_parameters service future could time out despite ros2 param get
        working from the terminal. Using the same CLI path avoids starving
        the supervisor's own single-threaded executor. Any failure is fatal:
        navigation is never authorized by assuming parameters are correct.
        """
        result = []
        for name in names:
            command = ['ros2', 'param', 'get', node, name]
            try:
                reply = subprocess.run(command, capture_output=True, text=True,
                                       timeout=22, check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f'TIMEOUT_CLI_PARAM_{node.split("/")[-1]}_{name}') from exc
            if reply.returncode != 0:
                raise RuntimeError('CLI_PARAM_UNAVAILABLE_' + node.split('/')[-1] + '_'
                                   + name + ': ' + (reply.stderr or reply.stdout).strip()[:300])
            try:
                value = parse_cli_param_output(name, reply.stdout)
            except ValueError as exc:
                raise RuntimeError('CLI_PARAM_PARSE_FAILED_' + node.split('/')[-1]
                                   + '_' + name + ': ' + str(exc)) from exc
            result.append(value)
        return result

    def validate_c6(self):
        plugin, cm = self.parameters('/j100_0001/planner_server',
                                     ['GridBased.plugin','GridBased.cost_travel_multiplier'])
        (radius,) = self.parameters('/j100_0001/global_costmap/global_costmap',
                                    ['inflation_layer.inflation_radius'])
        vmax,vmin = self.parameters('/j100_0001/velocity_smoother',
                                     ['max_velocity','min_velocity'])
        values = {'plugin': plugin, 'cost_travel_multiplier': cm,
                  'inflation_radius': radius,
                  'max_velocity': vmax, 'min_velocity': vmin}
        if (values['plugin']!='nav2_smac_planner/SmacPlanner2D'
            or abs(values['cost_travel_multiplier']-3.0)>1e-3
            or abs(values['inflation_radius']-0.7)>1e-3
            or len(values['max_velocity'])!=3 or len(values['min_velocity'])!=3
            or any(abs(a-b)>1e-3 for a,b in zip(values['max_velocity'],[1.,0.,1.5]))
            or any(abs(a-b)>1e-3 for a,b in zip(values['min_velocity'],[-.26,0.,-1.5]))):
            raise RuntimeError('C6_PARAMS_MISMATCH: '+json.dumps(values))
        return values

    def plan_home(self, home):
        if not self.plan_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('PLANNER_ACTION_UNAVAILABLE')
        self.wait_for(self.stationary, 10, 'PRE_RETURN_STOP')
        self.wait_for(self.fresh_costmap, 6, 'FRESH_GLOBAL_COSTMAP')
        if not self.footprint_clear(home):
            raise RuntimeError('HOME_FOOTPRINT_BLOCKED_OR_UNKNOWN')
        if not self.goal_status_seen or self.active_nav_ids:
            # Status is required to ensure the previous navigator has no active goals.
            self.wait_for(lambda: self.goal_status_seen and not self.active_nav_ids,
                          12, 'NO_ACTIVE_NAV_GOALS')
        current = self.tf_pose()
        if current is None:
            raise RuntimeError('CURRENT_POSE_UNAVAILABLE')
        d,yaw = pose_distance(current, home)
        if d < 0.15 and yaw < 0.20:
            return {'already_home': True, 'poses': 0, 'path_length_m': 0.0}
        goal = ComputePathToPose.Goal()
        goal.goal = self.as_pose_stamped(home)
        goal.planner_id = 'GridBased'
        goal.use_start = False
        sent = self.await_future(self.plan_client.send_goal_async(goal), 15, 'PLAN_ACCEPT')
        if not sent.accepted:
            raise RuntimeError('RETURN_PATH_REJECTED')
        response = self.await_future(sent.get_result_async(), 35, 'PLAN_RESULT')
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError('RETURN_PATH_FAILED_STATUS_'+str(response.status))
        if getattr(response.result,'error_code',0):
            raise RuntimeError('RETURN_PATH_ERROR_'+str(response.result.error_code))
        path = response.result.path
        poses = path.poses
        if path.header.frame_id.lstrip('/') != 'map' or len(poses)<2:
            raise RuntimeError('RETURN_PATH_EMPTY_OR_WRONG_FRAME')
        if math.hypot(poses[0].pose.position.x-current['x'],
                      poses[0].pose.position.y-current['y']) > 0.65:
            raise RuntimeError('RETURN_PATH_START_MISMATCH')
        length=0.0
        for p,q in zip(poses[:-1],poses[1:]):
            xy = (p.pose.position.x,p.pose.position.y,q.pose.position.x,q.pose.position.y)
            if not all(math.isfinite(v) for v in xy):
                raise RuntimeError('RETURN_PATH_NONFINITE')
            gap=math.hypot(xy[2]-xy[0],xy[3]-xy[1])
            if gap > 0.90:
                raise RuntimeError('RETURN_PATH_DISCONTINUITY')
            length+=gap
        if not self.fresh_costmap() or not self.footprint_clear(home):
            raise RuntimeError('RETURN_MAP_STALE_OR_GOAL_UNSAFE_AFTER_PLAN')
        return {'already_home':False,'poses':len(poses),'path_length_m':round(length,3)}

    def as_pose_stamped(self,home):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = home['x']
        pose.pose.position.y = home['y']
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = home['qx']
        pose.pose.orientation.y = home['qy']
        pose.pose.orientation.z = home['qz']
        pose.pose.orientation.w = home['qw']
        return pose

    def feedback(self, msg):
        self.recoveries = max(self.recoveries, int(msg.feedback.number_of_recoveries))

    def nav_home(self, home, timeout):
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('NAV_ACTION_UNAVAILABLE')
        if not self.fresh_costmap() or not self.footprint_clear(home):
            raise RuntimeError('PRE_NAV_HOME_UNSAFE')
        if self.active_nav_ids:
            raise RuntimeError('ANOTHER_ACTIVE_NAV_GOAL')
        goal = NavigateToPose.Goal()
        goal.pose = self.as_pose_stamped(home)
        send_future = self.nav_client.send_goal_async(goal,feedback_callback=self.feedback)
        try:
            handle = self.await_future(send_future,15.0,'NAV_ACCEPT')
        except StopNow:
            # If Ctrl+C arrives before acceptance, the goal may still be accepted
            # asynchronously. Resolve this handshake and cancel only this goal.
            try:
                late = self.await_future(send_future,15.0,'NAV_ACCEPT_AFTER_INTERRUPT',ignore_stop=True)
                if late.accepted:
                    self.await_future(late.cancel_goal_async(),12,'NAV_LATE_CANCEL',ignore_stop=True)
                    self.await_future(late.get_result_async(),15,'NAV_LATE_CANCEL_TERMINAL',ignore_stop=True)
                    deadline=time.monotonic()+8.0
                    while time.monotonic()<deadline and not self.stationary():
                        self.tick(ignore_stop=True)
                    if not self.stationary():
                        print('[RETURN_STOP_UNCONFIRMED] verify stop manually',flush=True)
            except Exception as exc:
                print('[RETURN_ACCEPT_OR_CANCEL_UNCONFIRMED]',str(exc),flush=True)
            raise
        if not handle.accepted:
            raise RuntimeError('RETURN_NAV_REJECTED')
        try:
            reply = self.await_future(handle.get_result_async(),timeout,'RETURN_NAV_RESULT')
            if reply.status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError('RETURN_NAV_NOT_SUCCEEDED_status_'+str(reply.status))
        except (StopNow, RuntimeError):
            print('[RETURN_CANCEL_REQUESTED] canceling only our owned goal',flush=True)
            try:
                response = self.await_future(handle.cancel_goal_async(),12,'RETURN_CANCEL_ACK',ignore_stop=True)
                if not response.goals_canceling:
                    print('[RETURN_CANCEL_UNCONFIRMED] no goals_canceling',flush=True)
                self.await_future(handle.get_result_async(),15,'RETURN_CANCEL_TERMINAL',ignore_stop=True)
                deadline=time.monotonic()+8
                while time.monotonic()<deadline and not self.stationary():
                    self.tick(ignore_stop=True)
                if not self.stationary():
                    print('[RETURN_STOP_UNCONFIRMED] verify actual robot stop',flush=True)
            except Exception as exc:
                print('[RETURN_CANCEL_OR_STOP_UNCONFIRMED]',str(exc),flush=True)
            raise
        self.wait_for(self.stationary, 12,'POST_RETURN_STOP')
        pose=self.tf_pose()
        if pose is None:
            raise RuntimeError('POST_RETURN_TF_UNAVAILABLE')
        dist, angle=pose_distance(pose,home)
        if dist > 0.25 or angle > 0.35:
            raise RuntimeError(f'POST_RETURN_POSE_OUTSIDE_TOLERANCE xy={dist:.3f} yaw={angle:.3f}')
        return {'arrival_xy_error_m':round(dist,3),
                'arrival_yaw_error_rad':round(angle,3),
                'return_recoveries':self.recoveries}


def mission_process(node, root, options, output, record):
    script=root/'scripts/exploration/navigation/run_exploration.sh'
    if not script.is_file():
        raise RuntimeError('MISSING_MISSION_SCRIPT: '+str(script))
    cmd=['bash',str(script),options.mode,'--threshold','0.96',
         '--max-goals',str(options.max_goals),
         '--max-plan-attempts',str(options.max_plan_attempts)]
    if options.connectivity_aware:
        cmd += ['--connectivity-aware']
        if options.path_blockage_check:
            cmd += ['--path-blockage-check','--path-topic',options.path_topic]
            if options.prior_topology_fixed:cmd += ['--prior-topology-fixed']
        for name in ('zone_hard_goals','audit_spacing_m','audit_headings','audit_max_positions',
                     'audit_max_probes','audit_build_budget_s','audit_zone_budget_s','audit_max_restarts',
                     'entry_access_max_probes','entry_access_per_component',
                     'entry_access_min_separation_m','entry_access_failed_radius_m'):
            cmd += ['--'+name.replace('_','-'),str(getattr(options,name))]
        if options.continue_on_nav_abort:cmd += ['--continue-on-nav-abort']
    if options.mission_budget_guard:
        cmd += ['--mission-budget-guard']
        for name in ('guard_stall_window_s','guard_stall_extent_m','guard_stall_recoveries',
                     'guard_goal_recovery_limit','guard_zone_cancel_limit','guard_zone_work_base_s',
                     'guard_zone_work_max_s','guard_extension_window_s','guard_extension_gain_pp'):
            cmd += ['--'+name.replace('_','-'),str(getattr(options,name))]
    if options.connectivity_aware:
        cmd += ['--audit-max-view-preflights',str(options.audit_max_view_preflights)]
    record['mission_command']=cmd
    record['mission_started_utc']=now_utc()
    json_save(output/'report.json',record)
    print('[MISSION_START]', ' '.join(cmd),flush=True)
    saved=None
    log_path=output/'exploration_console.log'
    with log_path.open('w',encoding='utf-8') as sink:
        proc=subprocess.Popen(cmd,cwd=root,stdout=sink,stderr=subprocess.STDOUT,
                              start_new_session=True)
        with log_path.open('r',encoding='utf-8',errors='replace') as source:
            canceled=False
            interrupt_wall=None
            try:
                while proc.poll() is None:
                    node.tick(timeout=0.15,ignore_stop=True)
                    for line in source.readlines():
                        print(line,end='',flush=True)
                        extracted=parse_saved_session_path(line)
                        if extracted:saved=extracted
                    if STOP_FLAG and not canceled:
                        canceled=True
                        interrupt_wall=time.monotonic()
                        print('[MISSION_CANCEL_REQUESTED] SIGINT to mission process group',flush=True)
                        os.killpg(proc.pid,signal.SIGINT)
                    if canceled and time.monotonic()-interrupt_wall>20:
                        print('[MISSION_SHUTDOWN_UNCONFIRMED] no return will be sent',flush=True)
                        raise StopNow('MISSION_CANCEL_UNCONFIRMED')
            finally:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid,signal.SIGINT)
                        proc.wait(timeout=15)
                    except (ProcessLookupError,subprocess.TimeoutExpired):
                        print('[MISSION_STOP_UNCONFIRMED] inspect navigation action/physical stop; no return',flush=True)
            for line in source.readlines():
                print(line,end='',flush=True)
                extracted=parse_saved_session_path(line)
                if extracted:saved=extracted
            code=proc.returncode
    record['mission_exit_code']=code
    record['mission_finished_utc']=now_utc()
    record['mission_saved_dir']=saved
    json_save(output/'report.json',record)
    if STOP_FLAG:
        raise StopNow('MISSION_USER_INTERRUPT')
    if code!=0:
        raise RuntimeError('MISSION_EXIT_CODE_'+str(code))
    if not saved:
        raise RuntimeError('MISSION_SUMMARY_PATH_NOT_REPORTED; no return')
    session=Path(saved).resolve()
    expected=(root/'logs/hospital_zone_mission').resolve()
    if session.parent!=expected or not session.name.startswith('session_'):
        raise RuntimeError('UNEXPECTED_SESSION_OUTPUT_PATH')
    manifest_file=session/'run_manifest.json'
    summary_file=session/'summary.json'
    if not manifest_file.is_file() or not summary_file.is_file():
        raise RuntimeError('MISSION_MANIFEST_OR_SUMMARY_MISSING')
    manifest=json.loads(manifest_file.read_text())
    summary=json.loads(summary_file.read_text())
    if options.mode == 'dry-run':
        if (summary.get('state') != 'DRY_RUN_PASS' or int(summary.get('sent', -1)) != 0
                or manifest.get('options', {}).get('execute', True)):
            raise RuntimeError('DRY_RUN_EXPECTED_NO_GOALS: '+str(summary.get('state')))
        record['status']='DRY_RUN_PASS_NO_RETURN'
        json_save(output/'report.json',record)
        print('[DRY_RUN_PASS] no navigation and no return requested',flush=True)
        return summary
    eligibility = evaluate_return_eligibility(manifest, summary)
    record['mission_verification'] = {
        'success': eligibility['exploration'] == 'FULL',
        'return_eligible': eligibility['eligible'],
        'exploration_class': eligibility['exploration'],
        'detail': eligibility['detail'],
        'partial_zones': eligibility['partial_zones'],
        'state': summary.get('state'), 'sent': summary.get('sent')}
    json_save(output/'report.json', record)
    if not eligibility['eligible']:
        raise RuntimeError('MISSION_NOT_SAFE_TO_RETURN: ' + eligibility['detail'])
    return summary



def load_prior_mission_for_return(root: Path, report_file: Path, node: HomeSupervisor,
                                  output: Path, record: dict):
    """Allow return without repeating 52 exploration goals only with same-session evidence.

    Never trusts just a prior exit code, a filesystem report, or a stored home.
    Live TF/clock/end pose checks complement Nav2 preflight, not replace it.
    """
    source = report_file.expanduser().resolve()
    expected = (root/'logs/dn011_return_home96').resolve()
    if (source.name != 'report.json' or source.parent.parent != expected
            or not source.parent.name.startswith('run_') or not source.is_file()):
        raise RuntimeError('RETURN_ONLY_UNAPPROVED_SOURCE_REPORT')
    prior = json.loads(source.read_text())
    if prior.get('mode') != 'execute' or prior.get('mission_exit_code') != 0:
        raise RuntimeError('RETURN_ONLY_SOURCE_MISSION_NOT_CLEANLY_TERMINATED')
    if prior.get('status') != 'STOPPED_WITHOUT_RETURN' or not str(prior.get('error','')).startswith('MISSION_NOT_FULLY_COMPLETE:'):
        raise RuntimeError('RETURN_ONLY_SOURCE_NOT_EXPECTED_PARTIAL_TERMINAL')
    if prior.get('return_plan') or prior.get('return_result'):
        raise RuntimeError('RETURN_ONLY_SOURCE_PREVIOUS_RETURN_ATTEMPT')
    finished = prior.get('mission_finished_utc')
    try:
        finished_dt = datetime.fromisoformat(finished)
        age = (datetime.now(timezone.utc) - finished_dt).total_seconds()
        if not 0 <= age <= 10800:
            raise ValueError('source mission older than 3h')
    except (ValueError, TypeError):
        raise RuntimeError('RETURN_ONLY_SOURCE_TOO_OLD_OR_UNDATED')
    home = prior.get('home_pose_map')
    if not isinstance(home, dict) or any(k not in home for k in ('x','y','yaw','qx','qy','qz','qw')):
        raise RuntimeError('RETURN_ONLY_HOME_POSE_MISSING')
    if not all(math.isfinite(float(home[k])) for k in ('x','y','yaw','qx','qy','qz','qw')):
        raise RuntimeError('RETURN_ONLY_HOME_POSE_INVALID')
    session = Path(prior.get('mission_saved_dir') or '').resolve()
    mission_base = (root/'logs/hospital_zone_mission').resolve()
    if session.parent != mission_base or not session.name.startswith('session_'):
        raise RuntimeError('RETURN_ONLY_UNEXPECTED_SESSION_PATH')
    manifest = json.loads((session/'run_manifest.json').read_text())
    summary = json.loads((session/'summary.json').read_text())
    eligibility = evaluate_return_eligibility(manifest, summary)
    if not eligibility['eligible']:
        raise RuntimeError('RETURN_ONLY_MISSION_NOT_ELIGIBLE: '+eligibility['detail'])
    # The old v1.3 session report lacks a captured ROS clock baseline.
    # Reject reset/advanced clocks, and a robot that is no longer near the last
    # successfully reached observation Goal (which was not canceled in this case).
    node.wait_for(lambda: node.tf_pose() is not None and node.map_msg is not None
                  and node.fresh_costmap(), 30, 'RETURN_ONLY_MAP_COSTMAP_TF')
    now_ros = node.get_clock().now().nanoseconds/1e9
    summary_ros = float(summary.get('ros_time_s',float('nan')))
    if not math.isfinite(summary_ros) or not 0 <= now_ros-summary_ros <= 7200:
        raise RuntimeError(f'RETURN_ONLY_CLOCK_SESSION_MISMATCH now={now_ros:.1f} mission={summary_ros:.1f}')
    visits = summary.get('visits')
    successful = [v for v in visits if isinstance(v,dict) and v.get('nav_status')=='SUCCEEDED'] if isinstance(visits,list) else []
    if not successful or not all(k in successful[-1] for k in ('x','y')):
        raise RuntimeError('RETURN_ONLY_LAST_VISIT_UNAVAILABLE')
    current = node.tf_pose()
    last = successful[-1]
    deviation = math.hypot(float(last['x'])-current['x'],float(last['y'])-current['y'])
    if deviation > 1.5:
        raise RuntimeError(f'RETURN_ONLY_LAST_VISIT_POSE_MISMATCH {deviation:.2f}m')
    record['source_report']=str(source)
    record['mission_saved_dir']=str(session)
    record['mission_verification']={
        'success': eligibility['exploration']=='FULL',
        'return_eligible': True,
        'exploration_class':eligibility['exploration'],
        'detail':eligibility['detail'],
        'partial_zones':eligibility['partial_zones'],
        'state':summary.get('state'),'sent':summary.get('sent'),
        'source_report':str(source),'source_age_seconds':round(age,1),
        'last_visit_xy_error_m':round(deviation,3)}
    record['home_pose_map']=home
    record['status']='PRIOR_EXPLORATION_VERIFIED_FOR_RETURN'
    json_save(output/'report.json',record)
    print('[RETURN_ONLY_PRIOR_SESSION_VERIFIED] '+eligibility['detail'],flush=True)
    return home,summary


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path.home()/('Projects/'+PROJECT_NAME))
    p.add_argument('--mode',choices=['dry-run','execute'],default='dry-run')
    p.add_argument('--max-goals',type=int,default=100)
    p.add_argument('--max-plan-attempts',type=int,default=300)
    p.add_argument('--path-blockage-check',action='store_true')
    p.add_argument('--prior-topology-fixed',action='store_true')
    p.add_argument('--path-topic',default='/j100_0001/plan')
    p.add_argument('--mission-budget-guard',action='store_true',help='Opt-in operational stagnation/work budgets and frozen deferrals')
    p.add_argument('--guard-stall-window-s',type=float,default=20.)
    p.add_argument('--guard-stall-extent-m',type=float,default=.20)
    p.add_argument('--guard-stall-recoveries',type=int,default=5)
    p.add_argument('--guard-goal-recovery-limit',type=int,default=20,help='Independent per-Goal recovery ceiling; 0 disables ceiling, never implies blockage')
    p.add_argument('--guard-zone-cancel-limit',type=int,default=2)
    p.add_argument('--guard-zone-work-base-s',type=float,default=180.)
    p.add_argument('--guard-zone-work-max-s',type=float,default=300.)
    p.add_argument('--guard-extension-window-s',type=float,default=60.)
    p.add_argument('--guard-extension-gain-pp',type=float,default=1.)
    p.add_argument('--connectivity-aware',action='store_true')
    p.add_argument('--continue-on-nav-abort',action='store_true',help='Opt-in only; confirmed stop and fresh data still required')
    p.add_argument('--zone-hard-goals',type=int,default=20)
    p.add_argument('--audit-spacing-m',type=float,default=.60)
    p.add_argument('--audit-headings',type=int,default=8)
    p.add_argument('--audit-max-positions',type=int,default=1200)
    p.add_argument('--audit-max-probes',type=int,default=1000)
    p.add_argument('--audit-max-view-preflights',type=int,default=48)
    p.add_argument('--audit-build-budget-s',type=float,default=30.)
    p.add_argument('--audit-zone-budget-s',type=float,default=600.)
    p.add_argument('--audit-max-restarts',type=int,default=2)
    p.add_argument('--entry-access-max-probes',type=int,default=6)
    p.add_argument('--entry-access-per-component',type=int,default=2)
    p.add_argument('--entry-access-min-separation-m',type=float,default=.80)
    p.add_argument('--entry-access-failed-radius-m',type=float,default=.50)
    p.add_argument('--source-report',type=Path,default=None,
                   help='Return-only from an existing failed-to-return report; no exploration rerun')
    p.add_argument('--return-plan-only',action='store_true',
                   help='Explore in execute mode, then validate/plans home but DO NOT send return goal')
    p.add_argument('--return-timeout-seconds',type=int,default=900)
    a,ros=p.parse_known_args()
    if a.mission_budget_guard and not a.connectivity_aware:
        p.error('--mission-budget-guard requires --connectivity-aware')
    if not 1 <= a.audit_max_view_preflights <= 300:
        p.error('Invalid audit-max-view-preflights')
    if a.path_blockage_check and not a.connectivity_aware:
        p.error('--path-blockage-check requires --connectivity-aware')
    if ros and ros[0]!='--ros-args':p.error('Unexpected arguments '+repr(ros))
    if a.source_report is not None and a.mode!='execute':
        p.error('--source-report requires execute mode')
    if not 1<=a.max_goals<=1000 or not 1<=a.max_plan_attempts<=3000:
        p.error('goal and plan attempt limits out of range')
    if not 60<=a.return_timeout_seconds<=3600:
        p.error('return timeout must be 60..3600 sec')
    return a,ros


def main():
    global STOP_FLAG
    args,ros=cli()
    root=args.project.expanduser().resolve()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=root/'logs/dn011_return_home96'/('run_'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    record={'implementation':'dn011_c6_coverage96_return_home_v1','mode':args.mode,
            'created_utc':now_utc(),'threshold':0.96,'status':'STARTING',
            'return_plan_only':args.return_plan_only}
    json_save(out/'report.json',record)
    print('[RTH_LOG_DIR]',out,flush=True)
    signal.signal(signal.SIGINT,on_interrupt)
    signal.signal(signal.SIGTERM,on_interrupt)
    node=None
    try:
        rclpy.init(args=ros)
        node=HomeSupervisor()
        record['c6_params']=node.validate_c6()
        if args.source_report:
            home, summary = load_prior_mission_for_return(root,args.source_report,node,out,record)
        else:
            home=node.stable_home()
            record['home_pose_map']=home
            record['home_captured_utc']=now_utc()
            record['status']='HOME_LOCKED'
            json_save(out/'report.json',record)
            print('[HOME_LOCKED] map x={x:.3f} y={y:.3f} yaw={yaw:.3f}'.format(**home),flush=True)
            if STOP_FLAG:raise StopNow('USER_INTERRUPT')
            summary=mission_process(node,root,args,out,record)
        if args.mode=='execute':
            verification=record.get('mission_verification',{})
            exploration_class=verification.get('exploration_class','BLOCKED')
            partial=exploration_class!='FULL'
            limited=exploration_class=='LIMITED'
            record['exploration_class']=exploration_class
            record['coverage_target_met']=exploration_class=='FULL'
            record['normal_exploration_termination']=exploration_class in ('FULL','LIMITED')
            record['status']='EXPLORATION_PARTIAL_RETURN_PREFLIGHT' if partial else 'EXPLORATION_COMPLETE_96_VERIFIED'
            json_save(out/'report.json',record)
            if partial:
                print(('[MISSION_LIMITED] ' if limited else '[MISSION_PARTIAL] ')+verification.get('detail','')+'; return preflight follows',flush=True)
            else:
                print('[MISSION_COMPLETE] 27/27 zones >=96%; return preflight follows',flush=True)
            plan=node.plan_home(home)
            record['return_plan']=plan
            if plan['already_home']:
                record['status']='ALREADY_HOME_EXPLORATION_PARTIAL' if partial else 'ALREADY_HOME_VERIFIED'
                print('[HOME_ALREADY_REACHED] no extra goal needed',flush=True)
            elif args.return_plan_only:
                record['status']='RETURN_PLAN_ONLY_PASS_EXPLORATION_PARTIAL' if partial else 'RETURN_PLAN_ONLY_PASS'
                print('[RETURN_PLAN_ONLY_PASS] no NavigateToPose sent',flush=True)
            else:
                record['status']='RETURN_NAV_RUNNING'
                json_save(out/'report.json',record)
                print('[RETURN_NAV_REQUESTED] Nav2 GridBased -> initial map pose',flush=True)
                record['return_result']=node.nav_home(home,args.return_timeout_seconds)
                record['status']='RETURN_SUCCEEDED_EXPLORATION_PARTIAL_STOP_VERIFIED' if partial else 'RETURN_SUCCEEDED_STOP_VERIFIED'
                print('[RETURN_SUCCEEDED] pose and odometry stop verified; exploration='+exploration_class,flush=True)
        if record.get('exploration_class')=='LIMITED':
            record['status']=record['status'].replace('EXPLORATION_PARTIAL','EXPLORATION_LIMITED')
        json_save(out/'report.json',record)
        print('[RTH_REPORT]',out/'report.json',flush=True)
        return 0
    except (StopNow,RuntimeError,ValueError,OSError) as e:
        record['status']='USER_INTERRUPT' if isinstance(e,StopNow) else 'STOPPED_WITHOUT_RETURN'
        record['error']=str(e)
        record['finished_utc']=now_utc()
        json_save(out/'report.json',record)
        print('[RTH_STOP]',str(e),flush=True)
        print('[RTH_REPORT]',out/'report.json',flush=True)
        return 2
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':
    raise SystemExit(main())
