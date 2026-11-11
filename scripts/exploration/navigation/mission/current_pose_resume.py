#!/usr/bin/env python3
"""Single-command T6 Resume for the STILL-RUNNING T5/Gazebo session.

No automatic cancellation, velocity output, teleport, T5 reset or budget reset.
The existing T6 performs fresh pose/footprint/odom/plan gates before motion.
ROS API basis: ros2/rclpy humble Node, qos; ros2/rcl_interfaces GoalStatus.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'scripts/exploration'),str(ROOT/'scripts/exploration/zone_exploration/src')]
from checkpoint import encode, atomic_bytes, project_context
from resume_selection import VERSION, LiveReadiness, select_checkpoint, recover_terminal_history
from zone_lifecycle import completion_facts


def ensure_t6_stopped(namespace):
    """Probe the same flock held by zone_node.main; never kill the holder."""
    for ns in {namespace,namespace.lstrip('/')}:
        key=hashlib.sha256((os.environ.get('ROS_DOMAIN_ID','0')+'|'+ns).encode()).hexdigest()[:16]
        with open('/tmp/hospital_observation_sequence_'+key+'.lock','a') as lock:
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('T6_STILL_RUNNING: close the old T6 before resume')
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


def live_status(namespace, timeout):
    """Passive ROS subscriptions only; no goals/services/twists are sent."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from std_msgs.msg import String
    from rosgraph_msgs.msg import Clock
    from action_msgs.msg import GoalStatusArray
    state=LiveReadiness(time.monotonic())
    topic=namespace+'/navigate_to_pose/_action/status'
    rclpy.init(args=[])
    node=Node('dn012_resume_readonly_preflight',start_parameter_services=False)
    reliable=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL)
    def coverage(msg):
        try:state.on_coverage(json.loads(msg.data),time.monotonic())
        except (ValueError,TypeError):pass
    handles=[node.create_subscription(String,'/hospital_coverage/status',coverage,reliable),
             node.create_subscription(Clock,'/clock',lambda m:state.on_clock(
                 float(m.clock.sec)+m.clock.nanosec*1e-9,time.monotonic()),qos_profile_sensor_data),
             node.create_subscription(GoalStatusArray,topic,lambda m:state.on_actions(
                 [int(s.status) for s in m.status_list],time.monotonic()),reliable)]
    started=time.monotonic();last_print=None
    try:
        while rclpy.ok() and time.monotonic()-started<timeout:
            rclpy.spin_once(node,timeout_sec=.1)
            now=time.monotonic()
            ready,reason=state.check(now,node.count_publishers('/hospital_coverage/status'),node.count_publishers(topic))
            if reason!=last_print:
                print('[RESUME_PREFLIGHT]',reason,flush=True);last_print=reason
            if ready:
                return dict(state.status)
        raise ValueError('RESUME_PREFLIGHT_TIMEOUT:'+str(last_print)+
                         '; T5/Gazebo/Nav2 left untouched; no navigation sent')
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true',help='Same auto-selection; no new NavigateToPose')
    p.add_argument('--namespace',default='/j100_0001')
    p.add_argument('--wait-seconds',type=float,default=20.)
    p.add_argument('--manual-note',help='Optional declared manual intervention; no location-based inference')
    a=p.parse_args()
    if not math.isfinite(a.wait_seconds) or not 3<=a.wait_seconds<=60:
        p.error('--wait-seconds must be in [3,60]')
    ns='/'+a.namespace.strip('/')
    if ns!='/j100_0001':p.error('This reviewed project launcher is for /j100_0001')
    if a.manual_note and len(a.manual_note)>1000:p.error('manual note too long')
    ensure_t6_stopped(ns)
    key=hashlib.sha256((str(ROOT)+'|'+os.environ.get('ROS_DOMAIN_ID','0')+'|'+ns).encode()).hexdigest()[:16]
    launch_lock=open('/tmp/dn012_autoresume_'+key+'.lock','a')
    try:fcntl.flock(launch_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('ANOTHER_AUTO_RESUME_IS_RUNNING')
    # Keep the launcher's lock across exec; the existing T6 also takes its own lock.
    os.set_inheritable(launch_lock.fileno(),True)
    print('[CURRENT_POSE_RESUME] T6 only; T5 remains running. No previous pose/Goal replay.',flush=True)
    print('[SCENE_CONDITION] resume assumes unchanged live obstacle/world layout; '
          'file fingerprints and live T5/clock are checked, GUI scene edits cannot be inferred.',flush=True)
    from hospital_coverage_core import load_bundle
    bundle=load_bundle(ROOT/'config/exploration/navigation/hospital_zones.json',
                       ROOT/'config/exploration/navigation/hospital_coverage_reference.json.gz')
    context=project_context(ROOT,bundle,ns)
    live=live_status(ns,a.wait_seconds)
    selected=select_checkpoint(ROOT,bundle,context,live)
    effective=recover_terminal_history(ROOT,selected['payload'],bundle,context)
    facts=completion_facts(effective)
    print('[AUTO_CHECKPOINT]',selected['path'],flush=True)
    print('[RESTORE_HISTORY] goals=',effective['counters']['sent_goals'],
          'complete=',len(facts),'remaining=',sorted(set(effective['scope'])-set(facts)),flush=True)
    print('[LIVE_T5]',live['session_id'],'observed_cells=',live['observed_cells'],flush=True)
    for item in selected['ignored'][:8]:
        print('[NOT_SELECTED]',item['session'],item['reason'],flush=True)
    # Never hand a mutable LATEST.json pointer to the child. Exact immutable file.
    directory=ROOT/'logs/dn012_current_pose_resume'/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    path=directory/'selection.json'
    note=dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
              session=selected['session'],checkpoint=selected['path'],payload_sha256=selected['payload_sha256'],
              context=context,live_coverage_session=live['session_id'],
              source_counters=effective['counters'],finished_zones=sorted(facts),
              remaining_zones=sorted(set(effective['scope'])-set(facts)),
              terminal_history_recovery=effective.get('completion_v183'),
              mode='DRY_RUN' if a.dry_run else 'EXECUTE',manual_note=a.manual_note,
              manual_intervention='DECLARED' if a.manual_note else 'NOT_DECLARED',
              selection_ignored=selected['ignored'],
              live_scene_assumption='operator requests continuation of unchanged scene',
              t5_modified=False,prior_goal_replayed=False)
    atomic_bytes(path,encode(note))
    ensure_t6_stopped(ns)
    env=os.environ.copy()
    env['DN012_RESUME_SELECTION_V183']=str(path)
    cmd=['bash',str(ROOT/'scripts/exploration/navigation/run_exploration.sh'),
         'mission', 'dry-run' if a.dry_run else 'execute',
         '--resume-from',selected['path'],'--resume-confirm-scene']
    os.chdir(ROOT)
    print('[RESUME_HANDOFF] current TF/odom/scan/costmaps + stopped footprint checked by T6 before new Goal.',flush=True)
    sys.stdout.flush()
    os.execvpe('bash',cmd,env)


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('[RESUME_CANCELED] Read-only preflight stopped; no new Goal sent.',file=sys.stderr)
        raise SystemExit(130)
    except (OSError,ValueError,RuntimeError,ImportError,KeyError) as exc:
        print('[RESUME_REFUSED]',str(exc),file=sys.stderr)
        raise SystemExit(2)
