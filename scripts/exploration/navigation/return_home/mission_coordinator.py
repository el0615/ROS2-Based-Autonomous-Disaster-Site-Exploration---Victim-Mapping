#!/usr/bin/env python3
"""DN012 initial-home/checkpoint coordinator. Existing T6 owns exploration."""
from __future__ import annotations
import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/'scripts/exploration/zone_exploration/src'
sys.path[:0]=[str(SRC),str(ROOT/'scripts/exploration')]
from home_state import HomeCheckpoint, pose_map, lock_origin, transition, validate_navigation, utc
from checkpoint import load_checkpoint, save_checkpoint, atomic_bytes, encode
from home_anchor import check_completion
from home_return import verify_disk_identity, load_supervisor, ros_init, latest_session


@contextlib.contextmanager
def coordinator_lock():
    # One coordinator through exploration AND return. Never terminate another node.
    key=hashlib.sha256((str(ROOT)+'|'+os.environ.get('ROS_DOMAIN_ID','0')).encode()).hexdigest()[:20]
    with open('/tmp/dn012_coordinator_v189_'+key+'.lock','a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('ANOTHER_DN012_COORDINATOR_IS_RUNNING')
        yield


@contextlib.contextmanager
def t6_idle_lock():
    """Hold the existing T6 lock throughout return/manual-origin writes."""
    streams=[]
    try:
        for ns in ('/j100_0001','j100_0001'):
            key=hashlib.sha256((os.environ.get('ROS_DOMAIN_ID','0')+'|'+ns).encode()).hexdigest()[:16]
            stream=open('/tmp/hospital_observation_sequence_'+key+'.lock','a')
            streams.append(stream)
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('T6_STILL_RUNNING_NO_RETURN_OR_HOME_EDIT')
        yield
    finally:
        for stream in reversed(streams):stream.close()


def selected_session(value=None):
    path=Path(value).expanduser().resolve() if value else latest_session()
    base=(ROOT/'logs/hospital_zone_mission').resolve()
    if path.parent != base or path.is_symlink():raise ValueError('SESSION_OUTSIDE_PROJECT')
    return path


def state_from_session(session):
    payload,digest,path=load_checkpoint(session/'checkpoints_v18/LATEST.json')
    chain=payload.get('mission_session_ids') or []
    if not chain:raise ValueError('ORIGINAL_MISSION_LINEAGE_MISSING')
    home=HomeCheckpoint(ROOT,payload['context'],chain,payload)
    return payload,digest,home.data


def append_checkpoint(session, navigation, expected_digest):
    """Append a complete new snapshot; retain ALL old snapshots; never edit old JSON."""
    payload,digest,_=load_checkpoint(session/'checkpoints_v18/LATEST.json')
    if digest!=expected_digest:raise ValueError('CHECKPOINT_CHANGED_CONCURRENTLY')
    ctx=payload['context'];mission_id=payload['mission_session_ids'][0]
    navigation=validate_navigation(navigation,mission_id,ctx)
    old=(payload.get('navigation_v189') or {}).get('original_start_pose')
    if old is not None and navigation['original_start_pose']!=old:
        raise ValueError('ORIGINAL_START_OVERWRITE_REFUSED')
    updated=copy.deepcopy(payload)
    updated['navigation_v189']=navigation
    updated['parent_payload_sha256']=digest
    updated['created_utc']=utc()
    updated['checkpoint_update_v189']='HOME_STATE_ONLY; mission/Coverage/blockage evidence unchanged'
    cp,new_digest=save_checkpoint(session/'checkpoints_v18',updated,retain=1_000_000)
    return new_digest,str(cp)


def report(session):
    from mission_report import export_report
    result=export_report(ROOT,session)
    print('[BLOCKED_MAP_REPORT]',result,flush=True)
    return result


def perform_return(session, *, plan_only=False):
    with t6_idle_lock():
        return _perform_return(session, plan_only=plan_only)


def _perform_return(session, *, plan_only=False):
    verified=check_completion(session)
    verify_disk_identity(verified)
    payload,digest,nav=state_from_session(session)
    home=nav.get('original_start_pose')
    rp=ROOT/'logs/dn012_return_home_v189'/('return_'+utc().replace(':','').replace('+','_'))/'report.json'
    record=dict(version='DN012_V18_9',session=session.name,plan_only=plan_only,
                original_start_pose=copy.deepcopy(home),created_utc=utc())
    ctx=payload['context'];rclpy=None;n=None
    def save(status,detail=None,new_attempt=False):
        nonlocal nav,digest
        if not plan_only:
            if n is not None:
                current=n.tf_pose()
                if current is not None:
                    nav['checkpoint_robot_pose']=dict(pose_map(current),frame_id='map',wall_utc=utc())
            nav=transition(nav,ctx,status,detail=detail,new_attempt=new_attempt)
            digest,cp=append_checkpoint(session,nav,digest)
            record['checkpoint']=cp
        record.update(state=status,detail=detail,wall_utc=utc())
        atomic_bytes(rp,encode(record))
    old_handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        if home is None:
            save('ORIGINAL_START_MISSING',dict(navigation_sent=False))
            print('[ORIGINAL_START_MISSING] Legacy mission has no first-start map pose. '
                  'Current/Resume/spawn pose will NOT be substituted; no return Goal.',flush=True)
            return 2
        save('RETURN_PREFLIGHT',new_attempt=not plan_only)
        rclpy,legacy=load_supervisor()
        legacy.STOP_FLAG=False
        signal.signal(signal.SIGINT,legacy.on_interrupt)
        signal.signal(signal.SIGTERM,legacy.on_interrupt)
        ros_init(rclpy)
        n=legacy.HomeSupervisor()
        print('[RETURN_HOME_TARGET] original mission='+nav['mission_id']+' map='+str([home[k] for k in ('x','y','yaw')]),flush=True)
        plan=n.plan_home(pose_map(home))
        record['plan']=plan
        if plan_only:
            record['state']='PLAN_ONLY_NO_NAVIGATION';atomic_bytes(rp,encode(record))
            print('[RETURN_PLAN_ONLY_PASS] no navigation; checkpoint unchanged',flush=True)
            return 0
        save('RETURN_PLAN_OK',plan)
        if plan['already_home']:
            save('HOME_REACHED',dict(already_home=True))
            print('[HOME_REACHED] already at original pose; no Goal',flush=True)
            return 0
        save('RETURN_IN_PROGRESS',dict(goal_pose_source='original_start_pose'))
        result=n.nav_home(pose_map(home),900.)
        save('HOME_REACHED',result)
        print('[HOME_REACHED] original pose and stopped odometry verified',flush=True)
        return 0
    except BaseException as exc:
        # Existing HomeSupervisor cancels only its own Goal and waits for stop.
        try:save('RETURN_FAILED_OR_INTERRUPTED',dict(error=type(exc).__name__+':'+str(exc)))
        except Exception as persist_error:
            print('[RETURN_CHECKPOINT_WRITE_FAILED]',repr(persist_error),flush=True)
        print('[RETURN_HOME_STOPPED]',repr(exc),'; verify physical stop independently',flush=True)
        if isinstance(exc,KeyboardInterrupt):return 130
        return 2
    finally:
        if n is not None:n.destroy_node()
        if rclpy is not None and rclpy.ok():rclpy.shutdown()
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        print('[RETURN_REPORT]',rp,flush=True)


def run_child(command, dry=False):
    env=os.environ.copy();env.update(DN012_V189_CHILD='1',PYTHONNOUSERSITE='1',ZB_PROJECT_ROOT=str(ROOT))
    saved=[];stop=[False]
    p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    old_handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def cancel(sig,frame):
        stop[0]=True
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGINT)
            except ProcessLookupError:pass
        print('[DN012_INTERRUPT] stop requested; no home Goal will follow',flush=True)
    try:
        for sig in old_handlers:signal.signal(sig,cancel)
        for line in p.stdout:
            print(line,end='',flush=True)
            if line.startswith('[SAVED] '):saved.append(line[8:].strip())
        code=p.wait()
    finally:
        if p.poll() is None:
            cancel(signal.SIGINT,None)
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print('[T6_STOP_UNCONFIRMED] inspect Nav2/robot; no automatic return',flush=True)
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
    if stop[0]:return 130
    if code!=0:return code
    if dry:
        print('[RETURN_HOME_SKIPPED] dry-run; no home capture or return navigation',flush=True)
        return 0
    if len(saved)!=1:raise ValueError('NO_UNAMBIGUOUS_EXECUTE_SESSION')
    session=selected_session(saved[0])
    # Only the existing all-COMPLETE lifecycle authorizes return; never reopen Zones.
    try:check_completion(session)
    except ValueError as exc:
        print('[RETURN_HOME_SKIPPED] exploration not all COMPLETE:',str(exc),flush=True)
        return 0
    result=perform_return(session)
    try:report(session)
    except Exception as exc:print('[REPORT_FAILED]',repr(exc),'; raw evidence retained',flush=True)
    return result


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    action=argv.pop(0) if argv else 'help'
    if action in ('help','-h','--help'):
        print('mission.sh start [--dry-run] | resume [--dry-run] | return-home [--plan-only] | report | home-status\n'
              'Legacy: home-set --x X --y Y --yaw RAD --confirm-original-map-pose [--session PATH]')
        return 0
    if action=='report':
        p=argparse.ArgumentParser();p.add_argument('--session');a=p.parse_args(argv)
        report(selected_session(a.session));return 0
    with coordinator_lock():
        if action=='resume':
            return run_child(['/usr/bin/python3',str(ROOT/'scripts/exploration/navigation/mission/current_pose_resume.py')]+argv,'--dry-run' in argv)
        if action in ('start','mission'):
            dry='--dry-run' in argv or (action=='mission' and argv and argv[0]=='dry-run')
            if action=='mission':
                if not argv or argv[0] not in ('execute','dry-run'):raise ValueError('MISSION_MODE_REQUIRED')
                mode=argv.pop(0)
            else:
                mode='dry-run' if dry else 'execute'
                if '--dry-run' in argv:argv.remove('--dry-run')
            if '--resume-from' in argv and action=='start':raise ValueError('USE_RESUME_FOR_CONTINUATION')
            return run_child(['bash',str(ROOT/'scripts/exploration/navigation/run_exploration.sh'),'mission',mode]+argv,dry)
        p=argparse.ArgumentParser();p.add_argument('--session')
        if action=='return-home':p.add_argument('--plan-only',action='store_true')
        elif action=='home-capture':
            p.add_argument('--confirm-at-original-start',action='store_true')
        elif action=='home-set':
            for key in ('x','y','yaw'):p.add_argument('--'+key,type=float,required=True)
            p.add_argument('--confirm-original-map-pose',action='store_true')
        elif action not in ('home-status',):raise ValueError('UNKNOWN_ACTION:'+action)
        a=p.parse_args(argv);session=selected_session(a.session)
        if action=='return-home':return perform_return(session,plan_only=a.plan_only)
        payload,digest,nav=state_from_session(session)
        if action=='home-status':
            print(json.dumps(nav,ensure_ascii=False,indent=2));return 0
        if action=='home-capture':
            if not a.confirm_at_original_start:raise ValueError('CONFIRM_AT_ORIGINAL_START_REQUIRED')
            if nav['original_start_pose'] is not None:
                print('[ORIGINAL_START_ALREADY_LOCKED]',nav['original_start_pose'],flush=True)
                return 0
            with t6_idle_lock():
                rc,legacy=load_supervisor();ros_init(rc);node=None
                try:
                    node=legacy.HomeSupervisor()
                    node.wait_for(lambda:node.goal_status_seen and not node.active_nav_ids,12,'NO_ACTIVE_NAV_GOALS')
                    pose=node.stable_home()
                    nav=lock_origin(nav,payload['context'],pose,'ORIGINAL_START_LIVE_TF_OPERATOR_CONFIRMED')
                    _,cp=append_checkpoint(session,nav,digest)
                    print('[ORIGINAL_START_LOCKED]',nav['original_start_pose'],'\n[CHECKPOINT]',cp,flush=True)
                finally:
                    if node is not None:node.destroy_node()
                    if rc.ok():rc.shutdown()
            return 0
        if not a.confirm_original_map_pose:raise ValueError('CONFIRM_ORIGINAL_MAP_POSE_REQUIRED')
        # Explicit legacy repair; never captures current pose implicitly.
        nav=lock_origin(nav,payload['context'],dict(x=a.x,y=a.y,yaw=a.yaw),
                        'ORIGINAL_START_MAP_POSE_OPERATOR_CONFIRMED')
        with t6_idle_lock():
            _,cp=append_checkpoint(session,nav,digest)
        print('[ORIGINAL_START_LOCKED]',nav['original_start_pose'],'\n[CHECKPOINT]',cp,flush=True)
        return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,KeyError,ImportError) as exc:
        print('[DN012_V189_REFUSED]',repr(exc),file=sys.stderr);raise SystemExit(2)
