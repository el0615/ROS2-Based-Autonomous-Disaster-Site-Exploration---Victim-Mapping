#!/usr/bin/env python3
"""Supervised original-home capture, return and resume handoff.

Never selects a home from a resume current pose. No goal without a verified
immutable anchor and successful 27/27 Zone lifecycle checkpoint.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, pathlib, re, signal, subprocess, sys, time
from home_anchor import check_completion, make_anchor, validate_anchor

ROOT = pathlib.Path(__file__).resolve().parents[4]
LOG = ROOT/'logs/dn012_return_home'
ANCHOR = LOG/'original_start_home_v188.json'
DN011 = ROOT/'scripts/exploration/navigation/return_home/run_mission_with_home.py'
DN011_POLICY = ROOT/'scripts/exploration/navigation/return_home/return_home_policy.py'
DN011_HASHES = {'run_mission_with_home.py':'f5a75da6b20b35cec0f13ae4b9c73322758814fe6d45d62abaeb97c7531b26b0',
                'return_home_policy.py':'1beb6a0ce8262c4981ee847860386c77c04aa4deee59890c04a8f67b29d07f24'}
STOP = False

def interrupted(sig,frame):
    global STOP
    STOP = True
    print('[RETURN_HOME_INTERRUPT] cancellation requested',flush=True)

def atomic_json(path,record,immutable=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(record,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode('utf-8')
    if immutable:
        # Exclusive new file: no silent overwrite of the first mission's home.
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
        try:
            with os.fdopen(fd,'wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
        finally: pass
    else:
        tmp=path.with_name('.'+path.name+'.tmp_'+str(os.getpid()))
        try:
            with open(tmp,'xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if tmp.exists():tmp.unlink()

def latest_session():
    base=ROOT/'logs/hospital_zone_mission'
    dirs=sorted((p for p in base.glob('session_*') if p.is_dir()),reverse=True)
    for p in dirs:
        if not re.fullmatch(r'session_\d{8}_\d{6}_\d{6}',p.name):continue
        manifest=p/'run_manifest.json'
        if not manifest.exists():continue
        o=json.loads(manifest.read_text())
        if o.get('options',{}).get('execute') is True:
            return p
    raise ValueError('NO_EXECUTE_SESSION')

def session_info(session):
    if session is None:session=latest_session()
    p=pathlib.Path(session).resolve(strict=True)
    base=(ROOT/'logs/hospital_zone_mission').resolve()
    if p.parent != base:raise ValueError('SESSION_OUTSIDE_PROJECT')
    return check_completion(p)

def verify_dn011_code():
    for p in (DN011,DN011_POLICY):
        if not p.is_file() or p.is_symlink():raise ValueError('RETURN_HOME_MODULE_MISSING_OR_SYMLINK:'+str(p))
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=DN011_HASHES[p.name]:raise ValueError('RETURN_HOME_MODULE_CHANGED:'+p.name)

def load_supervisor():
    verify_dn011_code()
    p=str(DN011.parent)
    if p not in sys.path:sys.path.insert(0,p)
    import rclpy
    import run_mission_with_home as legacy
    return rclpy,legacy

def ros_init(rclpy):
    # Preserve J100 namespaced TF remaps, exactly as the existing DN011 launcher.
    rclpy.init(args=['--ros-args','-p','use_sim_time:=true',
        '-r','/tf:=/j100_0001/tf','-r','/tf_static:=/j100_0001/tf_static'])

def save_anchor(pose,verified,source):
    anchor=make_anchor(verified,pose,source)
    atomic_json(ANCHOR,anchor,immutable=True)
    print('[ORIGINAL_HOME_LOCKED]',ANCHOR, 'map:',
          round(anchor['pose_map']['x'],3),round(anchor['pose_map']['y'],3),round(anchor['pose_map']['yaw'],3),flush=True)

def capture_home(verified):
    rclpy,legacy=load_supervisor()
    ros_init(rclpy)
    n=None
    try:
        n=legacy.HomeSupervisor()
        home=n.stable_home()  # 5 stable map TF samples, fresh costmap, stopped odometry
        save_anchor(home,verified,'ORIGINAL_START_LIVE_TF_OPERATOR_CONFIRMED')
    finally:
        if n is not None:n.destroy_node()
        if rclpy.ok():rclpy.shutdown()

def verify_disk_identity(verified):
    # The prior return supervisor validates live TF / occupancy / costmap / goals.
    # Disk snapshot identity prevents a different world/Nav2 configuration from
    # silently reusing a previous mission's origin.
    files={
        'world': pathlib.Path(os.environ.get('DN012_WORLD_FILE') or str(ROOT/'gazebo/worlds/disaster/disaster_hospital.sdf')),
        'zones':ROOT/'config/exploration/navigation/hospital_zones.json',
        'reference':ROOT/'config/exploration/navigation/hospital_coverage_reference.json.gz',
        'nav2':ROOT/'config/nav2/nav2_hospital_navigation.yaml',
    }
    for key,path in files.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError('RETURN_CONTEXT_FILE_MISSING:'+key)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=verified['context']['fingerprints'][key]:
            raise ValueError('RETURN_CONTEXT_CHANGED:'+key)

def report_path():
    return LOG/('return_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))/'report.json'

def perform_return(verified,*,plan_only=False,timeout=900):
    rp=report_path()
    record=dict(implementation='DN012_RETURN_TO_ORIGINAL_START_V1_8_8',
                exploration_session=verified['session'],
                parent_checkpoint_sha256=verified['checkpoint_sha256'],
                original_session=verified['initial_session'],
                state='RETURN_PREFLIGHT',plan_only=plan_only,
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic_json(rp,record)
    print('[RETURN_HOME_REPORT]',rp,flush=True)
    rclpy=None;legacy=None;n=None
    try:
        if not ANCHOR.is_file() or ANCHOR.is_symlink():
            raise ValueError('ORIGINAL_HOME_NOT_LOCKED: use home-capture AT INITIAL START or home-set with original MAP coordinates')
        home=validate_anchor(json.loads(ANCHOR.read_text()),verified)
        verify_disk_identity(verified)
        record['home']=home
        rclpy,legacy=load_supervisor()
        signal.signal(signal.SIGINT,legacy.on_interrupt)
        signal.signal(signal.SIGTERM,legacy.on_interrupt)
        ros_init(rclpy)
        n=legacy.HomeSupervisor()
        # The old DN011 96%-all policy is deliberately NOT used. v1.8.7's
        # 27 explicit terminal lifecycle certificates were verified above.
        plan=n.plan_home(home)
        record['plan']=plan
        record['state']='RETURN_PLAN_OK'
        atomic_json(rp,record)
        if plan['already_home']:
            record['state']='HOME_REACHED'
            print('[HOME_ALREADY_REACHED] No Goal sent',flush=True)
        elif plan_only:
            record['state']='PLAN_ONLY_NO_NAVIGATION'
            print('[RETURN_PLAN_ONLY_PASS] No Goal sent',flush=True)
        else:
            record['state']='RETURN_NAV_RUNNING'
            atomic_json(rp,record)
            print('[RETURN_NAV_REQUESTED] initial mission home (not resume start)',flush=True)
            record['nav_result']=n.nav_home(home,timeout)
            record['state']='HOME_REACHED'
            print('[HOME_REACHED] pose and stopped odometry confirmed',flush=True)
        atomic_json(rp,record)
    except BaseException as e:
        record['state']='RETURN_HOME_FAILED_OR_INTERRUPTED'
        record['error']=type(e).__name__+':'+str(e)
        atomic_json(rp,record)
        print('[RETURN_HOME_FAILED]',record['error'],'independent stop verification required',flush=True)
        raise
    finally:
        if n is not None:n.destroy_node()
        if rclpy is not None and rclpy.ok():rclpy.shutdown()

def resume(args):
    # Resume selection / Nav2 exploration remains the UNMODIFIED original code.
    cmd=['/usr/bin/python3',str(ROOT/'scripts/exploration/navigation/mission/current_pose_resume.py')]+args
    env=os.environ.copy();env['PYTHONNOUSERSITE']='1';env['ZB_PROJECT_ROOT']=str(ROOT)
    saved=[]
    dry='--dry-run' in args
    print('[DN012_RETURN_HOME_ARMED] Original home required after verified full mission',flush=True)
    with subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True,bufsize=1) as proc:
        try:
            for line in proc.stdout:
                print(line,end='',flush=True)
                if line.startswith('[SAVED] '):saved.append(line[8:].strip())
            status=proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print('[T6_STOP_UNCONFIRMED] check robot and Nav2 Goal manually; no home Goal',flush=True)
            raise
    if status!=0:
        print('[RETURN_HOME_SKIPPED] T6 did not exit normally; no return Goal',flush=True)
        return status
    if dry:
        print('[RETURN_HOME_SKIPPED] dry-run: no return Goal',flush=True)
        return 0
    if len(saved)!=1:
        raise ValueError('NO_UNAMBIGUOUS_T6_SAVED_SESSION: no return Goal')
    verified=session_info(saved[0])
    print('[RETURN_HOME_ELIGIBLE] 27/27 explicitly COMPLETE; initial session=',
          verified['initial_session'],flush=True)
    perform_return(verified)
    return 0

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='resume':return resume(argv[1:])
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('resume');p.add_argument('extra',nargs=argparse.REMAINDER)
    p=sub.add_parser('return-home');p.add_argument('--session',type=pathlib.Path);p.add_argument('--plan-only',action='store_true')
    p=sub.add_parser('home-capture');p.add_argument('--session',type=pathlib.Path);p.add_argument('--confirm-at-original-start',action='store_true')
    p=sub.add_parser('home-set');p.add_argument('--session',type=pathlib.Path);
    p.add_argument('--x',type=float,required=True);p.add_argument('--y',type=float,required=True);p.add_argument('--yaw',type=float,required=True)
    p.add_argument('--confirm-original-map-pose',action='store_true')
    a=parser.parse_args(argv)
    if a.action=='resume':return resume(a.extra)
    verified=session_info(a.session)
    if a.action=='home-capture':
        if not a.confirm_at_original_start:raise ValueError('CONFIRM_AT_ORIGINAL_START_REQUIRED')
        capture_home(verified);return 0
    if a.action=='home-set':
        if not a.confirm_original_map_pose:raise ValueError('CONFIRM_ORIGINAL_MAP_POSE_REQUIRED')
        save_anchor(dict(x=a.x,y=a.y,yaw=a.yaw),verified,
                    'ORIGINAL_START_MAP_POSE_OPERATOR_CONFIRMED');return 0
    if a.action=='return-home':
        perform_return(verified,plan_only=a.plan_only);return 0
    raise ValueError('UNKNOWN_ACTION')

if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,RuntimeError,ValueError,KeyError,ImportError) as e:
        print('[DN012_HOME_REFUSED]',str(e),file=sys.stderr,flush=True)
        raise SystemExit(2)
