"""Immutable first-mission map pose and resumable return state (ROS-free)."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from datetime import datetime, timezone

SCHEMA = 'dn012_navigation_v1_8_9'
SOURCES = {'FIRST_MISSION_FRESH_MAP_TF', 'ORIGINAL_START_MAP_POSE_OPERATOR_CONFIRMED',
           'ORIGINAL_START_LIVE_TF_OPERATOR_CONFIRMED'}
STATES = {'NOT_STARTED', 'RETURN_PREFLIGHT', 'RETURN_PLAN_OK', 'RETURN_IN_PROGRESS',
          'HOME_REACHED', 'RETURN_FAILED_OR_INTERRUPTED', 'ORIGINAL_START_MISSING'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pose_map(pose):
    if isinstance(pose, (list, tuple)):
        if len(pose) != 3:
            raise ValueError('HOME_POSE_LENGTH')
        pose = dict(zip(('x','y','yaw'),pose))
    if not isinstance(pose, dict):
        raise ValueError('HOME_POSE_OBJECT')
    for key in ('x','y','yaw'):
        v = pose.get(key)
        if type(v) not in (int,float) or not math.isfinite(v):
            raise ValueError('HOME_POSE_NONFINITE:'+key)
    if abs(pose['x']) > 10000 or abs(pose['y']) > 10000 or abs(pose['yaw']) > math.pi+1e-8:
        raise ValueError('HOME_POSE_OUT_OF_RANGE')
    return dict(x=float(pose['x']),y=float(pose['y']),yaw=float(pose['yaw']),
                qx=0.,qy=0.,qz=math.sin(pose['yaw']/2),qw=math.cos(pose['yaw']/2))


def initial_navigation(mission_id, context):
    return dict(schema=SCHEMA, mission_id=mission_id, context_sha256=fingerprint(context),
                original_start_pose=None, checkpoint_robot_pose=None, resume_start_pose=None,
                return_home=dict(status='NOT_STARTED', attempts=0,
                    goal_pose_source='original_start_pose', history=[]))


def validate_navigation(data, mission_id, context):
    if not isinstance(data, dict) or data.get('schema') != SCHEMA:
        raise ValueError('HOME_CHECKPOINT_SCHEMA')
    if data.get('mission_id') != mission_id or data.get('context_sha256') != fingerprint(context):
        raise ValueError('HOME_CHECKPOINT_MISSION_OR_MAP_MISMATCH')
    origin = data.get('original_start_pose')
    if origin is not None:
        if not isinstance(origin,dict) or origin.get('frame_id') != 'map':
            raise ValueError('HOME_FRAME_MUST_BE_MAP')
        if origin.get('source') not in SOURCES or origin.get('source_session') != mission_id:
            raise ValueError('HOME_ORIGINAL_PROVENANCE_INVALID')
        if origin.get('context_sha256') != fingerprint(context):
            raise ValueError('HOME_ORIGINAL_MAP_MISMATCH')
        pose_map(origin)
        content = {k:v for k,v in origin.items() if k!='sha256'}
        if origin.get('sha256') != fingerprint(content):
            raise ValueError('HOME_ORIGIN_DIGEST_MISMATCH')
    r = data.get('return_home') or {}
    if r.get('status') not in STATES or r.get('goal_pose_source') != 'original_start_pose':
        raise ValueError('HOME_RETURN_STATE_INVALID')
    if type(r.get('attempts')) is not int or r['attempts'] < 0 or not isinstance(r.get('history'),list):
        raise ValueError('HOME_RETURN_HISTORY_INVALID')
    if origin is None and r['status'] in {'RETURN_PLAN_OK','RETURN_IN_PROGRESS','HOME_REACHED'}:
        raise ValueError('HOME_RETURN_WITHOUT_ORIGIN')
    return copy.deepcopy(data)


def lock_origin(data, context, pose, source, ros_time_s=None):
    validate_navigation(data, data['mission_id'], context)
    if source not in SOURCES:
        raise ValueError('HOME_SOURCE_INVALID')
    p = pose_map(pose)
    if data['original_start_pose'] is not None:
        current = data['original_start_pose']
        if any(abs(current[k]-p[k]) > 1e-10 for k in ('x','y','yaw')):
            raise ValueError('ORIGINAL_START_IS_IMMUTABLE')
        return copy.deepcopy(data)
    if ros_time_s is not None and (type(ros_time_s) not in (int,float) or not math.isfinite(ros_time_s)):
        raise ValueError('HOME_STAMP_INVALID')
    origin=dict(p, frame_id='map', source=source, source_session=data['mission_id'],
                context_sha256=fingerprint(context), captured_utc=utc(), ros_time_s=ros_time_s)
    origin['sha256']=fingerprint(origin)
    result=copy.deepcopy(data);result['original_start_pose']=origin
    if result['return_home']['status']=='ORIGINAL_START_MISSING':
        result['return_home']['status']='NOT_STARTED'
    return result


def transition(data, context, status, *, detail=None, new_attempt=False):
    result=validate_navigation(data,data['mission_id'],context)
    if status not in STATES:raise ValueError('HOME_RETURN_STATE_INVALID')
    if status in {'RETURN_PLAN_OK','RETURN_IN_PROGRESS','HOME_REACHED'} and not result['original_start_pose']:
        raise ValueError('ORIGINAL_START_MISSING')
    r=result['return_home'];r['status']=status
    if new_attempt:r['attempts']+=1
    r['history'].append(dict(status=status,wall_utc=utc(),detail=copy.deepcopy(detail)))
    return result


class InitialPoseSampler:
    """Five fresh TF samples while stopped, following the existing home policy."""
    def __init__(self):
        self.samples=[]

    def feed(self, pose, stamp, wall, *, ready):
        if not ready:
            self.samples=[];return None
        p=pose_map(pose)
        if not math.isfinite(stamp) or stamp <= 0:return None
        if self.samples and (stamp <= self.samples[-1][0] or wall-self.samples[-1][1] < .3):
            return None
        if self.samples:
            base=self.samples[0][2]
            da=abs(math.atan2(math.sin(p['yaw']-base['yaw']),math.cos(p['yaw']-base['yaw'])))
            if math.hypot(p['x']-base['x'],p['y']-base['y'])>.08 or da>.12:
                self.samples=[]
        self.samples.append((stamp,wall,p))
        if len(self.samples)<5:return None
        return copy.deepcopy(self.samples[-1][2])


class HomeCheckpoint:
    def __init__(self, root, context, session_ids, parent=None):
        from pathlib import Path
        self.root=Path(root);self.context=context;self.mission_id=session_ids[0]
        self.parent=parent;self.sampler=InitialPoseSampler();self.capture_persisted=False
        self.data=initial_navigation(self.mission_id,context)
        if parent and parent.get('navigation_v189') is not None:
            self.data=validate_navigation(parent['navigation_v189'],self.mission_id,context)
        elif parent:
            # Legacy v188 explicit anchors are acceptable ONLY for the exact lineage.
            anchor=self.root/'logs/dn012_return_home/original_start_home_v188.json'
            if anchor.is_file() and not anchor.is_symlink():
                a=json.loads(anchor.read_text())
                expected=dict(initial_session=self.mission_id,
                    t5_session=parent['coverage']['source_session'],
                    bundle_signature=context['bundle_signature'],
                    fingerprints={k:context['fingerprints'][k] for k in ('world','reference','zones','nav2')})
                if a.get('identity')==expected:
                    if (a.get('frame_id')!='map' or a.get('not_resume_start') is not True
                            or a.get('source') not in SOURCES):
                        raise ValueError('LEGACY_HOME_ANCHOR_INVALID')
                    self.data=lock_origin(self.data,context,a['pose_map'],a['source'])
            if self.data['original_start_pose'] is None:
                self.data['return_home']['status']='ORIGINAL_START_MISSING'

    def before_initial_poll(self, controller):
        """Called before any first-episode planning / navigation, not during Resume."""
        n=controller.n
        if self.parent is not None or not controller.args.execute or n.stop_requested:
            return True
        if self.data['original_start_pose'] is not None:
            if not self.capture_persisted:
                controller.dirty=True;n.publish_report(force=True)
                if controller.write_failed or controller.last_saved is None:
                    raise ValueError('ORIGINAL_START_CHECKPOINT_WRITE_FAILED')
                self.capture_persisted=True
            return True
        if n.sent_goals != 0:
            raise ValueError('CANNOT_CAPTURE_HOME_AFTER_FIRST_NAV')
        # Reuse the original exploration input path. n.inputs() already checks
        # current map->robot_frame TF age and the live coverage/costmap/scan/odom
        # inputs used by Nav2 exploration. Do not add a second HOME-only
        # Action/stop/footprint gate, five-sample TF gate or 60-second abort.
        # The existing planner/navigation checks remain in the mission node.
        now,wall=n.ros_s(),time_monotonic()
        if n.last_clock is None or now>n.last_clock:n.clock_advanced_wall=wall
        if n.last_clock is not None and now<n.last_clock-.1:
            raise ValueError('CLOCK_RESET_BEFORE_FIRST_START')
        n.last_clock=now
        error,inp=n.inputs()
        if error or inp is None:
            n.info('home_wait','[ORIGINAL_START_WAIT] '+str(error),3.);return False
        pose=pose_map(inp['pose'])
        self.data=lock_origin(self.data,self.context,pose,'FIRST_MISSION_FRESH_MAP_TF',now)
        controller.dirty=True
        n.event('ORIGINAL_START_LOCKED_V189',original_start_pose=copy.deepcopy(self.data['original_start_pose']))
        n.info('original_home','[ORIGINAL_START_LOCKED] map='+str([pose[k] for k in ('x','y','yaw')])+'; immutable across Resume',0)
        n.publish_report(force=True)
        if controller.write_failed or controller.last_saved is None:
            raise ValueError('ORIGINAL_START_CHECKPOINT_WRITE_FAILED')
        self.capture_persisted=True
        return True

    def resumed(self, pose, session, ros_time_s):
        self.data['resume_start_pose']=dict(pose_map(pose),frame_id='map',session=session,
                                         ros_time_s=float(ros_time_s),wall_utc=utc())

    def snapshot(self, node):
        result=copy.deepcopy(self.data)
        snap=getattr(node,'_v183_last_pose',None)
        if snap:
            result['checkpoint_robot_pose']=dict(pose_map(snap['pose']),frame_id='map',
                                                ros_time_s=snap['ros_time_s'])
        return validate_navigation(result,self.mission_id,self.context)


def time_monotonic():
    import time
    return time.monotonic()
