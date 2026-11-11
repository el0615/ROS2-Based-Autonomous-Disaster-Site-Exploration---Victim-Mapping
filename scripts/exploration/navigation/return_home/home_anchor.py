"""Standalone, ROS-free completion and immutable home anchor checks.

Original exploration checkpoints are read-only; a separate return journal records home.
"""
from __future__ import annotations
import gzip, hashlib, json, math, re
from pathlib import Path

SCHEMA = 'dn012_return_home_origin_v188'
FINISHED = 'ALL_ZONES_EXPLORATION_COMPLETE'
SESSION = re.compile(r'session_\d{8}_\d{6}_\d{6}')
CPNAME = re.compile(r'checkpoint_[\w-]+\.json\.gz')

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read_checkpoint(pointer: Path):
    pointer = Path(pointer)
    if pointer.is_symlink() or pointer.name != 'LATEST.json':
        raise ValueError('LATEST_POINTER_REQUIRED')
    obj = json.loads(pointer.read_text(encoding='utf-8'))
    name = obj.get('checkpoint')
    if not isinstance(name,str) or not CPNAME.fullmatch(name):
        raise ValueError('UNSAFE_CHECKPOINT_NAME')
    p = pointer.parent / name
    if p.is_symlink() or not p.is_file():
        raise ValueError('CHECKPOINT_MISSING_OR_SYMLINK')
    raw = p.read_bytes()
    if digest(raw) != obj.get('sha256'):
        raise ValueError('CHECKPOINT_FILE_SHA_MISMATCH')
    outer = json.loads(gzip.decompress(raw))
    payload = outer.get('payload')
    if not isinstance(payload,dict):
        raise ValueError('CHECKPOINT_PAYLOAD_INVALID')
    canonical = json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    if digest(canonical) != outer.get('payload_sha256'):
        raise ValueError('CHECKPOINT_PAYLOAD_SHA_MISMATCH')
    if outer.get('payload_sha256') != obj.get('payload_sha256'):
        raise ValueError('CHECKPOINT_POINTER_DIGEST_MISMATCH')
    if not isinstance(outer.get('payload_sha256'),str) or len(outer['payload_sha256']) != 64:
        raise ValueError('CHECKPOINT_DIGEST_MISSING')
    return payload, outer['payload_sha256'], p

def check_completion(session_dir: Path):
    session_dir = Path(session_dir).resolve(strict=True)
    if not SESSION.fullmatch(session_dir.name) or session_dir.parent.name != 'hospital_zone_mission':
        raise ValueError('NOT_AN_APPROVED_MISSION_SESSION')
    manifest = json.loads((session_dir/'run_manifest.json').read_text())
    report = json.loads((session_dir/'summary.json').read_text())
    if manifest.get('options',{}).get('execute') is not True or report.get('execute') is not True:
        raise ValueError('RETURN_REQUIRES_EXECUTE_NOT_DRY_RUN')
    if report.get('state') != FINISHED or report.get('stop_reason') or report.get('owned_goal_id'):
        raise ValueError('EXPLORATION_NOT_NORMALLY_COMPLETE')
    if manifest.get('zone_bundle_signature') != report.get('zone_bundle_signature'):
        raise ValueError('ZONE_BUNDLE_MISMATCH')
    payload, payload_sha, cp_path = read_checkpoint(session_dir/'checkpoints_v18/LATEST.json')
    mission = payload.get('mission') or {}
    lifecycle = mission.get('zone_lifecycle_v187')
    if not isinstance(lifecycle,dict) or lifecycle.get('schema') != 'dn012_explicit_zone_lifecycle_v1_8_7':
        raise ValueError('EXPLICIT_ZONE_LIFECYCLE_REQUIRED')
    rows = lifecycle.get('zones')
    scope = mission.get('scope')
    if (not isinstance(scope,list) or len(scope) != 27 or len(set(scope)) != 27
            or not isinstance(rows,dict) or set(rows) != set(scope)):
        raise ValueError('ZONE_LIFECYCLE_SCOPE_NOT_27')
    for zid, row in rows.items():
        if (not isinstance(row,dict) or row.get('exploration_status') != 'COMPLETE'
                or row.get('phase') != 'TERMINAL'
                or not isinstance(row.get('completion_reason'),str) or not row['completion_reason'].strip()):
            raise ValueError('ZONE_NOT_TERMINAL:'+zid)
    if (payload.get('context',{}).get('bundle_signature') != report.get('zone_bundle_signature')
            or payload.get('coverage',{}).get('source_session') != report.get('coverage_session')):
        raise ValueError('CHECKPOINT_REPORT_CONTEXT_MISMATCH')
    if mission.get('source_execute') is not True or mission.get('source_session') != session_dir.name:
        raise ValueError('CHECKPOINT_SOURCE_SESSION_MISMATCH')
    if report.get('sent') != mission.get('counters',{}).get('sent_goals'):
        raise ValueError('MISSION_GOAL_COUNT_MISMATCH')
    if report.get('nav_succeeded') != mission.get('counters',{}).get('succeeded_goals'):
        raise ValueError('MISSION_GOAL_SUCCESS_COUNT_MISMATCH')
    if mission.get('source_state') != FINISHED:
        raise ValueError('CHECKPOINT_TERMINAL_STATE_MISMATCH')
    chain = payload.get('mission_session_ids')
    if not isinstance(chain,list) or not chain or not all(isinstance(s,str) and SESSION.fullmatch(s) for s in chain):
        raise ValueError('MISSION_SESSION_LINEAGE_MISSING')
    if chain[-1] != session_dir.name:
        raise ValueError('CHECKPOINT_CHAIN_TIP_MISMATCH')
    return dict(session=session_dir.name, checkpoint=str(cp_path),checkpoint_sha256=payload_sha,
                initial_session=chain[0],t5=report['coverage_session'], context=payload['context'],
                finished_zones=27,sent=report['sent'], succeeded=report['nav_succeeded'])

def anchor_identity(verified):
    ctx=verified['context']
    return dict(initial_session=verified['initial_session'],t5_session=verified['t5'],
                bundle_signature=ctx['bundle_signature'],fingerprints={
                key:ctx['fingerprints'][key] for key in ('world','reference','zones','nav2')})

def make_anchor(verified,pose,source):
    for key in ('x','y','yaw'):
        if type(pose.get(key)) not in (int,float) or not math.isfinite(pose[key]):
            raise ValueError('HOME_POSE_INVALID:'+key)
    if abs(pose['x'])>10000 or abs(pose['y'])>10000 or abs(pose['yaw'])>3.1415926535898:
        raise ValueError('HOME_POSE_RANGE')
    from math import sin,cos
    return dict(schema=SCHEMA,identity=anchor_identity(verified),
                pose_map=dict(x=float(pose['x']),y=float(pose['y']),yaw=float(pose['yaw']),
                    qx=0.,qy=0.,qz=sin(float(pose['yaw'])/2),qw=cos(float(pose['yaw'])/2)),
                source=source,frame_id='map',not_resume_start=True)

def validate_anchor(anchor,verified):
    if not isinstance(anchor,dict) or anchor.get('schema')!=SCHEMA:
        raise ValueError('HOME_ANCHOR_SCHEMA_MISMATCH')
    if anchor.get('identity') != anchor_identity(verified):
        raise ValueError('HOME_ANCHOR_OTHER_MISSION_OR_MAP')
    if anchor.get('frame_id')!='map' or anchor.get('not_resume_start') is not True:
        raise ValueError('HOME_ANCHOR_FRAME_INVALID')
    if anchor.get('source') not in ('ORIGINAL_START_LIVE_TF_OPERATOR_CONFIRMED',
                                    'ORIGINAL_START_MAP_POSE_OPERATOR_CONFIRMED'):
        raise ValueError('HOME_ANCHOR_ORIGIN_UNVERIFIED')
    pose=anchor.get('pose_map') or {}
    if any(type(pose.get(k)) not in (int,float) or not math.isfinite(pose[k]) for k in
           ('x','y','yaw','qx','qy','qz','qw')):
        raise ValueError('HOME_ANCHOR_POSE_INVALID')
    if (abs(pose['qx'])>1e-6 or abs(pose['qy'])>1e-6 or
            abs(math.hypot(pose['qz'],pose['qw'])-1)>.0001 or
            abs(math.atan2(math.sin(2*math.atan2(pose['qz'],pose['qw'])-pose['yaw']),
                           math.cos(2*math.atan2(pose['qz'],pose['qw'])-pose['yaw'])))>.001):
        raise ValueError('HOME_ANCHOR_QUAT_INVALID')
    return dict(pose)
