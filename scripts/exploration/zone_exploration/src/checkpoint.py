"""Validated, atomic observation/mission checkpoints. No ROS calls.

A checkpoint is historical evidence, never a live TF, costmap, action, or clock.
Only JSON data is accepted; pickle / eval / shell execution are not used.
Persisted version identifiers remain stable for existing checkpoint readers.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import base64
import copy
import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
import numpy as np

VERSION = 'DN012_UNIFIED_RESUME_V1_8'
SCHEMA = 'dn012_checkpoint_v1_8'
MAX_COMPRESSED = 64 * 1024 * 1024
MAX_JSON = 256 * 1024 * 1024
MIN_AREA_M2 = 1.0


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('CHECKPOINT_NONFINITE_NUMBER')
        return value
    if isinstance(value, (tuple, list, set)):
        return [plain(x) for x in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: plain(v) for k, v in value.items()}
    raise TypeError('CHECKPOINT_UNSUPPORTED_TYPE:' + type(value).__name__)


def encode(value):
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_COMPRESSED:
        raise ValueError('CHECKPOINT_FILE_TOO_LARGE')
    raw = path.read_bytes()
    if path.suffix == '.gz':
        # Bound decompression, including deliberately corrupted/gigantic inputs.
        import io
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            raw = stream.read(MAX_JSON + 1)
    if len(raw) > MAX_JSON:
        raise ValueError('CHECKPOINT_JSON_TOO_LARGE')
    value = json.loads(raw.decode('utf-8'),
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))
    if not isinstance(value, dict):
        raise ValueError('CHECKPOINT_OBJECT_REQUIRED')
    return value


def geometry_dict(geom):
    return dict(frame_id='map', width=int(geom.width), height=int(geom.height),
                resolution=float(geom.resolution), origin=list(map(float, geom.origin)),
                row_order='ROS bottom-up')


def runs_from_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    result = []
    for j, row in enumerate(mask):
        changes = np.diff(np.r_[False, row, False].astype(np.int8))
        result.extend([int(j), int(a), int(b)] for a, b in
                      zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    return result


def mask_from_runs(runs, height, width):
    if type(height) is not int or type(width) is not int or not (0 < height * width <= 20_000_000):
        raise ValueError('CHECKPOINT_INVALID_GEOMETRY')
    if not isinstance(runs, list):
        raise ValueError('CHECKPOINT_RUNS_REQUIRED')
    mask = np.zeros((height, width), dtype=bool)
    previous = (-1, 0)
    for run in runs:
        if not isinstance(run, (list, tuple)) or len(run) != 3 or any(type(v) is not int for v in run):
            raise ValueError('CHECKPOINT_INVALID_RUN')
        j, a, b = run
        if not (0 <= j < height and 0 <= a < b <= width):
            raise ValueError('CHECKPOINT_RUN_OUT_OF_GRID')
        if j < previous[0] or (j == previous[0] and a < previous[1]):
            raise ValueError('CHECKPOINT_RUN_OVERLAP_OR_UNSORTED')
        mask[j, a:b] = True
        previous = (j, b)
    return mask


def coverage_payload(bundle, observed, status):
    data, geom, labels, known, signature = bundle
    observed = np.asarray(observed)
    if observed.dtype != bool or observed.shape != labels.shape:
        raise ValueError('CHECKPOINT_OBSERVATION_SHAPE')
    if np.any(observed & ((labels <= 0) | ~known)):
        raise ValueError('CHECKPOINT_OBSERVATION_OUTSIDE_APPROVED_MASK')
    if status.get('zone_bundle_signature') != signature:
        raise ValueError('CHECKPOINT_COVERAGE_BUNDLE_MISMATCH')
    count = int(observed.sum())
    if status.get('observed_cells') != count:
        raise ValueError('CHECKPOINT_COVERAGE_PAIR_MISMATCH')
    by_id = {z['id']: z for z in status.get('zones', [])}
    if len(by_id) != len(data['zones']) or len(status.get('zones', [])) != len(data['zones']):
        raise ValueError('CHECKPOINT_ZONE_COUNT_MISMATCH')
    for z in data['zones']:
        target = labels == z['mask_id']
        old = by_id.get(z['id'], {})
        if (old.get('reference_free_cells') != int(target.sum()) or
                old.get('observed_cells') != int((target & observed).sum())):
            raise ValueError('CHECKPOINT_ZONE_COUNTS_MISMATCH:' + z['id'])
    if status.get('reference_free_survey_cells') != int((labels > 0).sum()):
        raise ValueError('CHECKPOINT_DENOMINATOR_MISMATCH')
    return dict(schema='observed_mask_ros_row_runs_v1', geometry=geometry_dict(geom),
                zone_bundle_signature=signature, source_session=status['session_id'],
                observed_cells=count, runs=runs_from_mask(observed),
                status_snapshot=copy.deepcopy(status))


def validate_coverage(payload, bundle):
    data, geom, labels, known, signature = bundle
    if payload.get('geometry') != geometry_dict(geom):
        raise ValueError('CHECKPOINT_GEOMETRY_MISMATCH')
    if payload.get('zone_bundle_signature') != signature:
        raise ValueError('CHECKPOINT_BUNDLE_MISMATCH')
    if payload.get('source_session') != payload.get('status_snapshot', {}).get('session_id'):
        raise ValueError('CHECKPOINT_SOURCE_SESSION_MISMATCH')
    mask = mask_from_runs(payload.get('runs'), geom.height, geom.width)
    verified = coverage_payload(bundle, mask, payload.get('status_snapshot', {}))
    if verified['observed_cells'] != payload.get('observed_cells'):
        raise ValueError('CHECKPOINT_MASK_COUNT_MISMATCH')
    return mask


def import_legacy_coverage(directory, bundle):
    directory = Path(directory).expanduser().resolve()
    status = read_json(directory / 'coverage_status.json')
    mask = read_json(directory / 'observed_mask.json.gz')
    if status.get('session_id') != directory.name:
        raise ValueError('LEGACY_COVERAGE_SESSION_MISMATCH')
    if (mask.get('schema') != 'observed_mask_ros_row_runs_v1' or
            mask.get('row_order') != 'ROS bottom-up' or
            mask.get('zone_bundle_signature') != bundle[4]):
        raise ValueError('LEGACY_COVERAGE_SCHEMA_OR_SIGNATURE_MISMATCH')
    if (mask.get('height'), mask.get('width')) != (bundle[1].height, bundle[1].width):
        raise ValueError('LEGACY_COVERAGE_DIMENSION_MISMATCH')
    observed = mask_from_runs(mask.get('runs'), bundle[1].height, bundle[1].width)
    if mask.get('observed_cells') != int(observed.sum()):
        raise ValueError('LEGACY_COVERAGE_MASK_COUNT_MISMATCH')
    return coverage_payload(bundle, observed, status)


def project_context(root, bundle, namespace='/j100_0001'):
    root = Path(root).expanduser().resolve()
    world = Path(os.environ.get('DN012_WORLD_FILE', str(root / 'gazebo/worlds/disaster/disaster_hospital.sdf'))).expanduser()
    files = {
        'world': world,
        'zones': root / 'config/exploration/navigation/hospital_zones.json',
        'reference': root / 'config/exploration/navigation/hospital_coverage_reference.json.gz',
        'nav2': root / 'config/nav2/nav2_hospital_navigation.yaml',
    }
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise ValueError('CHECKPOINT_CONTEXT_FILE_MISSING:' + ';'.join(missing))
    # Do not include absolute project paths: relocating the same project is valid.
    return dict(bundle_signature=bundle[4], geometry=geometry_dict(bundle[1]),
                namespace='/' + namespace.strip('/'),
                fingerprints={name: sha(path.read_bytes()) for name, path in files.items()},
                min_unobserved_area_m2=MIN_AREA_M2,
                scope='DISK_WORLD_AND_MAP_IDENTITY; operator must confirm unchanged live scene')


def assert_context(payload, context):
    if payload.get('context') != context:
        raise ValueError('CHECKPOINT_CONTEXT_CHANGED: world/map/zones/Nav2/namespace differ')


def load_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / 'LATEST.json'
    if path.name == 'LATEST.json':
        pointer = read_json(path)
        name = pointer.get('checkpoint')
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('CHECKPOINT_INVALID_POINTER')
        target = path.parent / name
        if target.is_symlink() or sha(target.read_bytes()) != pointer.get('sha256'):
            raise ValueError('CHECKPOINT_POINTER_HASH_MISMATCH')
        path = target
    envelope = read_json(path)
    payload = envelope.get('payload')
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA:
        raise ValueError('CHECKPOINT_WRONG_SCHEMA')
    digest = sha(encode(payload))
    if digest != envelope.get('payload_sha256'):
        raise ValueError('CHECKPOINT_PAYLOAD_HASH_MISMATCH')
    return payload, digest, path


def save_checkpoint(directory, payload, retain=3):
    directory = Path(directory)
    payload = plain(dict(payload, schema=SCHEMA, version=VERSION))
    digest = sha(encode(payload))
    raw = gzip.compress(encode(dict(payload=payload, payload_sha256=digest)), mtime=0)
    name = 'checkpoint_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_') + uuid.uuid4().hex[:8] + '.json.gz'
    target = directory / name
    atomic_bytes(target, raw)
    # Publish pointer only after the complete, fsync'ed, hash-checked snapshot.
    if sha(target.read_bytes()) != sha(raw):
        raise OSError('CHECKPOINT_WRITE_VERIFICATION_FAILED')
    atomic_bytes(directory / 'LATEST.json', encode(dict(checkpoint=name, sha256=sha(raw),
                 payload_sha256=digest, version=VERSION)))
    old = sorted(directory.glob('checkpoint_*.json.gz'), key=lambda p: p.name, reverse=True)
    for p in old[max(2, int(retain)):]:
        if p.is_file() and not p.is_symlink():
            p.unlink()
    return target, digest


def restore_coverage(coverage, payload, bundle):
    observed = validate_coverage(payload, bundle)
    coverage.observed = observed.copy()
    labels = bundle[2]
    coverage.observed_counts = np.bincount(labels[observed], minlength=len(coverage.denominators))
    coverage.observed_total = int(observed.sum())
    # Old ROS timestamps/scan counts cannot establish freshness after reset.
    coverage.scan_count = 0
    coverage.evidence_scan_count = 0
    coverage.last_scan_end_s = None
    coverage.threshold_since.clear()
    coverage.threshold_scan_count.clear()
    coverage.threshold_confirmed.clear()


def compatible_live_coverage(saved, live, status, parent_digest=None):
    # A warmer T5 may have additional observations; it must never lose saved cells.
    old = np.asarray(saved, dtype=bool)
    live = np.asarray(live, dtype=bool)
    if old.shape != live.shape or np.any(old & ~live):
        raise ValueError('RESUME_LIVE_COVERAGE_LOST_SAVED_CELLS')
    if status.get('data_health') != 'LIVE' or not status.get('map_validated') or status.get('fault'):
        raise ValueError('RESUME_REQUIRES_FRESH_VALIDATED_COVERAGE')


def enforce_area_policy():
    # The user approved 1 m2 for BOTH diagnosis and region-entry selection.
    for key in ('DN012_REGION_MIN_AREA_M2', 'DN012_SECOND_ENTRY_MIN_AREA_M2',
                'DN012_REGION_PRIORITY_AREA_M2'):
        os.environ[key] = '1.0'
