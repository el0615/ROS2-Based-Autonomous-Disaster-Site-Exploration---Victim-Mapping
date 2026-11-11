"""DN012 v1.7 evidence-only sidecar and all-bundle spatial-zone lookup.

No ROS imports, navigation requests, parameter writes or mask publication.
The existing v1.4 PointStore remains the mutable operational/mask view.
This sidecar retains immutable first-PENDING and first-CONFIRMED snapshots,
including full path evidence, even if that operational view is later updated.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import re
import tempfile

PATCH_VERSION = 'DN012_REROUTE_EVIDENCE_ZONE_V1_7'
SCHEMA = 'dn012_reroute_evidence_v1_7'


def detached(value):
    """A strict JSON copy: no references back into live detector/scan state."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _raw(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + '\n').encode('utf-8')


def _safe_path(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('EVIDENCE_SYMLINK_REFUSED')
    return path


def _sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path, data, *, once):
    """Publish a complete, fsynced file; immutable files use link, not replace.

    Hard-link publication is atomic/no-clobber on the local Ubuntu filesystem.
    Fail rather than falling back to a partially visible immutable file.
    """
    path = _safe_path(path)
    raw = _raw(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        if once:
            try:
                os.link(temp, path)
            except FileExistsError:
                _safe_path(path)
                return False
        else:
            os.replace(temp, path)
        _sync_directory(path.parent)
        return True
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class ZoneLookup:
    """Point -> existing ROS-order label -> ALL bundle zone names.

    Does not change node.zones_by_id (the intentionally selected mission scope),
    labels, Coverage denominators, boundary rules, or goal selection.
    No nearest-zone/target-zone fallback for outside or unlabeled cells.
    """
    def __init__(self, bundle):
        data, self.geometry, self.labels, _, _ = bundle
        self.by_mask_id = {}
        ids = set()
        for zone in data['zones']:
            mid, zid = zone['mask_id'], zone['id']
            if (type(mid) is not int or mid <= 0 or not isinstance(zid, str)
                    or not zid or mid in self.by_mask_id or zid in ids):
                raise ValueError('INVALID_ALL_ZONE_CATALOG')
            self.by_mask_id[mid] = zid
            ids.add(zid)
        if not self.by_mask_id:
            raise ValueError('EMPTY_ALL_ZONE_CATALOG')
        if tuple(self.labels.shape) != (self.geometry.height, self.geometry.width):
            raise ValueError('ZONE_LOOKUP_GEOMETRY_MISMATCH')

    def locate(self, point):
        out = dict(zone_id=None, mask_id=None, cell_ros=None,
                   scope='ALL_BUNDLE_ZONES', status='INVALID_COORDINATE', frame_id='map')
        try:
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                return out
            gx, gy = self.geometry.world_to_grid_float(x, y)
            i, j = math.floor(gx), math.floor(gy)
        except (TypeError, ValueError, IndexError, OverflowError):
            return out
        out['cell_ros'] = [i, j]
        if not self.geometry.contains(i, j):
            out['status'] = 'OUTSIDE_MAP'
            return out
        mid = int(self.labels[j, i])
        out['mask_id'] = mid
        if mid == 0:
            out['status'] = 'UNASSIGNED_CELL'
        elif mid not in self.by_mask_id:
            out['status'] = 'MASK_ID_NOT_IN_ZONE_CATALOG'
        else:
            out['zone_id'] = self.by_mask_id[mid]
            out['status'] = 'MATCHED'
        return out


class EvidenceJournal:
    """Per-mission historical evidence, NOT a source of navigation keepouts.

    A pending point stays historical if confirmation never arrives. This module
    cannot promote it to physical blockage, enable a mask, or cancel a goal.
    """
    def __init__(self, folder, context_hash):
        self.folder = _safe_path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.context_hash = str(context_hash)
        self.entries = {}
        # Recover the index from immutable source files, not from a possibly
        # stale index left by a process interruption.
        ids = set()
        for phase in ('pending', 'confirmed'):
            for path in self.folder.glob('*.' + phase + '.json'):
                ident = path.name.split('.')[0]
                if re.fullmatch(r'[0-9a-f]{20}', ident):
                    ids.add(ident)
        for ident in sorted(ids):
            self.entries[ident] = self._entry(ident)

    def _read(self, ident, phase):
        path = _safe_path(self.folder / (ident + '.' + phase.lower() + '.json'))
        if not path.exists():
            return None, None
        raw = path.read_bytes()
        data = json.loads(raw)
        if (data.get('schema') != SCHEMA or data.get('record_id') != ident
                or data.get('context_hash') != self.context_hash
                or data.get('phase') != phase):
            raise ValueError('EVIDENCE_IDENTITY_MISMATCH')
        if (data.get('physical_blockage_proven') is not False
                or data.get('path_execution_confirmed') is not False
                or data.get('plan_change_confirmed') is not (phase == 'CONFIRMED')
                or (phase == 'PENDING' and data.get('mask_eligible'))):
            raise ValueError('EVIDENCE_PROOF_OR_PHASE_MISMATCH')
        return data, hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _compact(data):
        if data is None:
            return None
        keys = ('phase', 'anchor_xy', 'anchor_ros_s', 'confirmed_ros_s',
                'point_xy', 'spatial_zone', 'spatial_lookup', 'anchor_spatial_lookup',
                'destination_zone', 'correction_source', 'captured_wall_utc',
                'mask_eligible', 'mask_hold_reason', 'observation_context')
        d = {key: data.get(key) for key in keys}
        metric = data.get('metric', {})
        d['metric'] = {k: v for k, v in metric.items()
                       if k not in ('old_remaining', 'new_remaining')}
        for name in ('old_remaining', 'new_remaining'):
            path = metric.get(name)
            d[name + '_sha256'] = None if path is None else hashlib.sha256(
                json.dumps(path, sort_keys=True, separators=(',', ':'),
                           allow_nan=False).encode()).hexdigest()
        return detached(d)

    def _entry(self, ident):
        pending, psha = self._read(ident, 'PENDING')
        confirmed, csha = self._read(ident, 'CONFIRMED')
        first = pending if pending is not None else confirmed
        if first is None:
            raise ValueError('EVIDENCE_SNAPSHOT_MISSING')
        if pending is not None and confirmed is not None:
            for key in ('goal_id', 'anchor_ros_s', 'anchor_xy', 'goal_xy'):
                if pending.get(key) != confirmed.get(key):
                    raise ValueError('CONFIRMATION_ANCHOR_MISMATCH:' + key)
        return dict(record_id=ident, goal_id=first['goal_id'],
            status='CONFIRMED' if confirmed is not None else 'PENDING',
            first_pending_available=pending is not None,
            pending_file=None if pending is None else str(self.folder / (ident + '.pending.json')),
            confirmation_file=None if confirmed is None else str(self.folder / (ident + '.confirmed.json')),
            pending_sha256=psha, confirmation_sha256=csha,
            first_detection=self._compact(first), confirmation=self._compact(confirmed),
            physical_blockage_proven=False, path_execution_confirmed=False)

    def record(self, ident, snapshot):
        if not re.fullmatch(r'[0-9a-f]{20}', ident):
            raise ValueError('INVALID_EVIDENCE_RECORD_ID')
        data = detached(snapshot)
        confirmed = data.get('plan_change_confirmed')
        if type(confirmed) is not bool:
            raise ValueError('INVALID_CONFIRMATION_FLAG')
        if (data.get('physical_blockage_proven') is not False
                or data.get('path_execution_confirmed') is not False
                or (not confirmed and data.get('mask_eligible'))):
            raise ValueError('INVALID_EVIDENCE_PROOF_FLAGS')
        phase = 'CONFIRMED' if confirmed else 'PENDING'
        # Refuse inconsistent linkage before publishing an immutable file.
        other, _ = self._read(ident, 'PENDING' if confirmed else 'CONFIRMED')
        if other is not None:
            for key in ('goal_id', 'anchor_ros_s', 'anchor_xy', 'goal_xy'):
                if data.get(key) != other.get(key):
                    raise ValueError('CONFIRMATION_ANCHOR_MISMATCH:' + key)
        data.update(schema=SCHEMA, patch_version=PATCH_VERSION, record_id=ident,
                    context_hash=self.context_hash, phase=phase,
                    captured_wall_utc=datetime.now(timezone.utc).isoformat())
        data['pending_record_id'] = ident if confirmed else None
        data['pending_link_means_same_event_not_physical_proof'] = True
        path = self.folder / (ident + '.' + phase.lower() + '.json')
        _write_json(path, data, once=True)
        self.entries[ident] = self._entry(ident)
        _write_json(self.folder / 'index.json', self.summary(), once=False)
        return detached(self.entries[ident])

    def summary(self):
        entries = [self.entries[k] for k in sorted(self.entries)]
        return detached(dict(schema=SCHEMA, patch_version=PATCH_VERSION,
            context_hash=self.context_hash, index_file=str(self.folder / 'index.json'),
            counts=dict(total=len(entries),
                        pending=sum(r['status'] == 'PENDING' for r in entries),
                        confirmed=sum(r['status'] == 'CONFIRMED' for r in entries)),
            records=entries, snapshots_are_historical=True,
            controls_masks_or_navigation=False,
            note='PENDING is retained evidence, not proof of blockage or an active mask.'))
