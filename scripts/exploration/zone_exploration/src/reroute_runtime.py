"""Opt-in v1.4 adapter. Observes owned Nav2 paths; never sends Goals/cmd_vel.

The existing detector and cancellation owner remain in charge of motion.
This adapter may request a controlled stop through the original node when its
apply-mode dependency fails. New memory is a separate schema/file: v1.3 memory
and all original evidence remain untouched.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import json
import math
import os
import time
from reroute_core import (VERSION,RerouteSettings,RerouteDetector,PointStore,
    atomic_json,context_for,geometry_dict,json_hash,choose_point,point_path_distance)
from reroute_evidence import PATCH_VERSION, ZoneLookup, EvidenceJournal
from blocked_policy import annotate_reroute


class RerouteRuntime:
    def __init__(self,monitor,node,bundle,cfg,args,output):
        self.monitor=monitor;self.n=node;self.bundle=bundle;self.cfg=cfg
        self.args=args;self.output=Path(output);self.queue=deque(maxlen=16)
        self.mode=os.environ.get('DN012_REROUTE_MODE','off')
        if self.mode not in ('observe','apply'):raise ValueError('INVALID_REROUTE_MODE')
        root=Path(os.environ['ZB_PROJECT_ROOT']).resolve()
        self.settings_file=root/'config/exploration/navigation/dn012_reroute_v1_4.json'
        self.config=json.loads(self.settings_file.read_text(encoding='utf-8'))
        self.s=RerouteSettings(**self.config['detector']).validate()
        self.detector=RerouteDetector(self.s)
        scene=os.environ.get('DN012_STATIC_SCENE_ID',self.config['scene_id'])
        ns='/'+str(args.namespace).strip('/')
        self.context=context_for(root,scene,ns);geom=geometry_dict(bundle[1])
        self.store=PointStore(self.output/'memory/reroute_points.json',
            self.context,geom,self.s.max_records)
        atomic_json(self.store.path,self.store.data)
        self.status=None;self.status_wall=None;self.bad_since=None
        self.comparisons=0;self.recorded=0;self.last_reason=None;self.current_goal=None
        self.closed=False
        self.trace_dir=self.output/'reroute_v1_4';self.trace_dir.mkdir(exist_ok=True)
        self.metrics=(self.trace_dir/'metrics.jsonl').open('x',encoding='utf-8')
        self.metric_lines=0
        self.evidence_zones=ZoneLookup(bundle)
        self.evidence=EvidenceJournal(self.output/'reroute_evidence_v1_7',self.store.context_hash)
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)
        self.status_sub=node.create_subscription(String,ns+'/dn012_reroute/status',self.on_status,qos)
        self.n.event('REROUTE_POLICY',version=VERSION,mode=self.mode,settings=self.config['detector'],
            memory_file=str(self.store.path),loaded_points=len(self.store.data['records']),
            static_scene_id=scene,point_only=True,physical_cut_claim=False,
            baseline='FIRST_NAV2_PATH_THEN_REBASE_AFTER_CONFIRMED_DETOUR',
            source='owned_goal_time_endpoint_filtered_plan_topic')
        self.n.info('reroute_policy',f'[REROUTE_POLICY] {VERSION}; mode={self.mode}; '
            f'AND: +{self.s.min_extra_m}m and +{100*self.s.min_fraction:.1f}%; '
            f'loaded={len(self.store.data["records"])}',0)

    def on_status(self,msg):
        try:
            d=json.loads(msg.data)
            if d.get('context_hash')!=self.store.context_hash:return
            if d.get('memory_file')!=str(self.store.path.resolve()):return
            if d.get('schema')!='dn012_reroute_mask_status_v1':return
            self.status=d;self.status_wall=time.monotonic()
        except (ValueError,TypeError):return

    def readiness(self,inp):
        if self.mode!='apply':return True,'OBSERVE_ONLY'
        if self.status is None or self.status_wall is None:return False,'WAIT_REROUTE_MASK_SERVER'
        if time.monotonic()-self.status_wall>4.:return False,'REROUTE_MASK_SERVER_STALE'
        d=self.status
        if d.get('mode')!='apply':return False,'REROUTE_MASK_NOT_APPLY_MODE'
        if d.get('active_overlap'):return False,'CURRENT_POSE_OVERLAPS_ACTIVE_MASK'
        if not d.get('ready'):return False,str(d.get('reason','REROUTE_MASK_NOT_READY'))
        if d.get('global_applied_revision')!=d.get('revision'):
            return False,'WAIT_GLOBAL_MEMORY_COSTMAP_ACK'
        return True,'REROUTE_MASK_READY'

    def health(self,inp):
        ok,reason=self.readiness(inp)
        if ok:self.bad_since=None;return
        self.last_reason=reason
        if self.bad_since is None:self.bad_since=time.monotonic()
        inflight=self.n.nav_uuid is not None and self.n.nav_terminal is None
        if inflight and (reason=='CURRENT_POSE_OVERLAPS_ACTIVE_MASK' or
                         time.monotonic()-self.bad_since>6.):
            self.n.request_stop(reason)

    def ready(self,inp):
        ok,reason=self.readiness(inp)
        if not ok:self.n.info('reroute_not_ready','['+reason+'] no new planning/navigation',3.)
        return ok

    def begin(self,inp):
        self.queue.clear();self.current_goal=self.n.nav_uuid
        c=self.n.current
        self.detector.reset(self.current_goal,(c['x'],c['y']),self._revision())

    def _revision(self):
        # Ignore self-induced path changes after mask updates. Also in observe
        # mode, an already-running apply publisher can affect the Nav2 path.
        if self.status is None:return 'NO_MASK_SERVER'
        return self.status.get('revision','NO_REVISION')

    def on_plan(self,points,stamp):
        n=self.n
        if n.nav_uuid is None or n.stop_requested or n.cancel_reason is not None:return
        if n.nav_terminal is not None or not n.current:return
        if n.foreign_active():return
        self.queue.append(dict(points=list(points),stamp=float(stamp),
            received=time.monotonic(),goal_id=n.nav_uuid,epoch=n.coverage_epoch,
            goal=[float(n.current['x']),float(n.current['y'])],revision=self._revision()))

    def _pose_at(self,stamp):
        from rclpy.time import Time
        from rclpy.duration import Duration
        from hospital_coverage_core import planar_pose
        tf=self.n.tf_buffer.lookup_transform('map',self.args.robot_frame,
            Time(nanoseconds=int(stamp*1e9)),timeout=Duration(seconds=0.))
        return planar_pose(tf.transform,math.radians(5))

    def tick(self,inp):
        if inp is None or self.n.stop_requested or self.n.cancel_reason is not None:return
        n=self.n
        if n.nav_handle is None or n.nav_terminal is not None or n.foreign_active():return
        while self.queue:
            q=self.queue[0];now=n.ros_s()
            if (q['goal_id']!=n.nav_uuid or q['epoch']!=inp['epoch'] or
                    time.monotonic()-q['received']>2. or now-q['stamp']>2.):
                self.queue.popleft();continue
            if q['stamp']>now+.1:return
            try:pose=self._pose_at(q['stamp'])
            except Exception:
                # Wait for exact-time TF, never substitute a future/current pose.
                if time.monotonic()-q['received']<1.:return
                self.queue.popleft()
                n.event('REROUTE_SAMPLE_SKIPPED',reason='EXACT_PATH_TIME_TF_MISSING',stamp=q['stamp'])
                continue
            self.queue.popleft()
            # A mask revision received after this path invalidates the comparison.
            if q['revision']!=self._revision():
                self.detector.reset(n.nav_uuid,q['goal'],self._revision());continue
            try:
                event,reason=self.detector.feed(q['points'],pose,q['stamp'],q['goal_id'],q['goal'],
                    revision=q['revision'],epoch=q['epoch'])
            except ValueError as exc:
                n.event('REROUTE_SAMPLE_SKIPPED',reason=str(exc));continue
            self.comparisons+=1;self.last_reason=reason
            self._metric(dict(goal_id=q['goal_id'],ros_time_s=q['stamp'],pose=list(pose),reason=reason,
                metrics=getattr(self.detector,'last_metric',None)))
            if event is not None:self._record(event,inp,q['points'])

    def _metric(self,d):
        if self.metric_lines>=20000:return
        self.metrics.write(json.dumps(d,allow_nan=False,ensure_ascii=False)+'\n')
        self.metrics.flush();self.metric_lines+=1

    def _record(self,event,inp,new_path):
        from path_blockage import path_obstacle_witness
        h=self.monitor.hits;sensor=None
        if (h and -.1<=self.n.ros_s()-h['stamp']<=1. and time.monotonic()-h['wall']<=3.):
            sensor=path_obstacle_witness(event['metric']['old_remaining'],inp['pose'],
                self.bundle[1],self.bundle[3],inp['cost']['values'],h['points'],self.cfg,self.monitor.settings)
        point,source=choose_point(event,self.s,sensor_xy=None if sensor is None else sensor['xy'])
        radius=self.s.mask_radius(self.bundle[1].resolution)
        # Existing eligibility/point selection is intentionally unchanged.
        i,j=map(math.floor,self.bundle[1].world_to_grid_float(*point))
        prior_free=bool(self.bundle[1].contains(i,j) and self.bundle[3][j,i])
        separated=point_path_distance(new_path,point)>radius+self.s.robot_envelope_m+self.s.safety_margin_m+.05
        eligible=bool(event.get('plan_change_confirmed') and prior_free and separated)
        hold_reason=('PENDING_PLAN_CONFIRMATION' if not event.get('plan_change_confirmed') else
            'POINT_NOT_APPROVED_PRIOR_FREE' if not prior_free else
            'POINT_TOO_CLOSE_TO_NEW_ROUTE' if not separated else None)
        # Mission scope stays unchanged. Spatial labels use all bundle zones.
        spatial=self.evidence_zones.locate(point)
        anchor_spatial=self.evidence_zones.locate(event['anchor_xy'])
        spatial_zone=spatial['zone_id']
        ident=json_hash([self.store.context_hash,event['goal_id'],event['anchor_ros_s']])[:20]
        snapshot=dict(event,point_xy=point,correction_source=source,
            mask_eligible=eligible,mask_radius_m=radius,sensor_evidence=sensor,
            mask_hold_reason=hold_reason,spatial_zone=spatial_zone,
            destination_zone=self.n.current['zone_id'],spatial_lookup=spatial,
            anchor_spatial_lookup=anchor_spatial,thresholds=self.config['detector'],
            observation_context=dict(record_ros_s=float(self.n.ros_s()),
                record_pose=list(inp['pose']),costmap_stamp=inp['cost'].get('stamp'),
                costmap_key=inp['cost'].get('key'),
                scan_stamp=None if h is None else h.get('stamp'),
                timing='anchor is exact path-time pose; sensor/cost context is record-time'))
        # Persist first evidence BEFORE the mutable operational PointStore view.
        # New confirmation goes to another file; original PENDING is not replaced.
        snapshot=annotate_reroute(snapshot)
        historical=self.evidence.record(ident,snapshot)
        stored_id,added=self.store.append(event,point,settings=self.s,source=source,eligible=eligible,
            zone=self.n.current['zone_id'],spatial_zone=spatial_zone,
            sensor_evidence=sensor,hold_reason=hold_reason)
        if stored_id!=ident:raise ValueError('REROUTE_EVIDENCE_STORE_ID_MISMATCH')
        links=dict(pending_file=historical['pending_file'],
            confirmation_file=historical['confirmation_file'],
            first_pending_available=historical['first_pending_available'])
        # Preserve the old latest-file interface for existing readers.
        atomic_json(self.trace_dir/(ident+'.json'),dict(snapshot,evidence_v1_7=links))
        self.recorded+=int(added)
        self.n.event('REROUTE_POINT_RECORDED',record_id=ident,new_record=added,
            plan_change_confirmed=bool(event.get('plan_change_confirmed')),
            thresholds=self.config['detector'], confirm_metric=event.get('confirm_metric'),
            blockage_status=snapshot['blockage_status'], decision_basis=snapshot['decision_basis'],
            blockage_policy=snapshot['blockage_policy'],
            anchor_xy=event['anchor_xy'],point_xy=point,extra_m=event['metric']['extra_m'],
            increase_fraction=event['metric']['increase_fraction'],correction_source=source,
            mask_eligible=eligible,mask_hold_reason=hold_reason,spatial_zone=spatial_zone,
            spatial_mask_id=spatial['mask_id'],spatial_zone_status=spatial['status'],
            anchor_spatial_zone=anchor_spatial['zone_id'],
            evidence_v1_7=links,patch_version=PATCH_VERSION,
            physical_blockage_proven=False,path_execution_confirmed=False)
        self.n.info('reroute_saved',f'[REROUTE_POINT_RECORDED] {ident}; status={snapshot["blockage_status"]}; point={point}; '
            f'extra={event["metric"]["extra_m"]:.2f}m; +{event["metric"]["increase_percent"]:.1f}%; '
            f'mask_eligible={eligible}; spatial_zone={spatial_zone}; '
            f'first_pending_preserved={historical["first_pending_available"]}',0)

    def summary(self):
        d=dict(version=VERSION,mode=self.mode,comparisons=self.comparisons,
            recorded_this_run=self.recorded,memory_path=str(self.store.path),
            last_reason=self.last_reason,context_hash=self.store.context_hash,
            points=self.store.data['records'],physical_cut_proof=False)
        d['evidence_v1_7']=self.evidence.summary()
        d['spatial_lookup_scope']='ALL_BUNDLE_ZONES'
        d['spatial_catalog_size']=len(self.evidence_zones.by_mask_id)
        if self.status is not None:
            d['mask_status']={k:self.status.get(k) for k in ('ready','reason','revision',
                'global_applied_revision','active_ids','pending_near_robot','active_overlap')}
        d['points']=[annotate_reroute(r) for r in d['points']]
        return d

    def close(self):
        if not self.closed:self.metrics.close();self.closed=True
