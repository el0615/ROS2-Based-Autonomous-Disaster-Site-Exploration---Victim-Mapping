"""Passive path/LiDAR watcher; cancellation uses the existing owned-goal gate.

Never sends ComputePathToPose while NavigateToPose is active: that could compete
with Nav2's BT planner client. Replanning is done only after terminal cancel,
fresh stopped odometry and new measured Coverage, through the v2 audit adapter.
"""
from __future__ import annotations
import hashlib
import json
import math
import time
from collections import deque
import numpy as np
from path_blockage import (BlockageSettings,PersistentBlockage,CANCEL_REASON,
    path_obstacle_witness,project_scan_hits,validate_cancel_record)
from exploration_io import stamp_s,atomic_json
from hospital_coverage_core import planar_pose

class PathBlockageRuntime:
    def __init__(self,node,bundle,cfg,args,output):
        self.n=node;self.bundle=bundle;self.cfg=cfg;self.args=args;self.output=output
        self.settings=BlockageSettings();self.watch=PersistentBlockage(self.settings)
        self.goal_id=None;self.goal_start=0.;self.path=None;self.hits=None
        self.pending=None;self.last_check=0.;self.serial=0
        self.pending_scan=None;self.pending_scans=deque(maxlen=32);self.projected_scan_stamp=None
        self.records=[];self.zone_counts={};self.latest_by_zone={};self.context_epoch=None
        # DN012_REROUTE_V1_4: separate optional path-change observer.
        import os as _reroute_os
        self.reroute=None
        if _reroute_os.environ.get('DN012_REROUTE_MODE','off') in ('observe','apply'):
            from reroute_runtime import RerouteRuntime
            self.reroute=RerouteRuntime(self,node,bundle,cfg,args,output)
        node.info('path_blockage_policy','[PATH_BLOCKAGE_POLICY] v3; sensor-backed new lethal obstacle + stalled translation; owned cancel then Nav2 replan; fixed-prior-topology='+str(args.prior_topology_fixed),0)
        node.event('PATH_BLOCKAGE_POLICY',enabled=True,version=3,
            fixed_prior_topology=bool(args.prior_topology_fixed),
            cancel_reason=CANCEL_REASON,lookahead_m=self.settings.lookahead_m,
            progress_window_ros_s=self.settings.stationary_window_s,
            persistence_ros_s=self.settings.persistence_s,minimum_samples=self.settings.minimum_samples)

    def begin(self,inp):
        self.goal_id=self.n.nav_uuid;self.goal_start=self.n.ros_s()
        self.context_epoch=inp['epoch'];self.watch.reset();self.pending=None
        self.path=dict(points=list(self.n.current_path),stamp=self.goal_start,
            wall=time.monotonic(),source='validated_preflight_reference')
        if self.reroute:self.reroute.begin(inp)

    def on_plan(self,msg):
        n=self.n
        if not self.goal_id or n.nav_uuid!=self.goal_id or n.nav_terminal is not None or not n.current:return
        try:
            stamp=stamp_s(msg.header.stamp)
            points=[(float(p.pose.position.x),float(p.pose.position.y)) for p in msg.poses]
            frames=[p.header.frame_id.lstrip('/') for p in msg.poses]
            if (msg.header.frame_id.lstrip('/')!='map' or any(f not in ('','map') for f in frames)
                    or not 2<=len(points)<=20000 or not np.isfinite(points).all()
                    or not self.goal_start-.05<=stamp<=n.ros_s()+.1
                    or math.dist(points[-1],(n.current['x'],n.current['y']))>.30):return
            if self.path is not None and stamp<self.path['stamp']:return
            if self.reroute:self.reroute.on_plan(points,stamp)
            self.path=dict(points=points,stamp=stamp,wall=time.monotonic(),source='nav2_plan_topic')
        except (ValueError,TypeError,AttributeError):return

    def on_scan(self,msg):
        # Keep a bounded history: the TF for a just-received scan can arrive
        # one or more ROS callbacks later. Never block the ROS executor here.
        if len(self.pending_scans)==self.pending_scans.maxlen:
            self.pending_scans.popleft()
            self.n.info('blockage_scan_overflow',
                '[PATH_BLOCKAGE_SCAN_QUEUE_DROP] pending scan queue overflow; no stale TF substitution',10.)
        self.pending_scans.append(dict(msg=msg,receive_wall=time.monotonic(),deferred=False))
        self.pending_scan=self.pending_scans[0]['msg']
        self.refresh_scan()

    def refresh_scan(self):
        # Retries are nonblocking, and always query transforms at the exact
        # first/last LiDAR beam times (NEVER at latest TF or time zero).
        from rclpy.time import Time
        from rclpy.duration import Duration
        from tf2_ros import TransformException
        while self.pending_scans:
            item=self.pending_scans[0]
            msg=item['msg']
            try:
                stamp=stamp_s(msg.header.stamp)
                if self.projected_scan_stamp is not None and stamp<=self.projected_scan_stamp:
                    self.pending_scans.popleft();continue
                inc=float(msg.time_increment)
                span=inc*max(0,len(msg.ranges)-1)
                if not math.isfinite(span) or not 0<=span<=.5:
                    raise ValueError('SCAN_SPAN')
                scan_end=stamp+span
                age=self.n.ros_s()-scan_end
                wall_age=time.monotonic()-item['receive_wall']
                if age>1. or wall_age>3.:
                    # Old scans can never become blockage evidence.
                    self.pending_scans.popleft();continue
                if age<-.1:
                    # A future-dated scan must not be projected prematurely.
                    return
                frame=msg.header.frame_id.lstrip('/')
                if not frame:raise ValueError('SCAN_FRAME_MISSING')
                def at(t):
                    tf=self.n.tf_buffer.lookup_transform(
                        'map',frame,Time(nanoseconds=int(t*1e9)),
                        timeout=Duration(seconds=0.))
                    return planar_pose(tf.transform,math.radians(5))
                first=at(stamp)
                last=at(scan_end) if span>0 else first
                points=project_scan_hits(msg.ranges,float(msg.angle_min),float(msg.angle_increment),
                    float(msg.range_min),float(msg.range_max),first,last)
                # Re-check at commit, since a slow transform/projection could
                # otherwise turn a stale scan into apparently fresh evidence.
                if (not -.1<=self.n.ros_s()-scan_end<=1.
                        or time.monotonic()-item['receive_wall']>3.):
                    self.pending_scans.popleft();continue
                self.hits=dict(points=points,stamp=scan_end,wall=item['receive_wall'])
                self.projected_scan_stamp=stamp
                if item['deferred']:
                    self.n.info('blockage_scan_tf_recovered',
                        '[PATH_BLOCKAGE_SCAN_TF_RECOVERED] exact-time TF arrived; buffered scan projected',10.)
                self.pending_scans.popleft()
                self.pending_scan=self.pending_scans[0]['msg'] if self.pending_scans else None
                return
            except TransformException as exc:
                # Keep the pending scan until TF arrives or it becomes stale;
                # spin_once can process TF callbacks between retry attempts.
                item['deferred']=True
                self.n.info('blockage_scan_wait',
                    '[PATH_BLOCKAGE_WAIT_SCAN_TF] '+type(exc).__name__+': '+str(exc),10.)
                return
            except Exception as exc:
                # Invalid geometry/data or other projection errors are not TF
                # delays. Skip this scan instead of holding the whole queue.
                self.n.info('blockage_scan_invalid',
                    '[PATH_BLOCKAGE_SCAN_REJECTED] '+type(exc).__name__+': '+str(exc),10.)
                self.pending_scans.popleft()
        self.pending_scan=None

    def suspend(self):
        self.watch.reset()

    def step(self,inp):
        n=self.n;wall=time.monotonic();now=n.ros_s()
        if self.reroute:self.reroute.tick(inp)
        # DN012_V1_6_SCOPE_GATE: application context only; evidence math unchanged.
        # Keep the independent Reroute observer above active in every mode.
        from path_blockage import enabled_for_node
        enabled=enabled_for_node(n,self.bundle,self.cfg,inp)
        signature=(n.nav_uuid,bool(enabled))
        if getattr(self,'_scope_signature',None)!=signature:
            self.watch.reset();self._scope_signature=signature
            n.event('PATH_BLOCKAGE_SCOPE',goal_id=n.nav_uuid,enabled=bool(enabled),
                goal_kind=(n.current or {}).get('goal_kind'),
                second_pass=bool((n.current or {}).get('second_pass')),
                detector_thresholds_changed=False)
        if not enabled:
            self.suspend();return
        if wall-self.last_check<.5:return
        self.last_check=wall
        self.refresh_scan()
        if (inp is None or n.stop_requested or n.cancel_reason is not None or n.nav_terminal is not None
                or n.nav_handle is None or n.nav_uuid!=self.goal_id or not n.current):
            self.suspend();return
        if self.zone_counts.get(n.current['zone_id'],0)>=self.settings.max_cancel_per_zone:return
        p=self.path;h=self.hits;c=inp['cost']
        if (p is None or h is None or inp['epoch']!=self.context_epoch
                or not -.1<=now-c['stamp']<=5 or wall-c['wall']>15
                or not -.1<=now-h['stamp']<=1. or wall-h['wall']>3.
                or not -.1<=now-p['stamp']<=self.settings.max_reference_age_s
                or wall-p['wall']>self.settings.max_reference_wall_s
                or math.dist(inp['pose'][:2],(n.current['x'],n.current['y']))<.4):
            self.suspend();return
        witness=path_obstacle_witness(p['points'],inp['pose'],self.bundle[1],self.bundle[3],
            c['values'],h['points'],self.cfg,self.settings)
        e=self.watch.update(now,inp['pose'],witness,c['stamp'],h['stamp'])
        if e is None:return
        e.update(cost_context=c['key'],bundle_signature=self.bundle[4],epoch=inp['epoch'],
            path_source=p['source'],path_stamp=p['stamp'],pose=list(inp['pose']),
            goal=[n.current['x'],n.current['y'],n.current['yaw']],zone=n.current['zone_id'],
            path_sha256=hashlib.sha256(json.dumps(p['points']).encode()).hexdigest())
        self.pending=dict(goal_id=self.goal_id,evidence=e)
        self.serial+=1;folder=self.output/'path_blockage';folder.mkdir(exist_ok=True)
        atomic_json(folder/f'blockage_{self.serial:04d}.json',dict(self.pending,
            reference_path=p['points'],note='current route witness, not whole-zone unreachable proof'))
        n.event('PATH_BLOCKED_EVIDENCE',goal_id=self.goal_id,evidence=e)
        n.warn('path_blocked',f'[PATH_BLOCKED_EVIDENCE] {e["zone"]}; new obstacle at {e["xy"]}; stalled; request owned cancellation',0)
        n.request_cancel(CANCEL_REASON)

    def authorizes_cancel(self):
        return bool(self.pending and self.goal_id==self.n.nav_uuid
            and self.pending['goal_id']==self.n.nav_uuid and self.n.cancel_reason==CANCEL_REASON
            and not self.n.stop_requested)

    def after_confirmed_stop(self,inp):
        """Caller already requires terminal status, fresh odometry stop + new Coverage."""
        n=self.n
        if not self.authorizes_cancel() or n.nav_terminal!=5:raise ValueError('UNAPPROVED_BLOCKAGE_CANCEL')
        if not n.local_start_clear(inp):
            n.finish('PATH_BLOCKAGE_LOCAL_START_UNSAFE',2);return False
        r=dict(self.pending,reason=CANCEL_REASON,terminal='CANCELED',stop_confirmed=True,new_coverage=True)
        if not validate_cancel_record(r):raise ValueError('INVALID_BLOCKAGE_CANCEL_EVIDENCE')
        self.records.append(r);z=n.current['zone_id'];self.zone_counts[z]=self.zone_counts.get(z,0)+1
        self.latest_by_zone[z]=r
        n.event('PATH_BLOCKAGE_STOP_CONFIRMED',zone=z,record=r)
        n.info('blockage_stop',f'[PATH_BLOCKAGE_STOP_CONFIRMED] {z}; fresh stop and Coverage; replan before alternate entry',0)
        if not n.book.is_complete(z):
            if self.zone_counts[z]>=self.settings.max_cancel_per_zone:
                n.book.finish_partial(z,'PATH_BLOCKAGE_RETRY_BUDGET_UNVERIFIED',evidence=r)
            else:n.connectivity.note_path_blockage(n.current,r,inp)
        self.pending=None;self.watch.reset()
        return True
