"""Synthetic filesystem/ROS-interface tests; no live DDS, Nav2 or Gazebo."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from checkpoint import coverage_payload, save_checkpoint, load_checkpoint, encode, sha
import resume_selection as a
import resume_runtime as rt

@dataclass
class Geom:
    width: int=12
    height: int=8
    resolution: float=1.
    origin: tuple=(0.,0.,0.)
    def contains(self,i,j):return 0<=i<self.width and 0<=j<self.height
    def world_to_grid_float(self,x,y):
        dx=x-self.origin[0];dy=y-self.origin[1];c=math.cos(self.origin[2]);s=math.sin(self.origin[2])
        return ((dx*c+dy*s)/self.resolution,(-dx*s+dy*c)/self.resolution)


def fixture():
    g=Geom();known=np.ones((8,12),bool);labels=np.ones((8,12),np.int16)
    labels[:,4:8]=2;labels[:,8:]=3
    zones=[dict(id=z,mask_id=i+1,known_free_cells=32) for i,z in enumerate(['R01','R10B','CORRIDOR_E'])]
    b=({'zones':zones},g,labels,known,'synthetic-signature')
    mask=np.zeros_like(known);mask[:,:4]=True;mask[0,4]=True
    s=dict(schema='hospital_room_coverage_v1',session_id='coverage-live',zone_bundle_signature=b[4],
           observed_cells=int(mask.sum()),reference_free_survey_cells=96,
           zones=[dict(id=z['id'],reference_free_cells=32,observed_cells=int((mask & (labels==z['mask_id'])).sum())) for z in zones],
           data_health='LIVE',map_validated=True,fault=None,ros_time_s=100.,evidence_scans=10,last_evidence_scan_end_s=99.9)
    mission=dict(scope=['R01','R10B','CORRIDOR_E'],threshold=.96,
                 counters=dict(sent_goals=50,plan_attempts=127,succeeded_goals=43,aborted_goals=0,coverage_cancels=0),
                 records=dict(R01=dict(state='OBSERVATION_SUFFICIENT',measured=1.,views=1,deferred_reason=None),
                   R10B=dict(state='PARTIAL_DEFERRED',measured=1/32,views=0,deferred_reason='PATH_BLOCKED_NO_CONFIRMED_DETOUR'),
                   CORRIDOR_E=dict(state='PARTIAL',measured=0.,views=0,deferred_reason=None)),
                 connectivity=dict(local_reviewed=['R10B'],stage='PRIMARY',local_second_zone=None),
                 visit_states={},visits=[],rejected=[],progress=50,source_session=None)
    p=dict(kind='MISSION',context={'identity':'fixture'},coverage=coverage_payload(b,mask,s),mission=mission,
           mission_session_ids=[],created_utc='2026-11-10T01:00:00+00:00')
    return b,mask,s,p


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.b,self.mask,self.live,self.payload=fixture();self.context=self.payload['context']
    def tearDown(self):self.tmp.cleanup()
    def session(self,n,execute=True,payload=None,status=None,checkpoint=True):
        name=f'session_20261110_10{n:02d}00_000001'
        d=self.root/'logs/hospital_zone_mission'/name;d.mkdir(parents=True)
        (d/'run_manifest.json').write_bytes(encode(dict(options={'execute':execute})))
        p=copy.deepcopy(payload or self.payload);p['mission']['source_session']=name
        if name not in p['mission_session_ids']:p['mission_session_ids'].append(name)
        if status is None:status=dict(coverage_session=p['coverage']['source_session'],sent=p['mission']['counters']['sent_goals'],
                                     plan_attempts=p['mission']['counters']['plan_attempts'])
        (d/'latest_status.json').write_bytes(encode(status))
        if checkpoint:save_checkpoint(d/'checkpoints_v18',p)
        return d
    def select(self):return a.select_checkpoint(self.root,self.b,self.context,self.live)
    def test_newer_dry_run_is_ignored(self):
        old=self.session(1);self.session(2,False)
        self.assertEqual(self.select()['session'],old.name)
    def test_failed_resume_does_not_shadow_real_execution(self):
        old=self.session(1)
        self.session(2,status=dict(coverage_session='coverage-live',sent=0,resume_v18={'requested':True,'restored':False}),checkpoint=False)
        self.assertEqual(self.select()['session'],old.name)
    def test_newest_execute_keeps_larger_counters(self):
        self.session(1);p=copy.deepcopy(self.payload);p['mission']['counters']['sent_goals']=53
        d=self.session(2,payload=p)
        r=self.select();self.assertEqual(r['session'],d.name);self.assertEqual(r['payload']['mission']['counters']['sent_goals'],53)
    def test_directory_mtime_is_not_authority(self):
        old=self.session(1);new=self.session(2);os.utime(old,(9999999999,9999999999))
        self.assertEqual(self.select()['session'],new.name)
    def test_missing_latest_checkpoint_no_fallback(self):
        self.session(1);self.session(2,checkpoint=False)
        with self.assertRaisesRegex(ValueError,'NO_CHECKPOINT'):self.select()
    def test_corrupt_latest_no_fallback(self):
        self.session(1);d=self.session(2);(d/'checkpoints_v18/LATEST.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'REFUSED'):self.select()
    def test_newest_changed_world_refuses_not_old(self):
        self.session(1);p=copy.deepcopy(self.payload);p['context']={'identity':'changed'};self.session(2,payload=p)
        with self.assertRaisesRegex(ValueError,'CONTEXT_CHANGED'):self.select()
    def test_different_t5_skipped(self):
        old=self.session(1);p=copy.deepcopy(self.payload);p['coverage']['source_session']='other'
        p['coverage']['status_snapshot']['session_id']='other';self.session(2,payload=p)
        self.assertEqual(self.select()['session'],old.name)
    def test_no_matching_t5_refuses(self):
        self.session(1);self.live['session_id']='new-session'
        with self.assertRaisesRegex(ValueError,'NO_EXECUTE_CHECKPOINT'):self.select()
    def test_live_clock_reset_refuses(self):
        self.session(1);self.live['ros_time_s']=1.
        with self.assertRaisesRegex(ValueError,'CLOCK_BEHIND'):self.select()
    def test_live_count_regression_refuses(self):
        self.session(1);self.live['observed_cells']=2
        with self.assertRaisesRegex(ValueError,'COUNT_REGRESSED'):self.select()
    def test_frozen_path_not_mutable_latest(self):
        self.session(1);p=self.select()['path'];self.assertTrue(p.endswith('.json.gz'));self.assertNotIn('LATEST',p)
    def test_latest_status_ahead_of_checkpoint_refuses(self):
        self.session(1,status=dict(coverage_session='coverage-live',sent=51))
        with self.assertRaisesRegex(ValueError,'BEHIND_SESSION_STATUS'):self.select()
    def test_metadata_absent_refuses(self):
        d=self.session(1);(d/'run_manifest.json').unlink()
        with self.assertRaisesRegex(ValueError,'METADATA'):self.select()
    def test_dry_run_only_refuses(self):
        self.session(1,False)
        with self.assertRaisesRegex(ValueError,'NO_EXECUTE'):self.select()
    def test_selection_does_not_modify_source(self):
        d=self.session(1);before={str(p):p.read_bytes() for p in d.rglob('*') if p.is_file()}
        self.select();self.assertEqual(before,{str(p):p.read_bytes() for p in d.rglob('*') if p.is_file()})
    def test_stale_t5_refuses(self):
        self.session(1);self.live['data_health']='STALE'
        with self.assertRaisesRegex(ValueError,'T5_NOT_LIVE'):self.select()
    def test_pointer_traversal_refuses(self):
        d=self.session(1);(d/'checkpoints_v18/LATEST.json').write_text('{"checkpoint":"../bad"}')
        with self.assertRaisesRegex(ValueError,'REFUSED'):self.select()
    def test_recover_legacy_lost_reviews_without_adding_counters(self):
        old=self.session(1)
        p=copy.deepcopy(self.payload);p['mission']['connectivity']['local_reviewed']=[]
        p['mission']['counters']['sent_goals']=53;p['mission_session_ids']=[old.name]
        new=self.session(2,payload=p);selected=self.select()
        merged=a.recover_terminal_history(self.root,selected['payload'],self.b,self.context)
        self.assertIn('R10B',merged['completion_v183']['finished_zones'])
        self.assertEqual(merged['counters']['sent_goals'],53)
        self.assertEqual(merged['records']['R10B']['measured'],1/32)
    def test_do_not_merge_unrelated_session(self):
        self.session(1);p=copy.deepcopy(self.payload);p['mission']['connectivity']['local_reviewed']=[]
        self.session(2,payload=p);selected=self.select()
        merged=a.recover_terminal_history(self.root,selected['payload'],self.b,self.context)
        self.assertNotIn('R10B',merged['completion_v183']['finished_zones'])
    def test_ancestor_counter_regression_refuses(self):
        old=self.session(1);p=copy.deepcopy(self.payload);p['mission_session_ids']=[old.name]
        p['mission']['counters']['plan_attempts']=100;self.session(2,payload=p)
        with self.assertRaisesRegex(ValueError,'COUNTER_EXCEEDS'):a.recover_terminal_history(self.root,self.select()['payload'],self.b,self.context)
    def test_missing_legacy_ancestor_refuses(self):
        p=copy.deepcopy(self.payload);p['mission_session_ids']=['session_20261110_090000_000001'];self.session(2,payload=p)
        with self.assertRaisesRegex(ValueError,'ANCESTOR_CHECKPOINT_MISSING'):a.recover_terminal_history(self.root,self.select()['payload'],self.b,self.context)
    def test_self_contained_v183_does_not_need_deleted_ancestors(self):
        p=copy.deepcopy(self.payload);p['mission_session_ids']=['session_20261110_090000_000001']
        p['mission']['completion_v183']={'lineage_reconciled':True,'finished_zones':['R01','R10B']}
        self.session(2,payload=p)
        q=a.recover_terminal_history(self.root,self.select()['payload'],self.b,self.context)
        self.assertEqual(set(q['completion_v183']['finished_zones']),{'R01','R10B'})


class CompletionAndPoseTests(unittest.TestCase):
    def setUp(self):self.b,self.mask,self.status,self.payload=fixture();self.m=self.payload['mission']
    def node(self):
        recs={z:types.SimpleNamespace(**r) for z,r in copy.deepcopy(self.m['records']).items()}
        for r in recs.values():r.low_gain_count=0
        book=types.SimpleNamespace(records=recs,threshold=.96,active_zone=None,locked_zone=None,outcomes={},audit_requested={},deferred_this_mission={'R10B'})
        book.eligible=lambda:[{'id':z} for z,r in recs.items() if r.state not in ('OBSERVATION_SUFFICIENT','PARTIAL_DEFERRED','THRESHOLD_PENDING')]
        book.is_complete=lambda z:recs[z].state=='OBSERVATION_SUFFICIENT'
        book.all_resolved=lambda:all(r.state=='OBSERVATION_SUFFICIENT' for r in recs.values())
        book.activate=lambda z:setattr(book,'active_zone',z)
        book.mark_entered=lambda z:setattr(book,'marked_entered',z)
        def update(s):
            for r in recs.values():
                if r.state=='EXPLORATION_COMPLETE':r.state='PARTIAL'
            return []
        book.update=update
        runtime=types.SimpleNamespace(local_reviewed=set(),exploration_stage='PRIMARY',primary_zone=None,
              local_second_zone=None,tasks=types.SimpleNamespace(tasks={}),_maybe_begin_local=lambda inp:'OLD_LOCAL_PASS')
        n=types.SimpleNamespace(book=book,connectivity=runtime,zones_by_id={z['id']:z for z in self.b[0]['zones']},event=lambda *a,**k:None)
        rt.install_simple_completion_policy(n)
        a.apply_terminal_history(n,self.m)
        return n
    def test_visited_does_not_mean_finished(self):
        self.m['records']['CORRIDOR_E']['views']=3
        self.assertNotIn('CORRIDOR_E',a.completion_facts(self.m))
    def test_deferred_without_local_review_is_not_finished(self):
        self.m['connectivity']['local_reviewed']=[]
        self.assertNotIn('R10B',a.completion_facts(self.m))
    def test_reviewed_under96_is_finished(self):self.assertIn('R10B',a.completion_facts(self.m))
    def test_96_inclusive(self):
        self.m['records']['CORRIDOR_E']['measured']=.96
        self.assertIn('CORRIDOR_E',a.completion_facts(self.m))
    def test_active_unfinished_second_not_completed(self):
        self.m['connectivity'].update(stage='SECOND',local_second_zone='R10B')
        self.assertNotIn('R10B',a.completion_facts(self.m))
    def test_explicit_terminal_survives_old_partial_state(self):
        self.m['completion_v183']={'finished_zones':['CORRIDOR_E']}
        self.assertIn('CORRIDOR_E',a.completion_facts(self.m))
    def test_no_repeat_after_repeated_coverage_updates(self):
        n=self.node()
        for _ in range(10):n.book.update({})
        self.assertEqual([z['id'] for z in n.book.eligible()],['CORRIDOR_E'])
        self.assertAlmostEqual(n.book.records['R10B'].measured,1/32)
    def test_stayed_pose_selects_remaining_only(self):
        n=self.node();out=a.current_pose_decision(n,self.m,{'pose':(2.,2.,0.)},self.b,None,lambda *a:True)
        self.assertEqual(out['remaining_zones'],['CORRIDOR_E']);self.assertIsNone(out['preferred_zone'])
    def test_manual_two_meters_uses_new_pose(self):
        n=self.node();self.m['pose_snapshot_v183']={'pose':[7.5,2.,0.]}
        d=a.current_pose_decision(n,self.m,{'pose':(9.5,2.,0.)},self.b,None,lambda *a:True)
        self.assertEqual(d['pose_map'],[9.5,2.,0.]);self.assertEqual(d['displacement_from_checkpoint_m'],2.)
        self.assertEqual(d['preferred_zone'],'CORRIDOR_E');self.assertFalse(d['old_pose_targeted'])
    def test_finished_current_room_only_transit_not_goal(self):
        n=self.node();d=a.current_pose_decision(n,self.m,{'pose':(5.5,2.,0.)},self.b,None,lambda *a:True)
        self.assertEqual(d['spatial_zone'],'R10B');self.assertIsNone(d['preferred_zone'])
    def test_unsafe_membership_does_not_prefer_current_zone(self):
        n=self.node();d=a.current_pose_decision(n,self.m,{'pose':(9.5,2.,0.)},self.b,None,lambda *a:False)
        self.assertIsNone(d['preferred_zone'])
    def test_missing_old_pose_not_invented(self):
        d=a.current_pose_decision(self.node(),self.m,{'pose':(9.5,2.,0.)},self.b,None,lambda *a:True)
        self.assertIsNone(d['displacement_from_checkpoint_m'])
    def test_bad_pose_refused(self):
        with self.assertRaisesRegex(ValueError,'POSE'):a.spatial_zone(self.b,(float('nan'),2.,0.))
    def test_outside_grid_no_nearest_zone_guess(self):self.assertIsNone(a.spatial_zone(self.b,(-1.,2.,0.)))
    def test_rotated_grid_lookup(self):
        self.b[1].origin=(10.,20.,math.pi/2)
        self.assertEqual(a.spatial_zone(self.b,(8.,29.5,0.)),'CORRIDOR_E')
    def test_duplicate_scope_refused(self):
        self.m['scope'].append('R01')
        with self.assertRaisesRegex(ValueError,'SCOPE'):a.completion_facts(self.m)
    def test_completion_unknown_zone_refused(self):
        self.m['connectivity']['local_reviewed'].append('OTHER')
        with self.assertRaisesRegex(ValueError,'UNKNOWN_ZONE'):a.completion_facts(self.m)
    def test_threshold_zone_does_not_start_second_pass(self):
        n=self.node();n.connectivity.primary_zone='R01'
        self.assertFalse(n.connectivity._maybe_begin_local({}))
    def test_preserve_in_progress_second_after_move(self):
        n=self.node();n.connectivity.exploration_stage='SECOND';n.connectivity.local_second_zone='CORRIDOR_E'
        d=a.current_pose_decision(n,self.m,{'pose':(2.,2.,0.)},self.b,None,lambda *a:True)
        self.assertEqual(d['preferred_zone'],'CORRIDOR_E')


class ReadinessTests(unittest.TestCase):
    def ready(self):
        state=a.LiveReadiness(0.)
        _,_,s,_=fixture();state.on_clock(100.,.1);state.on_actions([],1.)
        state.on_coverage(s,1.)
        state.on_clock(102.,2.1);s=dict(s,ros_time_s=102.,evidence_scans=11,last_evidence_scan_end_s=101.9)
        state.on_coverage(s,2.1);return state
    def test_pass_with_fresh_scan_idle_clock(self):self.assertTrue(self.ready().check(2.2)[0])
    def test_active_nav_refuses(self):
        for status in [1,2,3]:
            s=self.ready();s.on_actions([status],2.1)
            with self.assertRaisesRegex(ValueError,'STILL_ACTIVE'):s.check(2.2)
    def test_terminal_nav_allows(self):
        s=self.ready();s.on_actions([4,5,6],2.1);self.assertTrue(s.check(2.2)[0])
    def test_no_action_status_no_go(self):
        s=self.ready();s.action=None;self.assertEqual(s.check(2.2)[1],'WAIT_NAV2_GOAL_STATUS')
    def test_unknown_goal_status_no_go(self):
        s=self.ready();s.action=[0]
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):s.check(2.2)
    def test_two_t5_refuses(self):
        with self.assertRaisesRegex(ValueError,'MULTIPLE_T5'):self.ready().check(2.2,2,1)
    def test_two_nav2_refuses(self):
        with self.assertRaisesRegex(ValueError,'MULTIPLE_NAVIGATION'):self.ready().check(2.2,1,2)
    def test_paused_clock_no_go(self):self.assertFalse(self.ready().check(8.)[0])
    def test_clock_reset_refuses(self):
        s=self.ready();s.on_clock(1.,2.1)
        with self.assertRaisesRegex(ValueError,'CLOCK_RESET'):s.check(2.2)
    def test_latched_only_no_go(self):
        s=self.ready();s.evidence_advanced=False;self.assertFalse(s.check(2.2)[0])
    def test_t5_fault_no_go(self):
        s=self.ready();s.status['fault']='MAP_CHANGED';self.assertFalse(s.check(2.2)[0])
    def test_t5_session_changed_refuses(self):
        s=self.ready();s.on_coverage(dict(s.status,session_id='another'),2.1)
        with self.assertRaisesRegex(ValueError,'SESSION_CHANGED'):s.check(2.2)
    def test_future_status_not_fresh(self):
        s=self.ready();s.status['ros_time_s']=104.;self.assertFalse(s.check(2.2)[0])
    def test_missing_t5_no_go(self):self.assertFalse(self.ready().check(2.2,0,1)[0])


class ControllerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.b,self.mask,self.status,self.payload=fixture()
        self.path,_=save_checkpoint(self.root/'input',self.payload)
        self.args=types.SimpleNamespace(resume_from=self.path,resume_confirm_scene=True,
                                       namespace='/j100_0001',execute=True)
        self.events=[];self.safe=True;self.stopped=True;self.foreign=[];self.error=None
        self.inp=dict(status=self.status,observed=self.mask,pose=(9.5,2.,0.),cost={'key':'live'},epoch=1)
        case=CompletionAndPoseTests();case.setUp();n=case.node()
        # Simulate pristine process before restoration, not already-loaded history.
        n.connectivity.local_reviewed=set();n._v183_finished_zones=set()
        n.connectivity.tasks=None
        n.book.records={z:types.SimpleNamespace(state='UNOBSERVED',measured=0.,views=0,low_gain_count=0,deferred_reason=None)
                        for z in n.book.records}
        # Original closure fixture depends on its own record dict; replace dynamic selectors.
        n.book.eligible=lambda:[{'id':z} for z,r in n.book.records.items() if r.state != rt.EXPLORATION_COMPLETE and r.state not in ('OBSERVATION_SUFFICIENT','PARTIAL_DEFERRED','THRESHOLD_PENDING') and z not in n._v183_finished_zones]
        n.book.visit_state={z:types.SimpleNamespace(**{k:False if k=='entered' else 0 for k in rt.VISIT_FIELDS}) for z in n.zones_by_id}
        n.book.visits=[];n.book.rejected=[];n.book.progress=0
        n.book.deferral_history={z:[] for z in n.zones_by_id};n.book.nav_abort_counts={z:0 for z in n.zones_by_id}
        n.book.outcomes={}
        for k in rt.NODE_COUNTERS:setattr(n,k,0)
        for k in ('nav_uuid','nav_handle','nav_send','plan_handle','plan_send','plan_result'):setattr(n,k,None)
        n.stop_requested=None;n.fault=None;n.last_clock=None;n.clock_advanced_wall=0.
        n.ros_s=lambda:100.;n.inputs=lambda:(self.error,self.inp if self.error is None else None)
        n.foreign_active=lambda:self.foreign;n.local_start_clear=lambda inp:self.safe
        n.stop_evidence=types.SimpleNamespace(ready=lambda *x:self.stopped)
        n.event=lambda event,**kw:self.events.append((event,kw));n.info=lambda *x:None
        n.request_stop=lambda reason:setattr(n,'stop_requested',reason)
        n.path_blockage=None;n.mission_guard=None;n._v183_action_seen=True;n._v183_last_pose=None
        n.matched=dict(status=self.status,observed=self.mask);n.done=False;n.current=None
        self.n=n
        with patch.object(rt,'root_path',return_value=self.root),patch.object(rt,'project_context',return_value=self.payload['context']):
            self.c=rt.MissionCheckpoint(n,self.args,self.b,None,self.root/'output')
        n._dn012_v18=self.c
        self.zpatch=patch.dict(sys.modules,{'zone_planner':types.SimpleNamespace(is_inside=lambda *x:True)})
        self.zpatch.start()
    def tearDown(self):self.zpatch.stop();self.tmp.cleanup()
    def test_actual_restore_gate_reaches_current_pose(self):
        self.assertTrue(self.c.before_poll());self.assertTrue(self.c.restored)
        self.assertEqual(self.n.sent_goals,50);self.assertEqual(self.n.succeeded_goals,43)
        self.assertIsNone(self.n.nav_uuid);self.assertIsNone(self.n.current)
        self.assertEqual(self.c.pose_decision['preferred_zone'],'CORRIDOR_E')
        self.assertNotIn('R10B',[z['id'] for z in self.n.book.eligible()])
    def test_stopped_gate_prevents_restore(self):
        self.stopped=False;self.assertFalse(self.c.before_poll());self.assertEqual(self.n.sent_goals,0)
    def test_footprint_gate_prevents_restore(self):
        self.safe=False;self.assertFalse(self.c.before_poll());self.assertFalse(self.c.restored)
    def test_foreign_goal_prevents_restore(self):
        self.foreign=['other'];self.assertFalse(self.c.before_poll());self.assertEqual(self.n.sent_goals,0)
    def test_first_action_status_required(self):
        self.n._v183_action_seen=False;self.assertFalse(self.c.before_poll())
    def test_missing_inputs_keep_no_goal(self):
        self.error='WAIT_TF';self.assertFalse(self.c.before_poll());self.assertIsNone(self.n.nav_uuid)
    def test_timeout_no_movement(self):
        self.error='WAIT_TF';self.c.resume_started_wall=-100.
        with self.assertRaisesRegex(ValueError,'TIMEOUT'):self.c.before_poll()
    def test_wrong_t5_prevents_restore(self):
        self.inp['status']=dict(self.status,session_id='different')
        with self.assertRaisesRegex(ValueError,'T5_NOT_RESTORED'):self.c.before_poll()
    def test_lost_cells_prevents_restore(self):
        self.inp['observed']=np.zeros_like(self.mask)
        with self.assertRaisesRegex(ValueError,'LOST_SAVED_CELLS'):self.c.before_poll()
    def test_restore_once_not_double_goal_counts(self):
        self.c.before_poll();self.n.sent_goals+=1;self.c.before_poll()
        self.assertEqual(self.n.sent_goals,51)
    def test_current_pose_log_no_invented_manual_claim(self):
        self.c.before_poll();rec=next(r for e,r in self.events if e=='RESUME_CURRENT_POSE_V183')
        self.assertEqual(rec['manual_intervention'],'NOT_DECLARED')
        self.assertFalse(rec['old_goal_replayed'])
    def test_snapshot_keeps_completion_and_exact_counters(self):
        self.c.before_poll()
        # Capture does not need old action handles or a historical robot pose.
        m=rt.capture_mission(self.n,dict(execute=True,source_session_v18='session_20261110_110000_000001'))
        self.assertEqual(m['counters']['sent_goals'],50)
        self.assertEqual(set(m['completion_v183']['finished_zones']),{'R01','R10B'})
        self.assertTrue(m['completion_v183']['lineage_reconciled'])
    def test_original_checkpoint_bytes_unchanged_after_restore(self):
        before=self.path.read_bytes();self.c.before_poll();self.assertEqual(before,self.path.read_bytes())



class SendBoundaryTests(unittest.TestCase):
    def create(self):
        case=CompletionAndPoseTests();case.setUp()
        base_node=case.node()
        class Base:
            def __init__(self):
                self.__dict__.update(base_node.__dict__)
                self.current=None;self.sent=[];self.logged=[]
            def info(self,*args):pass
            def event(self,event,**kw):self.logged.append((event,kw))
            def nav_request(self,inp):self.sent.append(self.current['zone_id'])
            def transition(self,state):self.state=state
            def finish(self,state,code=0):self.finish_value=(state,code)
        controller=types.SimpleNamespace(event=lambda *a:None)
        with patch.dict(os.environ,{'DN012_V18_ENABLE':'1'}),patch.object(rt,'MissionCheckpoint',return_value=controller):
            n=rt.wrap_mission(Base,types.SimpleNamespace(),case.b,None,Path('/synthetic'))()
        return n,case
    def test_completed_zone_never_gets_navigation_send(self):
        n,_=self.create();n.current={'zone_id':'R10B','second_pass':True}
        n.nav_request({});self.assertEqual(n.sent,[]);self.assertIsNone(n.current)
    def test_unfinished_zone_still_uses_original_send(self):
        n,_=self.create();n.current={'zone_id':'CORRIDOR_E'}
        n.nav_request({});self.assertEqual(n.sent,['CORRIDOR_E'])
    def test_new96_drops_previously_planned_second_pass(self):
        n,_=self.create();n.book.records['CORRIDOR_E'].measured=.96
        n.current={'zone_id':'CORRIDOR_E','second_pass':True}
        n.nav_request({});self.assertEqual(n.sent,[])
    def test_fault_finish_not_renamed_success(self):
        n,_=self.create();n.finish('USER_INTERRUPT',2)
        self.assertEqual(n.finish_value,('USER_INTERRUPT',2))
    def test_finished_second_pass_waits_for_worker(self):
        n,case=self.create();calls=[]
        rt0=n.connectivity
        rt0.selection_hook=lambda inp:calls.append('old')
        rt0._end_local=lambda inp:calls.append('end')
        rt0.second_future=types.SimpleNamespace(done=lambda:False)
        rt0.exploration_stage='SECOND';rt0.local_second_zone='CORRIDOR_E'
        n.book.records['CORRIDOR_E'].measured=.96
        # Rebuild only this synthetic wrapper to include the test's new hook.
        n.book._dn012_simple_completion_installed=False
        rt.install_simple_completion_policy(n)
        self.assertTrue(rt0.selection_hook({}));self.assertEqual(calls,[])
        rt0.second_future=types.SimpleNamespace(done=lambda:True,result=lambda:None)
        rt0.selection_hook({});self.assertIn('end',calls)


if __name__=='__main__':unittest.main(verbosity=2)
