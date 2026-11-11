"""Offline policy/lattice/mock-ROS regression; not a Gazebo/DDS test."""
from pathlib import Path
import copy
import importlib.util
import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ZT=ROOT/'scripts/exploration/zone_exploration/tests'
SRC=ROOT/'scripts/exploration/zone_exploration/src'
for p in (ZT,SRC,ROOT/'scripts/exploration/navigation/return_home'):
    sys.path.insert(0,str(p))
import fixtures as fx
import fake_ros
fake_ros.install()
os.environ['ZB_PROJECT_ROOT']=str(ROOT)
from planning_core import Settings
from zone_policy import ConnectivityMission,limited_certificate_ok
from connectivity_audit import (AuditSettings,build_catalog,assess_catalog,useful_untried,
                                sensor_signature,optimistic_reachability,residual_counts)
from return_home_policy import evaluate_return_eligibility
import zone_node


def certificate():
    return dict(catalog_complete=True,inputs_fresh=True,context_matches=True,
        connectivity_verified=True,all_potential_information_accounted=True,
        positions=2,probes_succeeded=2,probes_failed=0,unrepresented_components=0,
        unknown_cells=0,unobserved_free_cells=0,remaining_gain_views=0,
        cost_key='grid1',bundle_signature='test-signature')


def view(zid='R_A',kind='OBSERVE',ident='x'):
    return dict(zone_id=zid,goal_kind=kind,id=ident,x=5.,y=2.,yaw=0.)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.b=fx.bundle()
        self.book=ConnectivityMission(self.b[0]['zones'],threshold=.96,max_views=4)
        self.book.context('grid1',1.)
    def test_two_low_gains_request_audit_keep_lock(self):
        for i in range(2):self.book.record_visit(view(ident=str(i)),.94,.9401,'SUCCEEDED',None)
        self.assertEqual(self.book.active_zone,'R_A')
        self.assertIn('R_A',self.book.audit_requested)
        self.assertNotEqual(self.book.records['R_A'].state,'PARTIAL_DEFERRED')
    def test_four_views_are_soft_not_complete(self):
        for i in range(4):self.book.record_visit(view(ident=str(i)),.8,.81,'SUCCEEDED',None)
        self.assertIn('R_A',self.book.audit_requested)
        self.assertFalse(self.book.is_complete('R_A'))
    def test_entry_zero_gain_does_not_trigger_low_gain(self):
        self.book.record_visit(view(kind='ENTRY'),0,0,'SUCCEEDED',None)
        self.assertEqual(self.book.visit_state['R_A'].low_gain,0)
        self.assertNotIn('R_A',self.book.audit_requested)
    def test_view_budget_is_partial_not_completion(self):
        b=ConnectivityMission(self.b[0]['zones'],threshold=.96,hard_zone_goals=2)
        for i in range(2):b.record_visit(view(ident=str(i)),.8,.81,'SUCCEEDED',None)
        self.assertEqual(b.records['R_A'].state,'PARTIAL_DEFERRED')
        self.assertFalse(b.outcomes['R_A']['normal_termination'])
    def test_no_time_or_progress_revisit(self):
        self.book.finish_partial('R_A','NO_EVIDENCE')
        self.book.context('other',999);self.book.progress=99
        self.assertEqual(self.book.reopen_changed(),[])
    def test_unknown_cannot_finish_limited(self):
        e=certificate();e['unknown_cells']=1
        with self.assertRaises(ValueError):self.book.finish_limited('R_A',e)
    def test_missing_probe_cannot_finish_limited(self):
        e=certificate();e['probes_succeeded']=1
        with self.assertRaises(ValueError):self.book.finish_limited('R_A',e)
    def test_low_gain_alone_cannot_finish_limited(self):
        with self.assertRaises(ValueError):self.book.finish_limited('R_A',{'low_gain_count':2})
    def test_map_changed_cannot_finish_limited(self):
        e=certificate();e['cost_key']='old'
        with self.assertRaises(ValueError):self.book.finish_limited('R_A',e)
    def test_valid_certificate_preserves_raw_coverage(self):
        self.book.records['R_A'].measured=.91
        self.book.finish_limited('R_A',certificate())
        self.assertEqual(self.book.records['R_A'].state,'OBSERVATION_LIMITED')
        self.assertEqual(self.book.records['R_A'].measured,.91)
        self.assertFalse(self.book.is_complete('R_A'))
        self.assertNotIn('R_A',[z['id'] for z in self.book.eligible()])
    def test_incomplete_search_is_not_certificate(self):
        e=certificate();e['catalog_complete']=False
        self.assertFalse(limited_certificate_ok(e))
    def test_numeric_missing_values_fail_closed(self):
        for name,val in [('positions',0),('probes_succeeded',True),('unobserved_free_cells',1),('remaining_gain_views',1)]:
            e=certificate();e[name]=val
            self.assertFalse(limited_certificate_ok(e))
    def test_certificate_survives_same_snapshot_update(self):
        s=fx.status(self.b)
        self.book.update(s);self.book.finish_limited('R_A',certificate())
        self.book.update(s)
        self.assertEqual(self.book.records['R_A'].state,'OBSERVATION_LIMITED')
    def test_changed_map_invalidates_limited_evidence(self):
        s=fx.status(self.b);self.book.update(s);self.book.finish_limited('R_A',certificate())
        self.book.context('changed',2);self.book.update(s)
        self.assertNotEqual(self.book.records['R_A'].state,'OBSERVATION_LIMITED')
        self.assertIn('R_A',self.book.audit_requested)


class LatticeTests(unittest.TestCase):
    def setUp(self):
        self.b=fx.bundle();self.s=fx.snapshot(self.b);self.z=self.b[0]['zones'][1]
        self.settings=AuditSettings(spacing_m=.8,headings=4,max_positions=600)
        self.cfg=Settings(threshold=.96)
    def visibility(self,g,free,target,*a):
        return np.flatnonzero(free & target).astype(np.int32)
    def build(self):
        return build_catalog(self.b,self.cfg,self.settings,self.s,self.z,self.visibility,threading.Event())
    def allprobes(self,c):
        return {p['position_id']:dict(success=True) for p in c['positions']}
    def test_finite_complete_catalog(self):
        c=self.build();self.assertTrue(c['catalog_complete']);self.assertGreater(len(c['positions']),1)
    def test_one_cell_gain_not_discarded(self):
        self.s['observed'][:]=True;self.s['observed'][25,60]=False
        c=self.build();v,n=useful_untried(c,self.s['observed'],[],self.b[1],self.cfg)
        self.assertGreater(len(v),0);self.assertEqual(v[0]['new_cells'],1)
    def test_history_filters_only_tried_pose_and_heading(self):
        c=self.build();v,_=useful_untried(c,self.s['observed'],[],self.b[1],self.cfg)
        other,_=useful_untried(c,self.s['observed'],[v[0]],self.b[1],self.cfg)
        self.assertNotIn(v[0]['id'],[x['id'] for x in other]);self.assertTrue(other)
    def test_all_tried_positive_prediction_is_unresolved(self):
        c=self.build();v,_=useful_untried(c,self.s['observed'],c['views'],self.b[1],self.cfg)
        self.assertEqual(v,[])
        verdict,_=assess_catalog(c,self.allprobes(c),self.s,self.b,self.z)
        self.assertEqual(verdict,'OBSERVATION_UNRESOLVED')
    def test_scattered_occupied_cells_free_observed_can_be_limited(self):
        self.s['cost']['values'][23:30,62:70]=100
        self.s['observed']=(self.b[2]==self.z['mask_id'])&(self.s['cost']['values']<100)
        c=self.build();verdict,e=assess_catalog(c,self.allprobes(c),self.s,self.b,self.z)
        self.assertEqual(verdict,'NORMAL_LIMITED');self.assertTrue(limited_certificate_ok(e))
        self.assertGreater(e['unobserved_occupied_cells'],0)
    def test_local_split_external_detour_not_discarded(self):
        self.s['cost']['values'][15:40,65:70]=100
        self.s['observed']=(self.b[2]==self.z['mask_id'])&(self.s['cost']['values']<100)
        c=self.build();self.assertEqual(c['components'],2)
        self.assertTrue(all(c['optimistic_connected'].values()))
        self.assertEqual(assess_catalog(c,self.allprobes(c),self.s,self.b,self.z)[0],'NORMAL_LIMITED')
    def test_occupied_global_cut_with_nav2_failure_is_candidate_only(self):
        self.s['cost']['values'][:,65:70]=100
        c=self.build();p=self.allprobes(c)
        for pos in c['positions']:
            if not c['optimistic_connected'][pos['position_id']]:
                p[pos['position_id']]={'success':False,'reason':'NAVFN_ABORTED'}
        self.assertEqual(assess_catalog(c,p,self.s,self.b,self.z)[0],'MAP_BLOCKED_CANDIDATE')
    def test_path_failure_alone_is_not_blocked_proof(self):
        c=self.build();p=self.allprobes(c);p[c['positions'][0]['position_id']]={'success':False,'reason':'NAVFN_ABORTED'}
        self.assertEqual(assess_catalog(c,p,self.s,self.b,self.z)[0],'CONNECTIVITY_UNVERIFIED')
    def test_unknown_never_normal_even_if_routes_exist(self):
        self.s['cost']['values'][25,60]=-1;self.s['observed'][:]=True
        c=self.build();v,e=assess_catalog(c,self.allprobes(c),self.s,self.b,self.z)
        self.assertNotEqual(v,'NORMAL_LIMITED');self.assertEqual(e['unknown_cells'],1)
    def test_inscribed_cost_not_misclassified_as_confirmed_occupied(self):
        cost=self.s['cost']['values'];cost[25,60]=99
        seen=np.ones_like(self.s['observed']);seen[25,60]=False
        r=residual_counts(self.b[2],self.b[3],cost,seen,self.z)
        self.assertEqual(r['unobserved_free_cells'],1);self.assertEqual(r['unobserved_occupied_cells'],0)
    def test_budget_limit_never_returns_partial_complete_catalog(self):
        self.settings=replace(self.settings,max_positions=1)
        with self.assertRaisesRegex(TimeoutError,'POSITION_BUDGET'):self.build()
    def test_cancel_build(self):
        cancel=threading.Event();cancel.set()
        with self.assertRaisesRegex(RuntimeError,'CANCELLED'):
            build_catalog(self.b,self.cfg,self.settings,self.s,self.z,self.visibility,cancel)
    def test_missing_probe_blocks_completion(self):
        self.s['observed'][:]=True;c=self.build()
        self.assertNotEqual(assess_catalog(c,{},self.s,self.b,self.z)[0],'NORMAL_LIMITED')
    def test_cost_context_change_blocks_completion(self):
        self.s['observed'][:]=True;c=self.build();self.s['cost']['key']='new'
        self.assertNotEqual(assess_catalog(c,self.allprobes(c),self.s,self.b,self.z)[0],'NORMAL_LIMITED')
    def test_no_safe_positions_never_proves_completion(self):
        self.s['cost']['values'][self.b[2]==self.z['mask_id']]=100
        c=self.build();self.assertEqual(c['positions'],[])
        self.assertNotEqual(assess_catalog(c,{},self.s,self.b,self.z)[0],'NORMAL_LIMITED')


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.b=fx.bundle();self.s=fx.snapshot(self.b)
        self.args=zone_node.parser().parse_args(['--connectivity-aware','--visibility-backend','python','--threshold','.96'])
        self.node=zone_node.make_node_type(self.args,self.b,Settings(threshold=.96),Path(self.tmp.name))()
        self.node.inputs=lambda:(None,self.s)
        self.node.stop_evidence.ready=lambda *a:True
        self.node.local_start_clear=lambda *a:True
        self.node.book.update(self.s['status']);self.node.book.context('grid1',time.monotonic())
        self.node.boot_wall-=5
        c=self.node.zone_planner.entry_batch(self.s,[self.b[0]['zones'][1]],[])['candidates'][0]
        self.c=dict(c,connectivity_probe=True,audit_position_id='R_A:probe')
        self.node.book.audit_requested['R_A']='test'
        self.rt=self.node.connectivity;self.rt.start_zone('R_A')
        self.rt.catalog=dict(cost_key='grid1',epoch=1,sensor_signature=sensor_signature(self.s['sensor']),
            start_pose=list(self.s['pose']),positions=[{'position_id':'R_A:probe'}])
    def tearDown(self):
        self.node.close();self.tmp.cleanup()
    def reply(self,points=None,status=4):
        self.node.plan_request(self.c,self.s);self.node.poll_actions(self.s)
        if points is None:points=[self.s['pose'][:2],(self.c['x'],self.c['y'])]
        poses=[]
        for x,y in points:
            p=fake_ros.PoseStamped();p.header.frame_id='map';p.pose.position.x=x;p.pose.position.y=y;poses.append(p)
        h=self.node.plan_client.handles[-1]
        h.result_future.set(NS(status=status,result=NS(path=NS(header=NS(frame_id='map'),poses=poses))))
        self.node.poll_actions(self.s)
    def test_probe_cannot_send_navigation(self):
        self.reply();self.assertEqual(self.node.state,'SELECTING')
        self.assertEqual(self.node.nav_client.sent,[]);self.assertIsNone(self.node.current)
        self.assertTrue(self.rt.probes['R_A:probe']['success'])
    def test_probe_allowed_outside_zone_path(self):
        self.reply([self.s['pose'][:2],(3.,1.),(6.,1.),(self.c['x'],self.c['y'])])
        self.assertTrue(self.rt.probes['R_A:probe']['path_left_target_zone'])
        self.assertEqual(self.node.book.active_zone,'R_A')
    def test_probe_is_not_navigation_plan_attempt_budget(self):
        self.reply();self.assertEqual(self.node.plan_attempts,0)
    def test_probe_abort_not_success(self):
        self.reply(status=6);self.assertFalse(self.rt.probes['R_A:probe']['success'])
        self.assertFalse(self.node.done)
    def test_probe_wrong_endpoint_not_success(self):
        self.reply([self.s['pose'][:2],(self.c['x']+.5,self.c['y'])])
        self.assertFalse(self.rt.probes['R_A:probe']['success'])
    def test_probe_stale_context_discarded(self):
        self.node.plan_request(self.c,self.s);self.node.poll_actions(self.s)
        self.s['cost']['key']='new'
        self.node.plan_client.handles[-1].result_future.set(NS(status=4,result=NS(path=None)))
        self.node.poll_actions(self.s)
        self.assertEqual(self.rt.probes,{})
        self.assertEqual(self.node.book.rejected,[])
    def test_probe_cannot_bypass_direct_nav_method(self):
        self.args.execute=True;self.args.supervised=True;self.node.current=self.c
        self.node.validated_plan_mode='NAVFN_GOAL_VALIDATED'
        with self.assertRaisesRegex(RuntimeError,'PROBE_MUST_NEVER_MOVE'):
            self.node.nav_request(self.s)
    def test_selected_zone_not_stolen_by_pending_other_zone(self):
        self.rt.zid=None;self.rt.clear_round();self.node.book.audit_requested={'R_B':'test'}
        self.node.book.activate('R_A')
        self.assertFalse(self.rt.step(self.s));self.assertEqual(self.node.book.active_zone,'R_A')
    def test_audit_time_budget_partial_not_normal(self):
        self.rt.started=time.monotonic()-1000
        self.rt.step(self.s)
        self.assertEqual(self.node.book.records['R_A'].state,'PARTIAL_DEFERRED')
        self.assertFalse(self.node.book.outcomes['R_A']['normal_termination'])
    def test_budget_does_not_move(self):
        self.rt.started=time.monotonic()-1000;self.rt.step(self.s)
        self.assertEqual(self.node.nav_client.sent,[])
    def test_novel_small_audit_goal_survives_refresh(self):
        c=dict(self.c,goal_kind='OBSERVE',audit_observation=True,visible_indices=np.array([20],dtype=np.int32),
               path_est_m=1.,new_indices=np.array([20]),new_cells=1,new_area_m2=.01)
        self.assertIsNotNone(zone_node.refresh_for_policy(c,self.s['observed'],self.b[1],Settings()))
    def test_local_start_failure_blocks_audit(self):
        self.node.local_start_clear=lambda *a:False
        self.rt.catalog.update(views=[],unrepresented_components=0)
        self.rt.step(self.s)
        self.assertEqual(self.node.stop_requested,'LOCAL_START_NOT_CLEAR_OR_STALE')
        self.assertEqual(self.node.nav_client.sent,[])
    def test_zero_probe_budget_cannot_finish_limited(self):
        self.rt.catalog.update(views=[],unrepresented_components=0)
        self.rt.probe_count=self.rt.settings.max_probes
        self.rt.step(self.s)
        self.assertEqual(self.node.book.records['R_A'].state,'PARTIAL_DEFERRED')
        self.assertEqual(self.node.book.records['R_A'].deferred_reason,'AUDIT_PROBE_BUDGET_UNVERIFIED')
    def test_e2e_audit_detour_limited_no_motion(self):
        self.s['cost']['values'][15:40,65:70]=100
        self.s['observed']=(self.b[2]==2)&(self.s['cost']['values']<100)
        self.s['status']=fx.status(self.b,self.s['observed'],seq=3,t=10.)
        self.node.book.update(self.s['status'])
        self.rt.catalog=None;self.rt.pending_context=None
        self.rt.settings=replace(self.rt.settings,spacing_m=.8,headings=4)
        deadline=time.monotonic()+8
        while 'R_A' in self.node.book.audit_requested and time.monotonic()<deadline:
            self.rt.step(self.s)
            if self.node.plan_send is not None:
                self.node.poll_actions(self.s)
                c=self.node.current
                points=[self.s['pose'][:2],(3.,1.),(c['x'],1.),(c['x'],c['y'])]
                poses=[]
                for x,y in points:
                    p=fake_ros.PoseStamped();p.header.frame_id='map';p.pose.position.x=x;p.pose.position.y=y;poses.append(p)
                self.node.plan_client.handles[-1].result_future.set(NS(status=4,result=NS(path=NS(header=NS(frame_id='map'),poses=poses))))
                self.node.poll_actions(self.s)
            time.sleep(.001)
        self.assertEqual(self.node.book.records['R_A'].state,'OBSERVATION_LIMITED')
        self.assertEqual(self.node.nav_client.sent,[])
        self.assertFalse(self.node.book.is_complete('R_A'))
        self.assertTrue(list((Path(self.tmp.name)/'connectivity_evidence').glob('*.json')))
    def test_actual_nav_abort_keeps_same_zone_only_after_stop_and_new_coverage(self):
        self.args.execute=True;self.args.supervised=True;self.args.continue_on_nav_abort=True
        self.rt.zid=None;self.rt.catalog=None;self.node.book.audit_requested={}
        self.node.current=dict(self.c,goal_kind='ENTRY',connectivity_probe=False)
        self.node.nav_terminal=6;self.node.nav_uuid='owned';self.node.post_result_ros=9.
        self.node.post_result_wall=time.monotonic();self.node.post_result_scans=1
        self.s['status']['evidence_scans']=2;self.s['status']['last_evidence_scan_end_s']=10.
        self.node.after_result(self.s)
        self.assertEqual(self.node.book.active_zone,'R_A')
        self.assertEqual(self.node.aborted_goals,1)
        self.assertIn('R_A',self.node.book.audit_requested)
        self.assertFalse(self.node.done)
    def test_actual_nav_abort_not_retried_without_stop(self):
        self.args.execute=True;self.args.continue_on_nav_abort=True
        self.node.stop_evidence.ready=lambda *a:False
        self.node.current=dict(self.c,connectivity_probe=False)
        self.node.nav_terminal=6;self.node.post_result_ros=9.;self.node.post_result_wall=time.monotonic()
        self.node.after_result(self.s)
        self.assertEqual(self.node.aborted_goals,0)
        self.assertEqual(self.node.nav_client.sent,[])
    def test_actual_nav_abort_bounded_after_three(self):
        self.args.execute=True;self.args.supervised=True;self.args.continue_on_nav_abort=True
        self.node.book.nav_abort_counts['R_A']=2
        self.node.current=dict(self.c,connectivity_probe=False)
        self.node.nav_terminal=6;self.node.nav_uuid='owned';self.node.post_result_ros=9.
        self.node.post_result_wall=time.monotonic();self.node.post_result_scans=1
        self.s['status']['evidence_scans']=2;self.s['status']['last_evidence_scan_end_s']=10.
        self.node.after_result(self.s)
        self.assertEqual(self.node.book.records['R_A'].state,'PARTIAL_DEFERRED')
        self.assertEqual(self.node.book.records['R_A'].deferred_reason,'NAV_ABORT_BUDGET_UNVERIFIED')

    def test_original_small_goal_filter_unchanged(self):
        c=dict(self.c,goal_kind='OBSERVE',visible_indices=np.array([20],dtype=np.int32),
               path_est_m=1.,new_indices=np.array([20]),new_cells=1,new_area_m2=.01)
        self.assertIsNone(zone_node.refresh_for_policy(c,self.s['observed'],self.b[1],Settings()))


def return_fixture():
    a={'options':{'connectivity_aware':True,'execute':True,'threshold':.96,'zone_ids':None},
       'zone_bundle_signature':'test-signature'}
    b={'connectivity_aware':True,'execute':True,'threshold':.96,'state':'ALL_REQUESTED_ZONES_OBSERVED',
       'sent':27,'nav_succeeded':27,'nav_aborted':0,'coverage_canceled':0,
       'zone_bundle_signature':'test-signature','owned_goal_id':None,'stop_reason':None,
       'zones':{f'Z{i}':{'state':'OBSERVATION_SUFFICIENT','measured_percent':97.} for i in range(27)}}
    return a,b


class ReturnPolicyTests(unittest.TestCase):
    def limited(self):
        a,b=return_fixture();b['state']='ALL_ZONES_ASSESSED_WITH_LIMITS'
        b['zones']['Z0']={'state':'OBSERVATION_LIMITED','measured_percent':89.,
                         'assessment':{'normal_termination':True,'evidence':certificate()}}
        return a,b
    def test_full_return(self):
        a,b=return_fixture();self.assertEqual(evaluate_return_eligibility(a,b)['exploration'],'FULL')
    def test_limited_return_separate_from_coverage(self):
        a,b=self.limited();r=evaluate_return_eligibility(a,b)
        self.assertTrue(r['eligible']);self.assertEqual(r['exploration'],'LIMITED')
        self.assertEqual(b['zones']['Z0']['measured_percent'],89.)
    def test_missing_certificate_no_return(self):
        a,b=self.limited();b['zones']['Z0']['assessment']={}
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])
    def test_unknown_certificate_no_return(self):
        a,b=self.limited();b['zones']['Z0']['assessment']['evidence']['unknown_cells']=1
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])
    def test_incomplete_probe_evidence_no_return(self):
        a,b=self.limited();b['zones']['Z0']['assessment']['evidence']['probes_succeeded']=1
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])
    def test_partial_return_not_normal(self):
        a,b=return_fixture();b['state']='PARTIAL_DEFERRED_REMAIN'
        b['zones']['Z0']={'state':'PARTIAL_DEFERRED','measured_percent':94.,'deferred_reason':'OBSERVATION_UNRESOLVED_PARTIAL'}
        self.assertEqual(evaluate_return_eligibility(a,b)['exploration'],'PARTIAL')
    def test_system_error_no_return(self):
        a,b=return_fixture();b['state']='DATA_LOSS_STALE_GLOBAL_COSTMAP'
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])
    def test_owned_goal_no_return(self):
        a,b=return_fixture();b['owned_goal_id']='active'
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])
    def test_nan_threshold_no_return(self):
        a,b=return_fixture();b['threshold']=float('nan')
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])
    def test_unapproved_abort_no_return(self):
        a,b=return_fixture();b['nav_succeeded']=26;b['nav_aborted']=1
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])
    def test_approved_bounded_abort_counts_preserved(self):
        a,b=return_fixture();a['options']['continue_on_nav_abort']=True
        b['nav_succeeded']=26;b['nav_aborted']=1
        self.assertTrue(evaluate_return_eligibility(a,b)['eligible'])
    def test_terminal_state_mismatch_no_return(self):
        a,b=self.limited();b['state']='ALL_REQUESTED_ZONES_OBSERVED'
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])

if __name__=='__main__':unittest.main()
