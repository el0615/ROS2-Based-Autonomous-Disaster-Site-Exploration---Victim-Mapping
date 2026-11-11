"""v3 synthetic geometry + fake transport regressions. No physical robot test."""
from __future__ import annotations
import copy, math, os, sys, tempfile, time, unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT/'scripts/exploration/zone_exploration/tests',ROOT/'scripts/exploration/zone_exploration/src',ROOT/'scripts/exploration/navigation/return_home'):
    sys.path.insert(0,str(p))
import fixtures as fx
import fake_ros
fake_ros.install()
os.environ['ZB_PROJECT_ROOT']=str(ROOT)
from planning_core import Settings
from path_blockage import *
import zone_node
from return_home_policy import evaluate_return_eligibility

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.b=fx.bundle();self.g=self.b[1];self.cost=np.zeros_like(self.b[2]);self.cfg=Settings()
        self.path=[(2.,2.),(6.,2.)];self.pose=(2.,2.,0.);self.cell=(30,20)
        self.cost[20,30]=100;self.hits=np.array([self.g.cell_to_world(30,20)])
    def witness(self,**kwargs):
        a=dict(points=self.path,pose=self.pose,geom=self.g,prior_free=self.b[3],cost=self.cost,hits=self.hits,cfg=self.cfg)
        a.update(kwargs);return path_obstacle_witness(**a)
    def test_new_lethal_obstacle_detected(self):
        self.assertEqual(self.witness()['cost'],100)
    def test_inflation_99_not_real_obstacle(self):
        self.cost[20,30]=99;self.assertIsNone(self.witness())
    def test_unknown_not_obstacle_evidence(self):
        self.cost[20,30]=-1;self.assertIsNone(self.witness())
    def test_existing_prior_wall_not_new_obstacle(self):
        self.b[3][20,30]=False;self.assertIsNone(self.witness())
    def test_no_scan_confirmation_no_witness(self):
        self.assertIsNone(self.witness(hits=np.array([[8.,7.]])))
    def test_empty_scan_no_witness(self):
        self.assertIsNone(self.witness(hits=np.empty((0,2))))
    def test_obstacle_behind_robot_ignored(self):
        self.assertIsNone(self.witness(pose=(4.,2.,0.)))
    def test_body_edge_not_just_centreline(self):
        self.cost[:]=0;self.cost[22,30]=100
        self.assertIsNotNone(self.witness(hits=np.array([self.g.cell_to_world(30,22)])))
    def test_obstacle_beyond_body_ignored(self):
        self.cost[:]=0;self.cost[27,30]=100
        self.assertIsNone(self.witness(hits=np.array([self.g.cell_to_world(30,27)])))
    def test_path_too_far_from_robot_no_guess(self):
        self.assertIsNone(self.witness(pose=(2.,4.,0.)))
    def test_malformed_path_no_claim(self):
        for path in ([],[(2.,2.)],[(float('nan'),2.),(4.,2.)]):
            self.assertIsNone(self.witness(points=path))
    def test_rotated_grid_supported(self):
        g=fx.Geometry(origin=(10.,3.,math.pi/2))
        self.assertIsNotNone(self.witness(points=[(8.,5.),(8.,9.)],pose=(8.,5.,math.pi/2),geom=g,
                             hits=np.array([g.cell_to_world(30,20)])))
    def test_same_data_not_mutated(self):
        orig=self.cost.copy();prior=self.b[3].copy();self.witness()
        np.testing.assert_array_equal(orig,self.cost);np.testing.assert_array_equal(prior,self.b[3])
    def test_range_max_and_infinite_not_hits(self):
        pts=project_scan_hits([1.,float('inf'),8.,float('nan'),0.],0.,.1,.1,8.,(0.,0.,0.))
        self.assertEqual(pts.shape,(1,2))
    def test_scan_motion_interpolation(self):
        pts=project_scan_hits([1.,1.],0.,0.,.1,8.,(0.,0.,0.),(1.,0.,0.))
        np.testing.assert_allclose(pts,[[1.,0.],[2.,0.]])
    def test_unsafe_settings_rejected(self):
        for s in (BlockageSettings(minimum_samples=1),BlockageSettings(hit_tolerance_m=1.),BlockageSettings(sample_step_m=float('nan'))):
            with self.assertRaises(ValueError):s.validate()

class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.w=PersistentBlockage();self.e=dict(xy=[3.,2.],cost=100,prior_free=True,scan_supported=True)
    def test_stall_and_independent_observations_trigger(self):
        r=None
        for t in range(7):r=self.w.update(t,(2.,2.,t),self.e,t,t) or r
        self.assertTrue(r['translation_stalled'])
    def test_same_costmap_cannot_count_repeated_messages(self):
        self.assertTrue(all(self.w.update(t,(2.,2.,0.),self.e,0,t) is None for t in range(10)))
    def test_same_scan_cannot_count_repeated_messages(self):
        self.assertTrue(all(self.w.update(t,(2.,2.,0.),self.e,t,0) is None for t in range(10)))
    def test_making_translation_progress_does_not_cancel(self):
        self.assertTrue(all(self.w.update(t,(2.+.3*t,2.,0.),self.e,t,t) is None for t in range(10)))
    def test_rotation_alone_not_translation_progress(self):
        self.test_stall_and_independent_observations_trigger()
    def test_clear_path_resets_obstacle_evidence(self):
        for t in range(5):self.w.update(t,(2.,2.,0.),self.e,t,t)
        self.w.update(5,(2.,2.,0.),None,5,5)
        self.assertIsNone(self.w.update(6,(2.,2.,0.),self.e,6,6))
    def test_clock_reset_cannot_trigger(self):
        for t in range(5):self.w.update(10+t,(2.,2.,0.),self.e,10+t,10+t)
        self.assertIsNone(self.w.update(0,(2.,2.,0.),self.e,0,0))
    def test_data_gap_resets(self):
        self.w.update(1,(2.,2.,0.),self.e,1,1)
        self.assertIsNone(self.w.update(10,(2.,2.,0.),self.e,10,10))

class CutTests(unittest.TestCase):
    def setUp(self):
        self.b=fx.bundle();self.v=np.zeros_like(self.b[2]);self.v[:,40]=100
    def cut(self,**kw):
        args=dict(geom=self.b[1],prior_free=self.b[3],cost=self.v,pose=(2.,2.,0.),labels=self.b[2],mask_id=2,fixed_prior_topology=True)
        args.update(kw);return current_map_cut(**args)
    def test_closed_wall_cut_confirmed_under_assumption(self):self.assertTrue(self.cut()['zone_cut'])
    def test_no_assumption_no_whole_zone_claim(self):self.assertFalse(self.cut(fixed_prior_topology=False)['zone_cut'])
    def test_outside_zone_detour_preserves_connectivity(self):
        self.v[60,40]=0;self.assertFalse(self.cut()['zone_cut'])
    def test_unknown_gap_may_be_route_not_blocked(self):
        self.v[60,40]=-1;self.assertFalse(self.cut()['zone_cut'])
    def test_inflated_only_barrier_is_not_proof(self):
        self.v[:,40]=99;self.assertFalse(self.cut()['zone_cut'])
    def test_bad_start_not_unreachable(self):self.assertFalse(self.cut(pose=(-1.,0.,0.))['zone_cut'])
    def test_empty_target_not_unreachable(self):self.assertFalse(self.cut(mask_id=100)['zone_cut'])
    def test_preexisting_disconnected_prior_not_new_cut(self):
        self.b[3][:,40]=False;self.assertFalse(self.cut()['zone_cut'])

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        # Baseline fake ROS intentionally omits Path; v3 opts in explicitly.
        sys.modules['nav_msgs.msg'].Path=type('Path',(),{})
        self.tmp=tempfile.TemporaryDirectory();self.b=fx.bundle();self.s=fx.snapshot(self.b)
        self.args=zone_node.parser().parse_args(['--connectivity-aware','--path-blockage-check','--prior-topology-fixed',
            '--execute','--supervised','--continue-on-nav-abort','--visibility-backend','python','--threshold','.96'])
        self.n=zone_node.make_node_type(self.args,self.b,Settings(threshold=.96),Path(self.tmp.name))()
        self.n.inputs=lambda:(None,self.s);self.n.local_start_clear=lambda *a:True
        self.n.stop_evidence.ready=lambda *a:True;self.n.book.update(self.s['status'])
        self.n.book.context('grid1',time.monotonic());self.n.boot_wall-=10
        self.n.current=dict(id='entry',zone_id='R_A',goal_kind='ENTRY',x=6.,y=2.,yaw=0.,
            new_indices=[],visible_indices=[],new_cells=0,new_area_m2=0.,path_est_m=4.)
        self.n.current_path=[(2.,2.),(6.,2.)];self.n.validated_plan_mode='NAVFN_GOAL_VALIDATED'
        self.n.nav_request(self.s);self.n.poll_actions(self.s)
        self.m=self.n.path_blockage;self.s['cost']['values'][20,30]=100
        self.hit=self.b[1].cell_to_world(30,20)
    def tearDown(self):self.n.close();self.tmp.cleanup()
    def tick(self,t,valid=True):
        self.n.sim=float(t);self.s['cost'].update(stamp=float(t),wall=time.monotonic())
        self.m.hits=dict(points=np.array([self.hit]),stamp=float(t),wall=time.monotonic())
        self.m.last_check=0.;self.m.step(self.s if valid else None)
    def trigger(self):
        for t in range(10,16):self.tick(t)
        self.assertEqual(self.n.cancel_reason,CANCEL_REASON)
    def terminal(self):
        self.n.nav_client.handles[0].result_future.set(NS(status=5))
        self.n.poll_actions(self.s)
    def stopped(self):
        self.n.sim=18.;self.s['status']['evidence_scans']=2;self.s['status']['last_evidence_scan_end_s']=18.
        self.s['cost'].update(stamp=18.,wall=time.monotonic())
        self.n.after_result(self.s)
    def test_stalled_blocked_route_cancels_owned_goal(self):
        self.trigger();self.assertEqual(self.n.nav_client.handles[0].cancels,1)
        self.assertFalse(self.n.stop_requested)
    def test_no_competing_planner_request_during_navigation(self):
        self.trigger();self.assertEqual(self.n.plan_client.sent,[])
    def test_cancel_ack_not_permission_to_plan(self):
        self.trigger();self.n.poll_actions(self.s)
        self.assertEqual(self.n.plan_client.sent,[]);self.assertEqual(self.m.records,[])
    def test_actual_terminal_then_stop_then_replan(self):
        self.trigger();self.terminal();self.stopped()
        self.assertEqual(len(self.m.records),1);self.assertEqual(self.n.succeeded_goals,0)
        self.assertEqual(self.n.coverage_cancels,0);self.assertIsNone(self.n.nav_uuid)
        self.n.connectivity.step(self.s)
        self.assertEqual(len(self.n.plan_client.sent),1)
        self.assertTrue(self.n.current['path_blockage_probe'])
        self.assertEqual(len(self.n.nav_client.sent),1)
    def test_no_fresh_stop_blocks_continuation(self):
        self.trigger();self.terminal();self.n.stop_evidence.ready=lambda *a:False;self.stopped()
        self.assertEqual(self.m.records,[]);self.assertEqual(self.n.plan_client.sent,[])
    def test_no_new_coverage_blocks_continuation(self):
        self.trigger();self.terminal();self.n.sim=16.;self.n.after_result(self.s)
        self.assertEqual(self.m.records,[])
    def test_unsafe_local_start_stops(self):
        self.trigger();self.terminal();self.n.local_start_clear=lambda *a:False;self.stopped()
        self.assertEqual(self.n.state,'PATH_BLOCKAGE_LOCAL_START_UNSAFE')
        self.assertEqual(self.n.plan_client.sent,[])
    def test_safety_stop_has_priority_over_blockage(self):
        self.trigger();self.n.request_stop('DATA_LOSS_STALE_GLOBAL_COSTMAP');self.terminal();self.stopped()
        self.assertEqual(self.n.state,'DATA_LOSS_STALE_GLOBAL_COSTMAP');self.assertEqual(self.m.records,[])
    def test_external_cancel_is_not_approved(self):
        self.n.nav_client.handles[0].result_future.set(NS(status=5));self.n.poll_actions(self.s)
        self.assertEqual(self.n.stop_requested,'EXTERNAL_OR_SAFETY_CANCEL')
    def test_stale_input_resets_persistence(self):
        for t in range(10,14):self.tick(t)
        self.tick(14,False);self.tick(15)
        self.assertIsNone(self.n.cancel_reason)
    def test_latest_clear_detour_prevents_cancel(self):
        for t in range(10,14):self.tick(t)
        self.n.sim=14.;self.send_plan([(2.,2.),(2.,1.),(6.,1.),(6.,2.)],14.)
        self.tick(14);self.tick(15);self.assertIsNone(self.n.cancel_reason)
    def send_plan(self,points,stamp,frame='map'):
        poses=[]
        for x,y in points:
            p=fake_ros.PoseStamped();p.header.frame_id=frame;p.pose.position.x=x;p.pose.position.y=y;poses.append(p)
        msg=NS(header=NS(frame_id=frame,stamp=NS(sec=int(stamp),nanosec=0)),poses=poses)
        self.m.on_plan(msg)
    def test_other_goal_plan_does_not_replace_route(self):
        self.send_plan([(2.,2.),(7.,7.)],10.)
        self.assertEqual(self.m.path['source'],'validated_preflight_reference')
    def test_old_or_wrong_frame_plan_not_used(self):
        self.send_plan([(2.,2.),(6.,2.)],1.)
        self.send_plan([(2.,2.),(6.,2.)],10.,'odom')
        self.assertEqual(self.m.path['source'],'validated_preflight_reference')
    def test_replan_failure_enters_alternative_audit_not_new_motion(self):
        self.trigger();self.terminal();self.stopped();self.n.connectivity.step(self.s)
        self.n.poll_actions(self.s)
        self.n.plan_client.handles[-1].result_future.set(NS(status=6,result=NS(path=None)))
        self.n.poll_actions(self.s)
        self.assertEqual(self.n.connectivity.blockage_results['R_A'][0]['reason'],'NAVFN_ABORTED')
        self.assertEqual(len(self.n.nav_client.sent),1)
    def test_replan_success_requires_fresh_preflight(self):
        self.trigger();self.terminal();self.stopped();self.n.connectivity.step(self.s);self.n.poll_actions(self.s)
        points=[(2.,2.),(2.,1.),(6.,1.),(6.,2.)];poses=[]
        for x,y in points:
            p=fake_ros.PoseStamped();p.header.frame_id='map';p.pose.position.x=x;p.pose.position.y=y;poses.append(p)
        self.n.plan_client.handles[-1].result_future.set(NS(status=4,result=NS(path=NS(header=NS(frame_id='map'),poses=poses))))
        self.n.poll_actions(self.s);self.assertIsNotNone(self.n.connectivity.blockage_verified)
        self.n.connectivity.step(self.s)
        self.assertFalse(self.n.current.get('connectivity_probe',False))
        self.assertEqual(len(self.n.nav_client.sent),1)
        self.assertEqual(len(self.n.plan_client.sent),2)
    def test_current_map_cut_after_replan_and_alternates_remains_partial(self):
        # Complete observed barrier, no new opening in the prior topology.
        from connectivity_audit import sensor_signature
        self.s['cost']['values'][:,30]=100
        self.trigger();self.terminal();self.stopped()
        rt=self.n.connectivity;rt.step(self.s);self.n.poll_actions(self.s)
        self.n.plan_client.handles[-1].result_future.set(NS(status=6,result=NS(path=None)))
        self.n.poll_actions(self.s)
        self.m.hits=dict(points=np.array([self.hit]),stamp=self.n.sim,wall=time.monotonic())
        c=dict(id='other_entry',zone_id='R_A',goal_kind='ENTRY',x=7.,y=3.,yaw=0.,
            visible_indices=[],new_indices=[],new_cells=0,new_area_m2=0.,path_est_m=5.)
        rt.catalog=dict(cost_key=self.s['cost']['key'],epoch=self.s['epoch'],
            sensor_signature=sensor_signature(self.s['sensor']),start_pose=list(self.s['pose']),
            positions=[dict(position_id='other',component=1,probe=c)])
        rt.entry_sites=[dict(position_id='other',component=1,probe=c)];rt.entry_index=0
        rt.step(self.s);self.n.poll_actions(self.s)
        self.n.plan_client.handles[-1].result_future.set(NS(status=6,result=NS(path=None)))
        self.n.poll_actions(self.s);rt.step(self.s)
        self.assertEqual(self.n.book.records['R_A'].state,'PARTIAL_DEFERRED')
        self.assertEqual(self.n.book.outcomes['R_A']['classification'],'CURRENT_MAP_UNREACHABLE')
        self.assertFalse(self.n.book.outcomes['R_A']['normal_termination'])
        self.assertIn('R_B',[z['id'] for z in self.n.book.eligible()])
        self.assertEqual(len(self.n.nav_client.sent),1)

    def test_same_probe_failure_without_sensor_witness_not_unreachable(self):
        self.trigger();self.terminal();self.stopped()
        rt=self.n.connectivity;rt.start_zone('R_A')
        rt.blockage_results['R_A']=[dict(reason='NAVFN_ABORTED')]
        self.m.hits=None
        e=rt.blockage_partial_evidence(self.s,{})
        self.assertFalse(e.get('current_map_unreachable',False))

    def test_cancel_record_is_serializable_evidence(self):
        import json
        self.trigger();self.terminal();self.stopped()
        self.assertTrue(validate_cancel_record(self.m.records[0]));json.dumps(self.m.records,allow_nan=False)
    def test_existing_v2_launcher_optout(self):
        a=zone_node.parser().parse_args(['--connectivity-aware']);self.assertFalse(a.path_blockage_check)

class ReturnTests(unittest.TestCase):
    def fixture(self):
        from test_connectivity import return_fixture
        a,b=return_fixture();a['options']['path_blockage_check']=True;b['path_blockage_check']=True
        b['sent']=28;b['path_blockage_canceled']=1
        b['path_blockage_cancellations']=[dict(goal_id='a'*32,reason=CANCEL_REASON,terminal='CANCELED',
            stop_confirmed=True,new_coverage=True,evidence=dict(zone='Z0',bundle_signature='test-signature',
            persistent=True,translation_stalled=True,scan_supported=True,prior_free=True,cost=100,independent_samples=3))]
        return a,b
    def test_approved_cancel_accounted_separately(self):
        a,b=self.fixture();self.assertTrue(evaluate_return_eligibility(a,b)['eligible'])
    def test_no_evidence_cannot_return(self):
        a,b=self.fixture();b['path_blockage_cancellations'][0]['evidence']={}
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])
    def test_stop_missing_cannot_return(self):
        a,b=self.fixture();b['path_blockage_cancellations'][0]['stop_confirmed']=False
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])
    def test_duplicate_uuid_cannot_return(self):
        a,b=self.fixture();b['sent']=29;b['path_blockage_canceled']=2;b['path_blockage_cancellations']*=2
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])
    def test_goal_limit_still_blocks_return(self):
        a,b=self.fixture();b['state']='GOAL_LIMIT_REACHED'
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])
    def test_forged_option_cannot_return(self):
        a,b=self.fixture();a['options']['path_blockage_check']=False
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])
    def test_wrong_bundle_cannot_return(self):
        a,b=self.fixture();b['path_blockage_cancellations'][0]['evidence']['bundle_signature']='other'
        self.assertFalse(evaluate_return_eligibility(a,b)['eligible'])

if __name__=='__main__':unittest.main()
