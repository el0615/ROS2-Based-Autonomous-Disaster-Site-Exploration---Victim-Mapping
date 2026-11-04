from fixtures import *
import os, time, tempfile, unittest
from types import SimpleNamespace as NS
import fake_ros
fake_ros.install()
os.environ['ZB_PROJECT_ROOT']='/synthetic'
import zone_node
from zb_reference_planning import Settings

class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.b=bundle();self.s=snapshot(self.b)
        self.args=zone_node.parser().parse_args(['--visibility-backend','python','--max-candidates','4','--heading-samples','2'])
        self.node=zone_node.make_node_type(self.args,self.b,Settings(),Path(self.tmp.name))()
        self.node.inputs=lambda:(None,self.s)
        self.node.stop_evidence.ready=lambda *a:True
        self.node.local_start_clear=lambda inp:True
        self.node.boot_wall-=5
        self.node.book.update(self.s['status']);self.node.book.context('grid1',time.monotonic())
        self.node.matched=dict(status=self.s['status'])
        self.c=self.node.zone_planner.entry_batch(self.s,[self.b[0]['zones'][1]],[])['candidates'][0]
    def tearDown(self):
        self.node.close();self.tmp.cleanup()
    def plan_ok(self):
        self.node.plan_request(self.c,self.s)
        self.node.poll_actions(self.s)
        h=self.node.plan_client.handles[-1]
        a=fake_ros.PoseStamped();a.header.frame_id='map';a.pose.position.x=self.s['pose'][0];a.pose.position.y=self.s['pose'][1]
        b=self.node.make_pose(self.c)
        h.result_future.set(NS(status=4,result=NS(path=NS(header=NS(frame_id='map'),poses=[a,b]))))
        self.node.poll_actions(self.s)
        self.assertEqual(self.node.state,'READY_TO_SEND')
    def start_nav(self):
        self.args.execute=True;self.args.supervised=True
        self.plan_ok();self.node.poll()
        self.assertEqual(self.node.sent_goals,1)
        self.node.poll_actions(self.s)
        self.assertEqual(self.node.state,'NAV_RUNNING')
    def test_dryrun_never_sends(self):
        self.plan_ok();self.node.poll()
        self.assertTrue(self.node.done);self.assertEqual(self.node.state,'DRY_RUN_PASS')
        self.assertEqual(len(self.node.nav_client.sent),0)
    def test_execute_refuses_missing_supervision(self):
        self.args.execute=True;self.node.current=self.c;self.node.validated_plan_mode='NAVFN_GOAL_VALIDATED'
        with self.assertRaises(RuntimeError):self.node.nav_request(self.s)
    def test_execute_refuses_old_strict_or_alt(self):
        self.args.execute=True;self.args.supervised=True;self.node.current=self.c
        for mode in [None,'STRICT','STRICT_SWEPT','ALT_STATIC_DRY_RUN_ONLY']:
            self.node.validated_plan_mode=mode
            with self.assertRaises(RuntimeError):self.node.nav_request(self.s)
    def test_entry_zero_gain_preflight_accepted(self):
        self.plan_ok();self.assertEqual(self.node.current['new_cells'],0)
        self.assertEqual(self.node.validated_plan_mode,'NAVFN_GOAL_VALIDATED')
    def test_navfn_accepted_when_only_legacy_full_sweep_would_reject_inflation(self):
        # Simulated Navfn route centerline is free. An inflated cell beside the
        # robot causes the OLD STRICT_SWEPT checker to reject the entire path.
        # The NEW pilot delegates path collision handling to Nav2; goal stays
        # valid and never lies on an occupied/unknown cell.
        from zb_reference_sequence import validate_oriented_swept_path
        self.s['cost']['values'][22,20]=99
        pts=[self.s['pose'][:2], (self.c['x'],self.c['y'])]
        with self.assertRaisesRegex(ValueError,'FOOTPRINT_SWEEP_COLLISION'):
            validate_oriented_swept_path(pts,self.s['pose'],self.c,
                self.b[1],self.b[3],self.s['cost']['values'],Settings())
        self.plan_ok()
        self.assertEqual(self.node.validated_plan_mode,'NAVFN_GOAL_VALIDATED')
        self.assertEqual(len(self.node.nav_client.sent),0)  # Dry Run is passive.
        self.assertIn('NAVFN_GOAL_CHECK_OK',
            (Path(self.tmp.name)/'events.jsonl').read_text())

    def test_navfn_goal_on_lethal_cell_stays_rejected(self):
        self.s['cost']['values'][self.c['cell'][1],self.c['cell'][0]]=100
        self.node.plan_request(self.c,self.s)
        self.node.poll_actions(self.s)
        h=self.node.plan_client.handles[-1]
        a=fake_ros.PoseStamped();a.header.frame_id='map'
        a.pose.position.x=self.s['pose'][0];a.pose.position.y=self.s['pose'][1]
        h.result_future.set(NS(status=4,result=NS(path=NS(header=NS(frame_id='map'),
            poses=[a,self.node.make_pose(self.c)]))))
        self.node.poll_actions(self.s)
        self.assertIsNone(self.node.validated_plan_mode)
        self.assertEqual(len(self.node.nav_client.sent),0)
        self.assertIn('PLAN_REJECTED',(Path(self.tmp.name)/'events.jsonl').read_text())

    def test_navfn_action_abort_cannot_authorize_goal(self):
        self.node.plan_request(self.c,self.s)
        self.node.poll_actions(self.s)
        h=self.node.plan_client.handles[-1]
        h.result_future.set(NS(status=6,result=NS(path=None)))
        self.node.poll_actions(self.s)
        self.assertIsNone(self.node.validated_plan_mode)
        self.assertEqual(len(self.node.nav_client.sent),0)

    def test_stale_costmap_result_not_authorized(self):
        self.node.plan_request(self.c,self.s);self.node.poll_actions(self.s)
        self.s['cost']['key']='changed'
        self.node.plan_client.handles[-1].result_future.set(NS(status=4,result=NS(path=None)))
        self.node.poll_actions(self.s)
        self.assertIsNone(self.node.validated_plan_mode);self.assertEqual(len(self.node.book.rejected),0)
    def test_stale_yaw_rejected(self):
        self.node.plan_request(self.c,self.s);self.node.poll_actions(self.s)
        self.s['pose']=(2.,2.,.5)
        self.node.plan_client.handles[-1].result_future.set(NS(status=4,result=NS(path=None)))
        self.node.poll_actions(self.s);self.assertIsNone(self.node.current)
    def test_foreign_goal_blocks_navigation(self):
        self.node.nav_statuses={'foreign':2};self.node.poll()
        self.assertEqual(self.node.stop_requested,'OTHER_NAVIGATION_GOAL_ACTIVE')
        self.assertEqual(len(self.node.nav_client.sent),0)
    def test_one_owned_goal_only(self):
        self.start_nav()
        with self.assertRaisesRegex(RuntimeError,'Overlapping'):self.node.nav_request(self.s)
    def test_success_requires_new_post_result_coverage(self):
        self.start_nav();self.node.nav_client.handles[-1].result_future.set(NS(status=4,result=NS()))
        self.node.poll_actions(self.s);self.node.after_result(self.s)
        self.assertEqual(self.node.succeeded_goals,0)
        self.assertEqual(self.node.state,'WAIT_STOP_AND_COVERAGE')
    def test_stop_confirmed_message_once_while_waiting_for_new_coverage(self):
        self.start_nav()
        self.node.nav_client.handles[-1].result_future.set(NS(status=4,result=NS()))
        self.node.poll_actions(self.s)
        notices=[]
        original_info=self.node.info
        def capture(key,message,*args,**kwargs):
            notices.append(message)
            return original_info(key,message,*args,**kwargs)
        self.node.info=capture
        for _ in range(75):
            self.node.after_result(self.s)  # Same old Coverage, fresh stop continues
        self.assertEqual(sum('[STOP_CONFIRMED]' in msg for msg in notices),1)
        self.assertEqual(self.node.state,'WAIT_STOP_AND_COVERAGE')
        self.assertEqual(self.node.succeeded_goals,0)
        self.assertEqual(self.node.sent_goals,1)
        self.assertIsNotNone(self.node.nav_uuid)
        self.assertFalse(self.node.done)

    def test_new_goal_can_log_stop_confirmed_again(self):
        self.start_nav()
        self.node.nav_client.handles[-1].result_future.set(NS(status=4,result=NS()))
        self.node.poll_actions(self.s)
        notices=[]
        orig=self.node.info
        def capture(key,message,*args,**kwargs):
            notices.append(message)
            return orig(key,message,*args,**kwargs)
        self.node.info=capture
        self.node.after_result(self.s)
        first=self.node.nav_uuid
        self.assertEqual(sum('[STOP_CONFIRMED]' in msg for msg in notices),1)
        # A simulated subsequent Goal is issued; only this message state changes.
        self.node.nav_uuid='e'*32
        self.node.after_result(self.s)
        self.node.after_result(self.s)
        self.assertNotEqual(first,self.node.nav_uuid)
        self.assertEqual(sum('[STOP_CONFIRMED]' in msg for msg in notices),2)

    def test_stop_check_is_not_cached_by_log_dedup(self):
        self.start_nav()
        self.node.nav_client.handles[-1].result_future.set(NS(status=4,result=NS()))
        self.node.poll_actions(self.s)
        self.node.after_result(self.s)
        self.node.stop_evidence.ready=lambda *a:False
        self.node.after_result(self.s)
        self.assertEqual(self.node.succeeded_goals,0)
        self.assertEqual(self.node.state,'WAIT_STOP_AND_COVERAGE')

    def test_success_requires_stop_not_action_alone(self):
        self.start_nav();self.node.nav_client.handles[-1].result_future.set(NS(status=4,result=NS()))
        self.node.poll_actions(self.s);self.node.stop_evidence.ready=lambda *a:False
        self.node.after_result(self.s);self.assertEqual(self.node.succeeded_goals,0)
    def test_entry_to_local_phase_after_real_pose_membership(self):
        self.start_nav();self.node.nav_client.handles[-1].result_future.set(NS(status=4,result=NS()))
        self.node.poll_actions(self.s)
        self.node.sim=12.;self.s['pose']=(self.c['x'],self.c['y'],self.c['yaw'])
        self.s['status']=status(self.b,seq=3,t=12.);self.node.book.update(self.s['status'])
        self.node.after_result(self.s)
        self.assertTrue(self.node.book.visit_state['R_A'].entered)
        self.assertEqual(self.node.succeeded_goals,1)
        self.node.select_zone_goal(self.s)
        self.assertEqual(self.node.job_context['kind'],'OBSERVE')
    def test_success_outside_zone_schedules_one_safe_alternate_entry(self):
        self.start_nav();self.node.nav_client.handles[-1].result_future.set(NS(status=4,result=NS()))
        self.node.poll_actions(self.s);self.node.sim=12.;self.s['status']=status(self.b,seq=3,t=12.)
        self.node.after_result(self.s)
        self.assertFalse(self.node.book.visit_state['R_A'].entered)
        self.assertEqual(self.node.book.visit_state['R_A'].entry_arrival_retries,1)
        self.assertIsNone(self.node.book.records['R_A'].deferred_reason)
        self.assertEqual(self.node.book.active_zone,'R_A')
        logs=(Path(self.tmp.name)/'events.jsonl').read_text()
        self.assertIn('ENTRY_ARRIVAL_MISMATCH',logs)
        self.assertIn('ENTRY_ALTERNATE_RETRY',logs)
        self.node.select_zone_goal(self.s)
        self.assertEqual(self.node.job_context['kind'],'ENTRY')
        self.assertEqual(self.node.job_context['zone'],'R_A')
        self.node.worker.result(timeout=10)

    def test_arrival_mismatch_exhausted_is_deferred(self):
        self.start_nav();self.node.nav_client.handles[-1].result_future.set(NS(status=4,result=NS()))
        self.node.poll_actions(self.s);self.node.sim=12.;self.s['status']=status(self.b,seq=3,t=12.)
        self.node.book.visit_state['R_A'].entry_arrival_retries=1
        self.node.after_result(self.s)
        self.assertFalse(self.node.book.visit_state['R_A'].entered)
        self.assertEqual(self.node.book.records['R_A'].deferred_reason,'TERMINAL_POSE_OUTSIDE_ZONE_REVIEW_TOLERANCE')
        self.assertIn('ENTRY_ARRIVAL_MISMATCH',(Path(self.tmp.name)/'events.jsonl').read_text())

    def test_arrival_mismatch_unsafe_start_is_deferred(self):
        self.start_nav();self.node.nav_client.handles[-1].result_future.set(NS(status=4,result=NS()))
        self.node.poll_actions(self.s);self.node.sim=12.;self.s['status']=status(self.b,seq=3,t=12.)
        self.node.local_start_clear=lambda inp:False
        self.node.after_result(self.s)
        self.assertEqual(self.node.book.records['R_A'].deferred_reason,'TERMINAL_POSE_OUTSIDE_ZONE_REVIEW_TOLERANCE')
        self.assertEqual(self.node.book.visit_state['R_A'].entry_arrival_retries,0)

    def test_entry_retry_not_authorized_after_goal_limit(self):
        self.args.max_goals=1
        self.start_nav();self.node.nav_client.handles[-1].result_future.set(NS(status=4,result=NS()))
        self.node.poll_actions(self.s);self.node.sim=12.;self.s['status']=status(self.b,seq=3,t=12.)
        self.node.after_result(self.s)
        self.assertEqual(self.node.book.records['R_A'].deferred_reason,'TERMINAL_POSE_OUTSIDE_ZONE_REVIEW_TOLERANCE')
        self.assertEqual(self.node.book.visit_state['R_A'].entry_arrival_retries,0)
        self.assertEqual(self.node.state,'GOAL_LIMIT_REACHED')
    def test_prefetch_sends_no_actions(self):
        self.start_nav();before=len(self.node.plan_client.sent)+len(self.node.nav_client.sent)
        self.node.prefetch_during_navigation(self.s)
        self.node.worker.result(timeout=10.)
        self.node.collect_job(self.s)
        self.assertEqual(len(self.node.plan_client.sent)+len(self.node.nav_client.sent),before)
        self.assertEqual(self.node.options,[])
    def test_abort_stops_by_default(self):
        self.start_nav();self.node.nav_client.handles[-1].result_future.set(NS(status=6,result=NS()))
        self.node.poll_actions(self.s)
        self.assertEqual(self.node.stop_requested,'NAV_ABORTED_INSPECT_BEFORE_RETRY')
    def test_optin_abort_defers_only_after_stop_and_fresh_coverage(self):
        self.args.continue_on_nav_abort=True;self.start_nav()
        self.node.nav_client.handles[-1].result_future.set(NS(status=6,result=NS()))
        self.node.poll_actions(self.s);self.assertIsNone(self.node.stop_requested)
        self.node.sim=12.;self.s['status']=status(self.b,seq=3,t=12.)
        self.node.after_result(self.s)
        self.assertEqual(self.node.book.records['R_A'].state,'PARTIAL_DEFERRED')
        self.assertIsNone(self.node.nav_uuid)
    def test_optin_abort_unsafe_pose_halts(self):
        self.args.continue_on_nav_abort=True;self.start_nav()
        self.node.nav_client.handles[-1].result_future.set(NS(status=6,result=NS()))
        self.node.poll_actions(self.s);self.node.local_start_clear=lambda inp:False
        self.node.sim=12.;self.s['status']=status(self.b,seq=3,t=12.)
        self.node.after_result(self.s)
        self.assertTrue(self.node.done)
    def test_data_loss_cancels_owned_goal(self):
        self.start_nav();self.node.inputs=lambda:('STALE_COVERAGE',None)
        self.node.bad_since=time.monotonic()-3;self.node.poll()
        self.assertTrue(self.node.stop_requested.startswith('DATA_LOSS'))
        self.assertGreater(self.node.nav_client.handles[-1].cancels,0)
    def test_late_acceptance_still_cancelled(self):
        self.args.execute=True;self.args.supervised=True;self.plan_ok()
        self.node.nav_request(self.s)
        self.node.request_stop('USER_STOP_SERVICE')
        self.node.poll_actions(self.s)
        self.assertGreater(self.node.nav_client.handles[-1].cancels,0)
    def test_completed_zone_transitions_to_another_room(self):
        self.node.book.activate('R_A')
        self.s['observed'] |= self.b[2] == 2
        self.node.book.update(status(self.b,self.s['observed'],2,11.))
        self.node.book.update(status(self.b,self.s['observed'],3,12.2))
        self.assertTrue(self.node.book.is_complete('R_A'))
        self.node.select_zone_goal(self.s)
        self.node.worker.result(timeout=10.)
        self.node.select_zone_goal(self.s)
        self.assertEqual(self.node.current['zone_id'],'R_B')
        self.assertEqual(self.node.current['goal_kind'],'ENTRY')
        self.assertEqual(len(self.node.nav_client.sent),0)

    def test_goal_limit_stops_without_extra_goal(self):
        self.args.max_goals=1;self.start_nav()
        self.node.nav_client.handles[-1].result_future.set(NS(status=4,result=NS()))
        self.node.poll_actions(self.s);self.node.sim=12.
        self.s['pose']=(self.c['x'],self.c['y'],self.c['yaw']);self.s['status']=status(self.b,seq=3,t=12.)
        self.node.after_result(self.s)
        self.assertEqual(self.node.state,'GOAL_LIMIT_REACHED');self.assertEqual(len(self.node.nav_client.sent),1)

if __name__=='__main__':unittest.main(verbosity=2)
