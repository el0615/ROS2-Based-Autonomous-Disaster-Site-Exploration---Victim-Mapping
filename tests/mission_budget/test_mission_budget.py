"""Synthetic policy + real patched state-machine / mocked ROS transport tests.
No ROS/DDS, physical safety, Gazebo or real LiDAR validation is claimed.
"""
import copy, json, math, os, sys, tempfile, time, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import asdict, replace
ROOT=Path(os.environ.get('DN012_TEST_PROJECT',Path(__file__).resolve().parents[2]))
for p in (ROOT/'scripts/exploration/zone_exploration/tests',
          ROOT/'scripts/exploration/zone_exploration/src',
          ROOT/'scripts/exploration/navigation/return_home'):
    sys.path.insert(0,str(p))
import fixtures as fx
import fake_ros
fake_ros.install()
os.environ['ZB_PROJECT_ROOT']=str(ROOT)
from mission_budget import *
from zone_policy import ConnectivityMission
from planning_core import Settings
import zone_node
from return_home_policy import evaluate_return_eligibility, _approved_operational_cancellations

class StagnationTests(unittest.TestCase):
    def setUp(self):self.s=GuardSettings();self.w=StagnationWindow(self.s)
    def run_samples(self,pose,rc=lambda t:t//4):
        results=[]
        for t in range(21):results.append(self.w.sample(float(t),pose(t),rc(t)))
        return results
    def test_stationary_recovery_triggers(self):
        e=self.run_samples(lambda t:(2.,2.,0.))[-1]
        self.assertEqual(e['reason'],STAGNATION);self.assertFalse(e['physical_blockage_proven'])
    def test_rotation_is_not_translation(self):
        self.assertIsNotNone(self.run_samples(lambda t:(2.,2.,t))[-1])
    def test_real_translation_not_canceled(self):
        self.assertTrue(all(e is None for e in self.run_samples(lambda t:(2.+.1*t,2.,0.))))
    def test_loop_endpoint_not_mistaken_for_stall(self):
        self.assertTrue(all(e is None for e in self.run_samples(lambda t:(math.sin(t*math.pi/10),math.cos(t*math.pi/10),0.))))
    def test_less_than_twenty_seconds_not_enough(self):
        self.assertTrue(all(e is None for e in self.run_samples(lambda t:(2.,2.,0.))[:-1]))
    def test_normal_rotation_without_recovery_does_not_cancel(self):
        self.assertTrue(all(e is None for e in self.run_samples(lambda t:(2.,2.,t),lambda t:0)))
    def test_gap_resets_evidence(self):
        for t in range(15):self.w.sample(t,(2,2),t)
        self.assertIsNone(self.w.sample(25,(2,2),25))
    def test_clock_reset_does_not_trigger(self):
        for t in range(15):self.w.sample(100+t,(2,2),t)
        self.assertIsNone(self.w.sample(1,(2,2),100))
    def test_same_timestamp_not_multiple_samples(self):
        for i in range(200):self.assertIsNone(self.w.sample(10.,(2,2),i))
        self.assertEqual(len(self.w.samples),1)
    def test_invalid_pose_resets(self):
        self.w.sample(10,(2,2),5);self.w.sample(11,(float('nan'),2),6)
        self.assertEqual(len(self.w.samples),0)
    def test_invalid_settings_rejected(self):
        for s in (replace(self.s,stall_extent_m=0),replace(self.s,work_max_s=10),
                  replace(self.s,goal_recovery_limit=-1),replace(self.s,stall_window_s=float('nan')),
                  replace(self.s,zone_cancel_limit=True)):
            with self.assertRaises(ValueError):s.validate()

class LedgerTests(unittest.TestCase):
    def setUp(self):self.l=ZoneWorkLedger(GuardSettings())
    def test_no_progress_expires_at_180(self):
        for t in range(181):self.l.tick(t,94.,True)
        self.assertTrue(self.l.expired);self.assertFalse(self.l.extended)
    def test_recent_progress_extends_once_to_300(self):
        for t in range(181):self.l.tick(t,90.+.02*t,True)
        self.assertTrue(self.l.extended);self.assertFalse(self.l.expired);self.assertEqual(self.l.allowance,300.)
        for t in range(181,301):self.l.tick(t,94.,True)
        self.assertTrue(self.l.expired)
    def test_initial_entry_gain_not_recent_credit(self):
        for t in range(181):self.l.tick(t,90. if t<5 else 94.,True)
        self.assertTrue(self.l.expired)
    def test_excluded_travel_not_charged(self):
        for t in range(1000):self.l.tick(t,80.,False)
        self.assertEqual(self.l.elapsed,0.);self.assertFalse(self.l.expired)
    def test_pause_does_not_erase_budget(self):
        for t in range(101):self.l.tick(t,94.,True)
        self.l.pause()
        for t in range(1000,1081):self.l.tick(t,94.,True)
        self.assertAlmostEqual(self.l.elapsed,180.);self.assertTrue(self.l.expired)
    def test_transport_gap_not_counted(self):
        self.l.tick(0,90,True);self.l.tick(100,90,True)
        self.assertEqual(self.l.elapsed,0.)
    def test_clock_reset_no_negative_time(self):
        self.l.tick(100,90,True);self.l.tick(101,90,True);self.l.tick(0,90,True)
        self.assertEqual(self.l.elapsed,1.)
    def test_transit_gain_does_not_extend(self):
        for t in range(100):self.l.tick(t,90.,True)
        for t in range(100,200):self.l.tick(t,94.,False)
        for t in range(200,282):self.l.tick(t,94.,True)
        self.assertTrue(self.l.expired)
    def test_invalid_coverage_does_not_add_time(self):
        self.l.tick(0,90,True);self.l.tick(1,float('nan'),True)
        self.assertEqual(self.l.elapsed,0.)

class NoRevisitTests(unittest.TestCase):
    def setUp(self):
        self.b=fx.bundle();self.book=ConnectivityMission(self.b[0]['zones'],threshold=.96,no_auto_revisit=True)
        self.book.context('a',1.)
    def test_costmap_progress_and_cooldown_do_not_reopen(self):
        self.book.finish_partial('R_A','ENTRY_ACCESS_UNVERIFIED_ALTERNATES_EXHAUSTED')
        self.book.context('b',1000.);self.book.progress=100
        self.assertEqual(self.book.reopen_changed(),[])
        self.assertNotIn('R_A',[z['id'] for z in self.book.eligible()])
    def test_low_gain_deferral_is_also_locked(self):
        self.book.finish_partial('R_A','LOW_GAIN_PLATEAU_UNVERIFIED')
        self.assertIn('R_A',self.book.deferred_this_mission)
    def test_passive_coverage_confirmed_without_revisit_or_topology_claim(self):
        # P0 specification correction: no revisit != suppress fresh Coverage.
        import numpy as np
        self.book.finish_partial('R_A','SAMPLED_BUDGET')
        for i in range(1,4):self.book.update(fx.status(self.b,np.ones_like(self.b[2],dtype=bool),i,float(i)))
        self.assertEqual(self.book.records['R_A'].measured,1.)
        self.assertEqual(self.book.records['R_A'].state,'OBSERVATION_SUFFICIENT')
        self.assertTrue(self.book.all_resolved())
        self.assertEqual(self.book.reopen_changed(),[])
        self.assertNotIn('R_A',[z['id'] for z in self.book.eligible()])
        self.assertEqual(self.book.outcomes['R_A']['classification'],'UNVERIFIED')
        self.assertEqual(self.book.summary()['R_A']['exploration_status'],'DEFERRED')
    def test_old_mode_reopens_unchanged(self):
        b=ConnectivityMission(self.b[0]['zones'],threshold=.96)
        b.context('a',1.);b.finish_partial('R_A','ENTRY_ACCESS_UNVERIFIED_ALTERNATES_EXHAUSTED')
        b.context('b',1000);b.progress=10
        self.assertEqual(b.reopen_changed(),['R_A'])
    def test_new_mission_does_not_inherit_locks(self):
        self.book.finish_partial('R_A','LIMIT')
        b=ConnectivityMission(self.b[0]['zones'],threshold=.96,no_auto_revisit=True)
        self.assertIn('R_A',[z['id'] for z in b.eligible()])
    def test_other_zone_remains_eligible(self):
        self.book.finish_partial('R_A','LIMIT')
        self.assertIn('R_B',[z['id'] for z in self.book.eligible()])

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.b=fx.bundle();self.s=fx.snapshot(self.b)
        self.args=zone_node.parser().parse_args(['--connectivity-aware','--mission-budget-guard',
              '--visibility-backend','python','--threshold','.96','--execute','--supervised','--continue-on-nav-abort'])
        self.n=zone_node.make_node_type(self.args,self.b,Settings(threshold=.96),Path(self.tmp.name))()
        self.g=self.n.mission_guard;self.n.inputs=lambda:(None,self.s)
        self.n.session_id='test-session';self.n.stop_evidence.ready=lambda *x:True
        self.n.local_start_clear=lambda *x:True
        self.n.book.update(self.s['status']);self.n.book.context('grid1',time.monotonic())
        self.n.boot_wall-=5
        self.c=self.n.zone_planner.entry_batch(self.s,[self.b[0]['zones'][1]],[])['candidates'][0]
    def tearDown(self):self.n.close();self.tmp.cleanup()
    def start(self,kind='ENTRY'):
        self.n.current=dict(self.c,goal_kind=kind)
        self.n.validated_plan_mode='NAVFN_GOAL_VALIDATED'
        self.n.nav_request(self.s);self.n.poll_actions(self.s)
    def feed(self,recoveries,gid=None):
        gid=gid or self.n.nav_uuid
        self.n.on_feedback(NS(goal_id=NS(uuid=list(bytes.fromhex(gid))),
           feedback=NS(number_of_recoveries=recoveries,distance_remaining=5.)))
    def request_cap(self):
        self.feed(20);self.g.step(self.s)
        self.assertEqual(self.n.cancel_reason,RECOVERY_LIMIT)
    def terminal(self,status=5):
        self.n.nav_client.handles[-1].result_future.set(NS(status=status))
        self.n.poll_actions(self.s)
        self.s['status']['evidence_scans']=100
        self.s['status']['last_evidence_scan_end_s']=self.n.sim+1
        self.n.sim+=1
    def test_policy_is_optin_and_old_parser_defaults_off(self):
        self.assertFalse(zone_node.parser().parse_args([]).mission_budget_guard)
    def test_running_goal_recovery_ceiling_cancels_owned_only(self):
        self.start();self.request_cap()
        self.assertEqual(self.n.nav_client.handles[-1].cancels,1)
        self.assertEqual(len(self.n.plan_client.sent),0)
    def test_translating_recovery_ceiling_still_operational(self):
        self.start();self.s['pose']=(4.,4.,0.);self.request_cap()
        self.assertFalse(self.g.pending['evidence']['physical_blockage_proven'])
    def test_normal_long_entry_with_progress_is_not_timed_out(self):
        self.start()
        for t in range(10,350):
            self.n.sim=float(t);self.s['pose']=(2.+.1*t,2.,0.)
            self.feed(2);self.g.account(self.s);self.g.step(self.s)
        self.assertIsNone(self.n.cancel_reason)
        self.assertEqual(self.g.ledgers['R_A'].elapsed,0.)
    def test_twenty_second_stall_uses_fresh_poses(self):
        self.start()
        for t in range(21):
            self.n.sim=float(10+t);self.feed(t//4);self.g.step(self.s)
        self.assertEqual(self.n.cancel_reason,STAGNATION)
    def test_external_goal_feedback_ignored(self):
        self.start();self.feed(100,'f'*32)
        self.assertIsNone(self.g.recoveries)
    def test_stale_feedback_cannot_create_stagnation(self):
        self.start();self.feed(128);self.g.feedback_wall=time.monotonic()-4
        self.g.step(self.s);self.assertIsNone(self.n.cancel_reason)
    def test_long_missing_feedback_safely_stops_mission(self):
        self.start();self.g.feedback_start=time.monotonic()-11
        self.g.step(self.s);self.assertEqual(self.n.stop_requested,'DATA_LOSS_GUARD_FEEDBACK')
    def test_existing_safety_stop_takes_priority(self):
        self.start();self.n.request_stop('DATA_LOSS_STALE_GLOBAL_COSTMAP');self.feed(128);self.g.step(self.s)
        self.assertIsNone(self.g.pending);self.assertEqual(self.n.cancel_reason,'DATA_LOSS_STALE_GLOBAL_COSTMAP')
    def test_path_blockage_cancel_is_not_overwritten(self):
        self.start();self.n.request_cancel('PATH_BLOCKAGE_REPLAN_REQUIRED');self.feed(128);self.g.step(self.s)
        self.assertIsNone(self.g.pending)
    def test_ack_alone_never_continues(self):
        self.start();self.request_cap();self.n.poll_actions(self.s)
        self.assertIsNone(self.n.nav_terminal);self.assertEqual(len(self.g.records),0)
    def test_no_new_coverage_blocks_continuation(self):
        self.start();self.request_cap();self.terminal()
        self.s['status']['evidence_scans']=self.n.post_result_scans
        self.n.after_result(self.s)
        self.assertEqual(len(self.g.records),0);self.assertIsNotNone(self.n.nav_uuid)
    def test_no_fresh_stop_blocks_continuation(self):
        self.start();self.request_cap();self.terminal();self.n.stop_evidence.ready=lambda *x:False
        self.n.after_result(self.s);self.assertEqual(len(self.g.records),0)
    def test_unsafe_local_pose_stops_no_alternate(self):
        self.start();self.request_cap();self.terminal();self.n.local_start_clear=lambda *x:False
        self.n.after_result(self.s)
        self.assertEqual(self.n.state,'MISSION_GUARD_LOCAL_START_UNSAFE')
        self.assertEqual(len(self.g.records),0)
    def test_confirmed_entry_cancel_reuses_existing_alternative_audit(self):
        self.start();self.request_cap();self.terminal();self.n.after_result(self.s)
        self.assertEqual(len(self.g.records),1);self.assertIsNone(self.n.nav_uuid)
        self.assertEqual(self.n.book.audit_requested['R_A'],'ENTRY_OPERATIONAL_CANCEL_ACCESS_RECHECK')
        self.n.connectivity.start_zone('R_A')
        self.assertTrue(self.n.connectivity.entry_mode)
        self.assertEqual(len(self.n.plan_client.sent),0)
        self.assertEqual(self.n.aborted_goals,0)
    def test_observe_cancel_requests_observation_audit(self):
        self.start('OBSERVE');self.request_cap();self.terminal();self.n.after_result(self.s)
        self.assertEqual(self.n.book.audit_requested['R_A'],'OPERATIONAL_CANCEL_OBSERVATION_RECHECK')
    def test_second_operational_cancel_defers_only_target_zone(self):
        for i in range(2):
            self.start();self.request_cap();self.terminal();self.n.after_result(self.s)
        self.assertEqual(self.n.book.records['R_A'].state,'PARTIAL_DEFERRED')
        self.assertEqual(self.n.book.outcomes['R_A']['reason'],'NAVIGATION_RETRY_BUDGET_UNVERIFIED')
        self.assertIn('R_B',[z['id'] for z in self.n.book.eligible()]);self.assertFalse(self.n.done)
    def test_success_racing_cancel_is_not_counted_as_cancel(self):
        self.start();self.request_cap();self.terminal(4);self.s['pose']=(6.,2.,0.)
        self.n.after_result(self.s)
        self.assertEqual(len(self.g.records),0);self.assertEqual(self.n.succeeded_goals,1)
    def test_user_stop_overrides_pending_operational_cancel(self):
        self.start();self.request_cap();self.n.request_stop('USER_STOP_SERVICE');self.terminal();self.n.after_result(self.s)
        self.assertEqual(self.n.state,'USER_STOP_SERVICE');self.assertEqual(len(self.g.records),0)
    def test_completed_zone_not_deferred_due_cancel(self):
        self.start();self.request_cap();self.terminal();self.n.book.records['R_A'].state='OBSERVATION_SUFFICIENT'
        self.n.after_result(self.s)
        self.assertEqual(self.n.book.records['R_A'].state,'OBSERVATION_SUFFICIENT')
    def test_wrong_epoch_fails_closed(self):
        self.start();self.request_cap();self.terminal();self.s['epoch']=2;self.n.after_result(self.s)
        self.assertEqual(self.n.state,'MISSION_GUARD_SESSION_UNVERIFIED')
    def test_guard_record_saved_and_reported_separately(self):
        self.start();self.request_cap();self.terminal();self.n.after_result(self.s);self.n.publish_report(force=True)
        d=json.loads((Path(self.tmp.name)/'latest_status.json').read_text())
        self.assertEqual(d['operational_canceled'],1);self.assertEqual(d['path_blockage_canceled'],0)
        self.assertEqual(len(list((Path(self.tmp.name)/'mission_guard').glob('*.json'))),1)
    def test_poll_wiring_requests_cancel_without_new_planner(self):
        self.start();self.feed(20);self.n.prefetch_during_navigation=lambda inp:None
        self.n.poll()
        self.assertEqual(self.n.cancel_reason,RECOVERY_LIMIT)
        self.assertEqual(len(self.n.plan_client.sent),0)
    def test_poll_data_loss_has_priority_over_recovery_cap(self):
        self.start();self.feed(128)
        self.n.inputs=lambda:('STALE_GLOBAL_COSTMAP',None)
        self.n.bad_since=time.monotonic()-3.;self.n.prefetch_during_navigation=lambda inp:None
        self.n.poll()
        self.assertEqual(self.n.stop_requested,'DATA_LOSS_STALE_GLOBAL_COSTMAP')
        self.assertIsNone(self.g.pending)
    def test_poll_ready_to_send_budget_cannot_launch(self):
        self.expired_ledger();self.n.current=dict(self.c);self.n.state='READY_TO_SEND'
        self.n.validated_plan_mode='NAVFN_GOAL_VALIDATED'
        self.n.poll()
        self.assertEqual(len(self.n.nav_client.sent),0)
        self.assertEqual(self.n.book.records['R_A'].state,'PARTIAL_DEFERRED')
    def expired_ledger(self):
        l=ZoneWorkLedger(self.g.s);l.elapsed=181.;l.expired=True
        self.g.ledgers['R_A']=l;self.n.book.activate('R_A')
    def test_work_budget_cancels_then_defers_without_retry(self):
        self.start('OBSERVE');self.expired_ledger();self.g.step(self.s)
        self.assertEqual(self.n.cancel_reason,ZONE_LIMIT)
        self.terminal();self.n.after_result(self.s)
        self.assertEqual(self.n.book.outcomes['R_A']['reason'],ZONE_LIMIT)
    def test_work_budget_before_new_goal_prevents_navigation(self):
        self.expired_ledger();self.n.current=dict(self.c);self.n.state='READY_TO_SEND'
        self.assertTrue(self.g.before_selection(self.s))
        self.assertEqual(len(self.n.nav_client.sent),0)
        self.assertEqual(self.n.book.records['R_A'].state,'PARTIAL_DEFERRED')
    def test_work_budget_waits_for_active_cpu_worker(self):
        self.expired_ledger();self.n.worker=fake_ros.Future()
        self.assertTrue(self.g.before_selection(self.s))
        self.assertNotEqual(self.n.book.records['R_A'].state,'PARTIAL_DEFERRED')
        self.n.worker.set({});self.assertTrue(self.g.before_selection(self.s))
        self.assertEqual(self.n.book.records['R_A'].state,'PARTIAL_DEFERRED')
    def test_budget_never_interrupts_pending_planner_as_if_navigation(self):
        self.expired_ledger();self.n.plan_send=fake_ros.Future()
        self.assertFalse(self.g.before_selection(self.s))
    def test_inside_observe_charges_time(self):
        self.start('OBSERVE');self.s['pose']=(6.,2.,0.)
        for t in range(10,40):self.n.sim=float(t);self.g.account(self.s)
        self.assertAlmostEqual(self.g.ledgers['R_A'].elapsed,29.)
    def test_outside_zone_observe_detour_is_excluded(self):
        self.start('OBSERVE');self.s['pose']=(2.,2.,0.)
        for t in range(10,400):self.n.sim=float(t);self.g.account(self.s)
        self.assertEqual(self.g.ledgers['R_A'].elapsed,0.)
    def test_feedback_recovery_cap_can_be_disabled(self):
        self.start();self.g.s=replace(self.g.s,goal_recovery_limit=0)
        self.feed(128);self.g.step(self.s);self.assertIsNone(self.n.cancel_reason)

class WiringTests(unittest.TestCase):
    def test_launcher_forwards_new_guard_and_audit_24(self):
        import subprocess,shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);nav=root/'scripts/exploration/navigation';(nav/'return_home').mkdir(parents=True)
            shutil.copy2(ROOT/'scripts/exploration/navigation/run_legacy_mission.sh',nav/'run_legacy_mission.sh')
            (nav/'return_home/run_mission_with_home.py').write_text('import sys\nprint("\\n".join(sys.argv[1:]))\n')
            out=subprocess.run(['bash',str(nav/'run_legacy_mission.sh'),'budget','dry-run','--max-goals','3'],
                env=dict(os.environ,PROJECT=str(root)),capture_output=True,text=True,check=True).stdout.splitlines()
            self.assertIn('--mission-budget-guard',out)
            self.assertEqual(out[out.index('--audit-max-view-preflights')+1],'24')
            self.assertEqual(out[out.index('--mode')+1],'dry-run')
            self.assertEqual(out[out.index('--max-goals')+1],'3')
    def test_supervisor_cli_and_child_options_match_exactly(self):
        import ast,argparse
        source=ROOT/'scripts/exploration/navigation/return_home/run_mission_with_home.py'
        tree=ast.parse(source.read_text());cli=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='cli')
        process=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='mission_process')
        body=[]
        for stmt in process.body:
            if isinstance(stmt,ast.Assign) and any(ast.unparse(t)=="record['mission_command']" for t in stmt.targets):break
            body.append(stmt)
        process.body=body+[ast.Return(value=ast.Name(id='cmd',ctx=ast.Load()))]
        module=ast.fix_missing_locations(ast.Module(body=[cli,process],type_ignores=[]))
        env=dict(argparse=argparse,Path=Path,PROJECT_NAME='synthetic',__doc__='synthetic')
        exec(compile(module,str(source),'exec'),env)
        old=sys.argv
        try:
            sys.argv=['test','--connectivity-aware','--mission-budget-guard','--path-blockage-check','--audit-max-view-preflights','24']
            options,_=env['cli']()
        finally:sys.argv=old
        cmd=env['mission_process'](None,ROOT,options,None,{})
        parsed=zone_node.parser().parse_args(cmd[3:])
        self.assertTrue(parsed.mission_budget_guard)
        self.assertEqual(parsed.audit_max_view_preflights,24)
        self.assertEqual(parsed.guard_goal_recovery_limit,20)

class ReturnTests(unittest.TestCase):
    def setUp(self):
        cfg=asdict(GuardSettings());opts=dict(connectivity_aware=True,mission_budget_guard=True,
                 threshold=.96,execute=True,zone_ids=None,continue_on_nav_abort=True)
        mapping={'stall_window_s':'guard_stall_window_s','stall_extent_m':'guard_stall_extent_m',
             'stall_recoveries':'guard_stall_recoveries','goal_recovery_limit':'guard_goal_recovery_limit',
             'zone_cancel_limit':'guard_zone_cancel_limit','work_base_s':'guard_zone_work_base_s',
             'work_max_s':'guard_zone_work_max_s','extension_window_s':'guard_extension_window_s',
             'extension_gain_pp':'guard_extension_gain_pp'}
        opts.update({name:cfg[key] for key,name in mapping.items()})
        self.m=dict(options=opts,zone_bundle_signature='sig')
        self.row=dict(schema='dn012_operational_cancel_v1',goal_id='a'*32,zone='R0',goal_kind='ENTRY',
              reason=RECOVERY_LIMIT,bundle_signature='sig',coverage_session='session',
              requested_ros_s=10.,confirmed_ros_s=12.,terminal='CANCELED',stop_confirmed=True,
              new_coverage=True,local_start_clear=True,settings=cfg,
              evidence=dict(reason=RECOVERY_LIMIT,recoveries=20,recovery_limit=20,
                            physical_blockage_proven=False,normal_termination=False))
        zs={f'R{i}':dict(state='OBSERVATION_SUFFICIENT',measured_percent=99.) for i in range(27)}
        zs['R0']=dict(state='PARTIAL_DEFERRED',measured_percent=3.,deferred_reason='NAVIGATION_RETRY_BUDGET_UNVERIFIED')
        self.s=dict(state='PARTIAL_DEFERRED_REMAIN',threshold=.96,execute=True,connectivity_aware=True,
             mission_budget_guard=True,zone_bundle_signature='sig',coverage_session='session',
             sent=2,nav_succeeded=1,nav_aborted=0,coverage_canceled=0,stop_reason=None,owned_goal_id=None,
             operational_canceled=1,operational_cancellations=[self.row],zones=zs)
    def eval(self):return evaluate_return_eligibility(self.m,self.s)
    def test_verified_operational_cancel_can_return_as_partial(self):
        e=self.eval();self.assertTrue(e['eligible']);self.assertEqual(e['exploration'],'PARTIAL')
    def test_missing_stop_rejected(self):self.row['stop_confirmed']=False;self.assertFalse(self.eval()['eligible'])
    def test_local_start_unsafe_rejected(self):self.row['local_start_clear']=False;self.assertFalse(self.eval()['eligible'])
    def test_missing_coverage_rejected(self):self.row['new_coverage']=False;self.assertFalse(self.eval()['eligible'])
    def test_counter_only_rejected(self):self.s['operational_cancellations']=[];self.assertFalse(self.eval()['eligible'])
    def test_wrong_bundle_rejected(self):self.row['bundle_signature']='bad';self.assertFalse(self.eval()['eligible'])
    def test_wrong_session_rejected(self):self.row['coverage_session']='bad';self.assertFalse(self.eval()['eligible'])
    def test_wrong_options_rejected(self):self.m['options']['mission_budget_guard']=False;self.assertFalse(self.eval()['eligible'])
    def test_insufficient_recovery_evidence_rejected(self):self.row['evidence']['recoveries']=19;self.assertFalse(self.eval()['eligible'])
    def test_duplicate_uuid_rejected(self):
        self.s['operational_canceled']=2;self.s['operational_cancellations'].append(copy.deepcopy(self.row))
        self.assertFalse(self.eval()['eligible'])
    def test_external_cancel_not_authorized(self):self.row['reason']='USER_STOP_SERVICE';self.assertFalse(self.eval()['eligible'])
    def test_user_stop_never_returns(self):self.s['stop_reason']='USER_STOP_SERVICE';self.assertFalse(self.eval()['eligible'])
    def test_goal_limit_never_returns(self):self.s['state']='GOAL_LIMIT_REACHED';self.assertFalse(self.eval()['eligible'])
    def test_operational_cancel_does_not_make_unvisited_zone_resolved(self):
        self.s['zones']['R1']['state']='UNOBSERVED';self.assertFalse(self.eval()['eligible'])
    def test_forged_large_limit_rejected(self):self.row['settings']['goal_recovery_limit']=30;self.assertFalse(self.eval()['eligible'])
    def test_valid_stagnation_record(self):
        self.row['reason']=STAGNATION;self.row['evidence']=dict(reason=STAGNATION,window_s=20.,extent_m=.1,
          recovery_increase=5,sample_count=30,physical_blockage_proven=False,normal_termination=False)
        self.assertTrue(self.eval()['eligible'])
    def test_valid_work_budget_record(self):
        self.row['reason']=ZONE_LIMIT;self.row['evidence']=dict(reason=ZONE_LIMIT,work_time_ros_s=180.,allowance_ros_s=180.,
          expired=True,physical_blockage_proven=False,normal_termination=False)
        self.assertTrue(self.eval()['eligible'])
    def test_invalid_work_budget_record(self):
        self.row['reason']=ZONE_LIMIT;self.row['evidence'].update(reason=ZONE_LIMIT,work_time_ros_s=20.,allowance_ros_s=180.,expired=True)
        self.assertFalse(self.eval()['eligible'])
    def test_held_zone_passive_coverage_not_full_mission(self):
        self.s['zones']['R0']['measured_percent']=99.
        self.assertEqual(self.eval()['exploration'],'PARTIAL')

if __name__=='__main__':unittest.main()
