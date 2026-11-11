"""P0 coverage state and reporting regression. Synthetic ROS only; no real motion."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest

ROOT = Path(os.environ.get('DN012_TEST_PROJECT', Path(__file__).resolve().parents[2])).resolve()
ZT=ROOT/'scripts/exploration/zone_exploration/tests'
SRC=ROOT/'scripts/exploration/zone_exploration/src'
REPORTS=ROOT/'scripts/exploration/navigation/reports'
for p in (ZT,SRC,REPORTS,ROOT/'scripts/exploration/navigation/return_home'):
    sys.path.insert(0,str(p))
import fixtures as fx
import numpy as np
from zone_policy import ConnectivityMission
import zone_report as cr
import zone_report as v1

class StateTests(unittest.TestCase):
    def setUp(self):
        self.b=fx.bundle();self.z='R_A'
        self.book=ConnectivityMission(self.b[0]['zones'],threshold=.96,no_auto_revisit=True)
        self.book.context('grid1',time.monotonic())
        self.book.update(self.status(.94,1,1.))
        self.evidence={'source':'test','connectivity_verified':False,'point':[1.,2.]}
        self.book.finish_partial(self.z,'LOW_GAIN_PLATEAU_UNVERIFIED',evidence=self.evidence)
    def status(self,f,seq,t,all_zones=False):
        seen=np.zeros_like(self.b[2],dtype=bool)
        ids=np.flatnonzero((self.b[2]==2).ravel())
        seen.ravel()[ids[:int(len(ids)*f)]]=True
        if all_zones:seen[:]=True
        return fx.status(self.b,seen,seq,t)
    def confirm(self):
        self.book.update(self.status(.97,2,2.))
        return self.book.update(self.status(.97,3,3.1))
    def test_first_threshold_measurement_pending_not_complete(self):
        self.book.update(self.status(.97,2,2.))
        self.assertEqual(self.book.records[self.z].state,'THRESHOLD_PENDING')
        self.assertFalse(self.book.is_complete(self.z))
        self.assertNotIn(self.z,[z['id'] for z in self.book.eligible()])
    def test_second_distinct_scan_confirms_deferred_coverage(self):
        self.assertEqual(self.confirm(),[self.z])
        self.assertEqual(self.book.records[self.z].state,'OBSERVATION_SUFFICIENT')
    def test_repeated_sequence_does_not_confirm(self):
        self.book.update(self.status(.97,2,2.))
        self.book.update(self.status(.97,2,20.))
        self.assertFalse(self.book.is_complete(self.z))
    def test_too_short_interval_does_not_confirm(self):
        self.book.update(self.status(.97,2,2.))
        self.book.update(self.status(.97,3,2.9))
        self.assertFalse(self.book.is_complete(self.z))
    def test_confirm_does_not_unlock_revisit(self):
        self.confirm();self.book.context('changed',1000.);self.book.progress+=30
        self.assertIn(self.z,self.book.deferred_this_mission)
        self.assertEqual(self.book.reopen_changed(),[])
        self.assertNotIn(self.z,[z['id'] for z in self.book.eligible()])
    def test_original_assessment_and_reason_retained(self):
        old=deepcopy(self.book.outcomes[self.z]);self.confirm()
        self.assertEqual(old,self.book.outcomes[self.z])
        self.assertEqual(self.book.records[self.z].deferred_reason,'LOW_GAIN_PLATEAU_UNVERIFIED')
    def test_other_zone_not_claimed_complete(self):
        self.confirm();self.assertFalse(self.book.all_complete());self.assertFalse(self.book.all_resolved())
    def test_all_zones_can_confirm_without_losing_deferral_history(self):
        self.book.update(self.status(1.,2,2.,True));self.book.update(self.status(1.,3,3.1,True))
        self.assertTrue(self.book.all_complete());self.assertTrue(self.book.all_resolved())
        self.assertEqual(self.book.summary()[self.z]['exploration_status'],'DEFERRED')
    def test_progress_counted_exactly_once(self):
        before=self.book.progress;self.confirm();after=self.book.progress
        self.assertEqual(after-before,1)
        for i in range(4,10):self.assertEqual(self.book.update(self.status(.97,i,float(i))),[])
        self.assertEqual(after,self.book.progress)
    def test_false_complete_not_created_below_target(self):
        self.book.update(self.status(.95,2,2));self.book.update(self.status(.95,3,3.1))
        self.assertEqual(self.book.records[self.z].state,'PARTIAL_DEFERRED')
    def test_new_session_still_fatal(self):
        s=self.status(.97,2,2);s['session_id']='other'
        with self.assertRaisesRegex(ValueError,'SESSION_CHANGED'):self.book.update(s)
    def test_regression_still_fatal(self):
        with self.assertRaisesRegex(ValueError,'REGRESSION'):self.book.update(self.status(.8,2,2))
    def test_confirmation_clears_pending_audit(self):
        self.book.audit_requested[self.z]='existing';self.confirm()
        self.assertNotIn(self.z,self.book.audit_requested)
    def test_confirmation_does_not_release_other_zone_lock(self):
        self.book.activate('R_B');self.confirm();self.assertEqual(self.book.active_zone,'R_B')
    def test_same_zone_lock_released_on_confirmation(self):
        self.book.activate(self.z);self.confirm();self.assertIsNone(self.book.active_zone)
    def test_history_measured_at_deferral_not_final(self):
        initial=100*self.book.records[self.z].measured;self.confirm()
        r=self.book.summary()[self.z]
        self.assertEqual(r['coverage_at_last_deferral_percent'],initial)
        self.assertGreater(r['measured_percent'],initial)
        self.assertEqual(r['coverage_confirmation_evidence']['coverage_session'],'test-session')
    def test_input_evidence_mutation_does_not_change_history(self):
        self.evidence['point'][0]=99.
        self.assertEqual(self.book.outcomes[self.z]['evidence']['point'][0],1.)
        self.assertEqual(self.book.deferral_history[self.z][0]['evidence']['point'][0],1.)
    def test_summary_history_is_a_copy(self):
        s=self.book.summary();s[self.z]['deferral_history'][0]['reason']='bad'
        self.assertEqual(self.book.deferral_history[self.z][0]['reason'],'LOW_GAIN_PLATEAU_UNVERIFIED')
    def test_no_redeferral_overwrites_confirmed_outcome(self):
        self.confirm();old=deepcopy(self.book.outcomes[self.z]);self.book.finish_partial(self.z,'NEW_REASON')
        self.assertTrue(self.book.is_complete(self.z));self.assertEqual(old,self.book.outcomes[self.z])
    def test_certificate_not_created_by_target_coverage(self):
        self.confirm();s=self.book.summary()[self.z]
        self.assertEqual(s['assessment']['classification'],'UNVERIFIED')
        self.assertFalse(s['assessment']['normal_termination'])
        self.assertTrue(s['coverage_confirmed']);self.assertTrue(s['coverage_does_not_certify_connectivity'])
    def test_duplicate_deferral_not_appended_for_locked_zone(self):
        self.book.finish_partial(self.z,'LATER');self.assertEqual(len(self.book.deferral_history[self.z]),1)
    def test_low_gain_time_and_access_deferrals_same_coverage_semantics(self):
        for reason in ['ZONE_WORK_BUDGET_UNVERIFIED','AUDIT_CONTEXT_UNSTABLE_UNVERIFIED','PATH_BLOCKED_NO_CONFIRMED_DETOUR']:
            with self.subTest(reason=reason):
                b=ConnectivityMission(self.b[0]['zones'],threshold=.96,no_auto_revisit=True)
                b.update(self.status(.94,1,1));b.finish_partial(self.z,reason)
                b.update(self.status(.97,2,2));b.update(self.status(.97,3,3.1))
                self.assertTrue(b.is_complete(self.z));self.assertEqual(b.outcomes[self.z]['reason'],reason)
    def test_pending_summary_is_explicit(self):
        self.book.update(self.status(.97,2,2));r=self.book.summary()[self.z]
        self.assertTrue(r['coverage_target_reached']);self.assertFalse(r['coverage_confirmed'])
        self.assertEqual(r['coverage_status'],'TARGET_REACHED_UNCONFIRMED')
    def test_old_connectivity_mode_still_confirms(self):
        b=ConnectivityMission(self.b[0]['zones'],threshold=.96)
        b.update(self.status(.94,1,1));b.finish_partial(self.z,'LOW_GAIN_PLATEAU_UNVERIFIED')
        b.update(self.status(.97,2,2));b.update(self.status(.97,3,3.1))
        self.assertTrue(b.is_complete(self.z))

class ReportTests(unittest.TestCase):
    def setUp(self):
        self.raw=json.loads((Path(__file__).parent/'fixtures/zone_results_reported_20261105.json').read_text())
    def test_historical_counts_are_16_numeric_12_confirmed_4_unconfirmed(self):
        r=cr.enrich(self.raw);c=r['coverage_counts']
        self.assertEqual(c['numeric_target_reached'],16);self.assertEqual(c['TARGET_CONFIRMED'],12)
        self.assertEqual(c['TARGET_REACHED_UNCONFIRMED'],4);self.assertEqual(c['BELOW_TARGET'],11)
        self.assertEqual(set(r['coverage_confirmation_gap_zones']),{'R04','R05','R08A','R10A'})
    def test_no_mutation_of_historical_input(self):
        old=deepcopy(self.raw);cr.enrich(self.raw);self.assertEqual(self.raw,old)
    def test_mission_and_zone_verdicts_retained(self):
        r=cr.enrich(self.raw)
        self.assertEqual(r['mission_state'],'PARTIAL_DEFERRED_REMAIN');self.assertFalse(r['normal_termination'])
        for zid,z in r['zones'].items():
            self.assertEqual(z['zone_state'],self.raw['zones'][zid]['zone_state'])
            self.assertEqual(z['exploration_result'],self.raw['zones'][zid]['exploration_result'])
    def test_blockage_points_and_accessibility_unchanged(self):
        r=cr.enrich(self.raw)
        for zid,z in r['zones'].items():
            for k in ['blocked_points_xy','local_connectivity','global_access','assessment_classification']:
                self.assertEqual(z[k],self.raw['zones'][zid][k])
    def test_old_missing_history_not_invented(self):
        z=cr.enrich(self.raw)['zones']['R04'];self.assertIsNone(z['deferral_history'])
        self.assertIsNone(z['coverage_at_last_deferral_percent']);self.assertIsNone(z['coverage_confirmation_evidence'])
    def test_original_confirmed_state_is_labeled_recorded_only(self):
        r=cr.enrich(self.raw);self.assertEqual(r['zones']['R01']['confirmation_source'],'RECORDED_STATE_ONLY')
    def test_no_retrospective_confirmation_for_high_coverage(self):
        z=cr.enrich(self.raw)['zones']['R08A'];self.assertTrue(z['coverage_target_reached'])
        self.assertFalse(z['coverage_confirmed']);self.assertEqual(z['confirmation_source'],'NOT_PROVEN')
    def test_subthreshold_value_never_rounds_up(self):
        self.raw['zones']['R04']['measured_coverage_percent']=95.999999
        z=cr.enrich(self.raw)['zones']['R04'];self.assertFalse(z['coverage_target_reached'])
    def test_bad_numeric_input_refused(self):
        for value in [True,float('nan'),float('inf'),-1,101,'96']:
            with self.subTest(value=value):
                r=deepcopy(self.raw);r['zones']['R04']['measured_coverage_percent']=value
                with self.assertRaises(ValueError):cr.enrich(r)
    def test_sufficient_below_threshold_refused(self):
        self.raw['zones']['R01']['measured_coverage_percent']=95.
        with self.assertRaisesRegex(ValueError,'STATE_BELOW_THRESHOLD'):cr.enrich(self.raw)
    def test_csv_and_markdown_keep_distinct_axes(self):
        r=cr.enrich(self.raw);text=cr.csv_text(r);md=cr.markdown(r)
        self.assertIn('coverage_status',text);self.assertIn('exploration_status',text)
        self.assertIn('수치 달성·확인 부족',md);self.assertIn('PARTIAL_DEFERRED_REMAIN',md)
    def test_cli_legacy_output_does_not_change_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'zone_results.json';original=json.dumps(self.raw);p.write_text(original)
            self.assertEqual(cr.main(['coverage','--legacy-report',str(p)]),0)
            self.assertEqual(p.read_text(),original)
            self.assertTrue((Path(tmp)/'zone_assessment_p0/zone_results_p0.json').is_file())
    def test_confirmed_live_evidence_requires_new_scan_and_time(self):
        row={'coverage_confirmation_evidence':{'source':'EXISTING_THRESHOLD_TRACKER','coverage_session':'s',
             'confirms_accessibility':False,'first_above_threshold_scan_end_s':1.,'last_evidence_scan_end_s':2.1,
             'first_above_threshold_scan_sequence':1,'evidence_scans':2,'threshold_percent':96.,'measured_percent':97.}}
        self.assertTrue(cr.evidence_valid(row,'s',96.))
        for key,val in [('evidence_scans',1),('last_evidence_scan_end_s',1.9),('coverage_session','x'),('confirms_accessibility',True)]:
            r=deepcopy(row);r['coverage_confirmation_evidence'][key]=val
            self.assertFalse(cr.evidence_valid(r,'s',96.))
    def test_source_schema_mismatch_refused(self):
        self.raw['schema']='x'
        with self.assertRaisesRegex(ValueError,'EXPECTED_V1'):cr.enrich(self.raw)

class NodeIntegrationTests(unittest.TestCase):
    def setUp(self):
        import fake_ros
        fake_ros.install()
        import zone_node
        from planning_core import Settings
        os.environ['ZB_PROJECT_ROOT']=str(ROOT)
        self.tmp=tempfile.TemporaryDirectory();self.b=fx.bundle();self.s=fx.snapshot(self.b)
        self.args=zone_node.parser().parse_args(['--connectivity-aware','--mission-budget-guard',
                    '--threshold','.96','--visibility-backend','python'])
        self.n=zone_node.make_node_type(self.args,self.b,Settings(threshold=.96),Path(self.tmp.name))()
        self.n.session_id='test-session';self.n.book.context('grid1',time.monotonic())
        self.n.book.update(self.s['status']);self.n.book.finish_partial('R_A','LOW_GAIN_PLATEAU_UNVERIFIED')
        self.n.stop_evidence.ready=lambda *x:True;self.n.inputs=lambda:(None,self.s);self.n.boot_wall-=5
    def tearDown(self):self.n.close();self.tmp.cleanup()
    def confirm(self):
        seen=self.b[2]==2
        self.n.book.update(fx.status(self.b,seen,2,2.));self.n.book.update(fx.status(self.b,seen,3,3.1))
    def test_report_serializes_coverage_and_historical_deferral(self):
        self.confirm();self.n.publish_report(force=True)
        r=json.loads((Path(self.tmp.name)/'latest_status.json').read_text())['zones']['R_A']
        self.assertEqual(r['state'],'OBSERVATION_SUFFICIENT');self.assertEqual(r['exploration_status'],'DEFERRED')
        self.assertEqual(len(self.n.nav_client.sent),0)
    def test_passive_update_does_not_send_or_cancel_goal(self):
        self.confirm();self.assertEqual(len(self.n.nav_client.sent),0)
        self.assertIsNone(self.n.cancel_reason);self.assertIsNone(self.n.nav_uuid)
    def test_budget_does_not_redefer_confirmed_coverage(self):
        from mission_budget import ZoneWorkLedger
        self.confirm();self.n.book.activate('R_A')
        ledger=ZoneWorkLedger(self.n.mission_guard.s);ledger.elapsed=181;ledger.expired=True
        self.n.mission_guard.ledgers['R_A']=ledger
        self.n.mission_guard.account(self.s)
        self.assertIsNone(self.n.mission_guard.work_zone)
        self.assertTrue(self.n.book.is_complete('R_A'))
    def test_low_gain_does_not_downgrade_confirmed_coverage(self):
        self.confirm();r={'zone_id':'R_A','goal_kind':'OBSERVE','nav_status':'SUCCEEDED'}
        self.assertFalse(self.n.connectivity.consider_low_gain_plateau(r,'test-session',True))
        self.assertTrue(self.n.book.is_complete('R_A'))
    def test_new_summary_to_extended_report_retains_unverified_accessibility(self):
        self.confirm();self.n.publish_report(force=True)
        summary=json.loads((Path(self.tmp.name)/'latest_status.json').read_text())
        zones={zid:v1.classify_zone(zid,row,[],{}) for zid,row in summary['zones'].items()}
        r=dict(schema='dn012_zone_assessment_report_v1',coverage_target_percent=96.,
            coverage_session='test-session',zones=zones,warnings=[],limits=[],mission_state='PARTIAL_DEFERRED_REMAIN')
        q=cr.enrich(r,summary)['zones']['R_A']
        self.assertTrue(q['coverage_confirmed']);self.assertEqual(q['exploration_status'],'DEFERRED')
        self.assertEqual(q['global_access'],'NOT_VERIFIED');self.assertEqual(q['local_connectivity'],'LOCAL_TOPOLOGY_UNVERIFIED')
    def test_connectivity_certification_not_derived_from_confirmed_coverage(self):
        self.confirm();self.assertFalse(self.n.book.outcomes['R_A']['normal_termination'])
        self.assertEqual(self.n.book.outcomes['R_A']['classification'],'UNVERIFIED')

if __name__=='__main__':
    unittest.main()
