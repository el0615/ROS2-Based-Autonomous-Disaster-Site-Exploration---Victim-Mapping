"""Offline tests only. No ROS publisher/action client, map edits or robot movement."""
from __future__ import annotations
import copy,gzip,hashlib,json,math,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/exploration/zone_exploration/src'),str(ROOT/'scripts/exploration'),
             str(ROOT/'scripts/exploration/navigation/return_home')]
from blocked_policy import reroute_decision,annotate_reroute,normalize_points,normalize_report
from home_state import (initial_navigation,lock_origin,validate_navigation,transition,
    InitialPoseSampler,HomeCheckpoint,fingerprint)
from reroute_core import RerouteDetector,RerouteSettings,compare_paths
from checkpoint import save_checkpoint,load_checkpoint
from mission_report import collect,pgm_png
import mission_coordinator as co

FIRST='session_20261110_131743_816532'
LAST='session_20261110_163410_816469'
CTX=dict(bundle_signature='a'*64,fingerprints={k:hashlib.sha256(k.encode()).hexdigest() for k in ('world','zones','reference','nav2')},namespace='/j100_0001')
POSE=dict(x=1.2,y=-.4,yaw=1.1)


def reroute(**kw):
    d=dict(id='route1',source='REROUTE',point_xy=[3.,4.],xy=[3.,4.],plan_change_confirmed=True,
           extra_m=3.,increase_fraction=.5,thresholds=dict(min_extra_m=3.,min_fraction=.5),
           mask_eligible=False,physical_blockage_proven=False,sensor_evidence=None)
    d.update(kw);return d


class BlockedPolicyTests(unittest.TestCase):
    def test_existing_and_boundary_blocked(self):
        self.assertEqual(annotate_reroute(reroute())['blockage_status'],'BLOCKED')
    def test_absolute_alone_not_blocked(self):self.assertFalse(reroute_decision(reroute(increase_fraction=.49))[0])
    def test_ratio_alone_not_blocked(self):self.assertFalse(reroute_decision(reroute(extra_m=2.99))[0])
    def test_no_confirmation_no_blocked(self):self.assertFalse(reroute_decision(reroute(plan_change_confirmed=False))[0])
    def test_no_extra_sensor_or_mask_or_proof_gate(self):
        r=reroute(mask_eligible=False,physical_blockage_proven=False,sensor_evidence=None)
        self.assertTrue(reroute_decision(r)[0]);self.assertEqual(normalize_points([r])[0][0]['classification'],'BLOCKED')
    def test_nonfinite_refused(self):self.assertFalse(reroute_decision(reroute(extra_m=float('nan')))[0])
    def test_no_invented_thresholds(self):self.assertFalse(reroute_decision(reroute(thresholds={}))[0])
    def test_confirm_metric_below_rule_not_blocked(self):
        self.assertFalse(reroute_decision(reroute(confirm_metric=dict(extra_m=2.,increase_fraction=.6)))[0])
    def test_full_confirmed_snapshot(self):
        raw=reroute();raw['metric']=dict(extra_m=3.5,increase_fraction=.8)
        self.assertTrue(reroute_decision(raw)[0])
    def test_pending_diagnostics_are_preserved_not_a_block(self):
        a,b=normalize_points([reroute(plan_change_confirmed=False)])
        self.assertEqual(a,[]);self.assertEqual(len(b),1)
    def test_point_no_region_inference(self):
        a,_=normalize_points([reroute()]);self.assertFalse(a[0]['zone_inaccessibility_inferred'])
    def test_physical_flag_unchanged_in_raw_evidence(self):
        r=reroute();a=annotate_reroute(r);self.assertFalse(a['physical_blockage_proven']);self.assertNotIn('blockage_status',r)
    def test_obstacle_and_reroute_unified(self):
        raw=[reroute(),dict(id='p',source='PATH_BLOCKAGE',xy=[1.,2.])]
        a,b=normalize_points(raw);self.assertEqual(len(a),2)
        self.assertEqual({x['display_label'] for x in a},{'차단지점'})
    def test_registry_nested_evidence_classified(self):
        raw=dict(xy=[3.,4.],sources=['REROUTE'],observations=[dict(source='REROUTE',evidence=reroute())])
        a,_=normalize_points([raw]);self.assertEqual(len(a),1)
    def test_journal_confirmation_classified(self):
        raw=dict(id='ev',xy=[3.,4.],source='REROUTE',evidence=dict(confirmation=reroute()))
        a,_=normalize_points([raw]);self.assertEqual(len(a),1)
    def test_report_counts_and_list_agree(self):
        obj=dict(categories=dict(blocked_points=[reroute(),reroute(plan_change_confirmed=False)]),blocked_areas=[])
        out=normalize_report(obj);self.assertEqual(out['counts']['blocked_points'],1)
        self.assertEqual(len(out['blocked_areas']),1);self.assertEqual(len(obj['categories']['blocked_points']),2)
    def test_live_detector_time_sample_gates_not_removed(self):
        d=RerouteDetector();old=[[0.,0.],[10.,0.]];new=[[0.,0.],[0.,3.],[10.,3.],[10.,0.]]
        self.assertEqual(d.feed(old,[0.,0.],1.,'g',[10.,0.])[1],'BASELINE_CAPTURED')
        e,reason=d.feed(new,[0.,0.],2.,'g',[10.,0.]);self.assertFalse(e['plan_change_confirmed'])
        e,reason=d.feed(new,[0.,0.],2.6,'g',[10.,0.]);self.assertTrue(e['plan_change_confirmed'])
        e['thresholds']=dict(min_extra_m=3.,min_fraction=.5)
        self.assertTrue(reroute_decision(e)[0])


class HomeStateTests(unittest.TestCase):
    def setUp(self):self.nav=initial_navigation(FIRST,CTX)
    def home(self):return lock_origin(self.nav,CTX,POSE,'FIRST_MISSION_FRESH_MAP_TF',1.)
    def test_new_checkpoint_no_fabricated_home(self):self.assertIsNone(self.nav['original_start_pose'])
    def test_capture_map_pose_not_spawn_or_resume(self):
        h=self.home();self.assertEqual(h['original_start_pose']['x'],1.2)
        self.assertEqual(h['original_start_pose']['source_session'],FIRST)
    def test_origin_immutable(self):
        with self.assertRaisesRegex(ValueError,'IMMUTABLE'):lock_origin(self.home(),CTX,dict(POSE,x=3.),'FIRST_MISSION_FRESH_MAP_TF')
    def test_same_pose_does_not_rewrite_timestamp(self):
        h=self.home();self.assertEqual(h,lock_origin(h,CTX,POSE,'FIRST_MISSION_FRESH_MAP_TF',88.))
    def test_other_mission_refused(self):
        with self.assertRaises(ValueError):validate_navigation(self.home(),LAST,CTX)
    def test_other_map_refused(self):
        with self.assertRaises(ValueError):validate_navigation(self.home(),FIRST,dict(CTX,namespace='other'))
    def test_bad_pose_refused(self):
        with self.assertRaises(ValueError):lock_origin(self.nav,CTX,dict(POSE,x=float('inf')),'FIRST_MISSION_FRESH_MAP_TF')
    def test_unknown_frame_refused(self):
        h=self.home();h['original_start_pose']['frame_id']='odom'
        with self.assertRaises(ValueError):validate_navigation(h,FIRST,CTX)
    def test_corrupt_origin_refused(self):
        h=self.home();h['original_start_pose']['x']=9.
        with self.assertRaises(ValueError):validate_navigation(h,FIRST,CTX)
    def test_unknown_home_cannot_return(self):
        with self.assertRaises(ValueError):transition(self.nav,CTX,'RETURN_IN_PROGRESS')
    def test_return_resume_state_preserves_home(self):
        h=self.home();a=transition(h,CTX,'RETURN_PREFLIGHT',new_attempt=True)
        a=transition(a,CTX,'RETURN_IN_PROGRESS');a=transition(a,CTX,'RETURN_FAILED_OR_INTERRUPTED')
        b=transition(a,CTX,'RETURN_PREFLIGHT',new_attempt=True)
        self.assertEqual(b['return_home']['attempts'],2);self.assertEqual(b['original_start_pose'],h['original_start_pose'])
    def test_real_checkpoint_roundtrip(self):
        payload=dict(context=CTX,mission_session_ids=[FIRST],mission={},navigation_v189=self.home())
        with tempfile.TemporaryDirectory() as t:
            save_checkpoint(t,payload);restored,_,_=load_checkpoint(Path(t)/'LATEST.json')
            self.assertEqual(restored['navigation_v189'],payload['navigation_v189'])
    def test_legacy_resume_does_not_capture_current(self):
        with tempfile.TemporaryDirectory() as t:
            c=HomeCheckpoint(t,CTX,[FIRST,LAST],dict(coverage={'source_session':'t5'}))
            c.resumed([99.,88.,0.],LAST,3.)
            self.assertIsNone(c.data['original_start_pose']);self.assertEqual(c.data['resume_start_pose']['x'],99.)
    def test_legacy_anchor_wrong_mission_not_used(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'logs/dn012_return_home';p.mkdir(parents=True)
            (p/'original_start_home_v188.json').write_text(json.dumps(dict(identity={},pose_map=POSE)))
            c=HomeCheckpoint(t,CTX,[FIRST],dict(coverage={'source_session':'t5'}))
            self.assertIsNone(c.data['original_start_pose'])
    def test_existing_v188_anchor_import_exact_lineage(self):
        from home_anchor import make_anchor
        v=dict(initial_session=FIRST,t5='t5',context=CTX)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'logs/dn012_return_home';p.mkdir(parents=True)
            (p/'original_start_home_v188.json').write_text(json.dumps(make_anchor(v,POSE,'ORIGINAL_START_MAP_POSE_OPERATOR_CONFIRMED')))
            c=HomeCheckpoint(t,CTX,[FIRST,LAST],dict(coverage={'source_session':'t5'}))
            self.assertEqual(c.data['original_start_pose']['x'],POSE['x'])
    def test_five_unique_stopped_samples(self):
        s=InitialPoseSampler()
        for i in range(4):self.assertIsNone(s.feed(POSE,1.+i,.4*i,ready=True))
        self.assertIsNotNone(s.feed(POSE,5.,1.6,ready=True))
    def test_duplicate_tf_does_not_capture(self):
        s=InitialPoseSampler()
        for i in range(10):self.assertIsNone(s.feed(POSE,1.,i,ready=True))
    def test_motion_resets_samples(self):
        s=InitialPoseSampler()
        for i in range(4):s.feed(POSE,1.+i,i,ready=True)
        self.assertIsNone(s.feed(dict(POSE,x=2.),5.,5.,ready=True))
        self.assertEqual(len(s.samples),1)
    def test_unready_resets_samples(self):
        s=InitialPoseSampler();s.feed(POSE,1.,1.,ready=True);s.feed(POSE,2.,2.,ready=False)
        self.assertEqual(s.samples,[])
    def test_multi_resume_keeps_exact_first_pose(self):
        with tempfile.TemporaryDirectory() as t:
            original=self.home();data=original
            for i in range(5):
                c=HomeCheckpoint(t,CTX,[FIRST,LAST],dict(navigation_v189=data))
                c.resumed([float(i+20),5.,.2],LAST,i+10.)
                data=c.snapshot(SimpleNamespace())
            self.assertEqual(data['original_start_pose'],original['original_start_pose'])
    def test_return_status_never_changes_zone_completion(self):
        payload=dict(mission={'zone_lifecycle_v187':{'zones':{'R04':{'exploration_status':'COMPLETE'}}}},navigation_v189=self.home())
        before=copy.deepcopy(payload['mission'])
        payload['navigation_v189']=transition(payload['navigation_v189'],CTX,'RETURN_IN_PROGRESS')
        self.assertEqual(before,payload['mission'])


class IntegrationTests(unittest.TestCase):
    def test_runtime_hooks_compile_and_import(self):
        import resume_runtime as runtime
        self.assertTrue(callable(runtime.wrap_mission));self.assertTrue(callable(runtime.capture_mission))
    def test_append_return_checkpoint_keeps_source_and_counts(self):
        with tempfile.TemporaryDirectory() as t:
            session=Path(t);original=dict(context=CTX,mission_session_ids=[FIRST],mission=dict(counters=dict(sent_goals=58)),coverage={'observed_cells':448000})
            cp,digest=save_checkpoint(session/'checkpoints_v18',original);raw=cp.read_bytes()
            nav=lock_origin(initial_navigation(FIRST,CTX),CTX,POSE,'FIRST_MISSION_FRESH_MAP_TF')
            new_digest,path=co.append_checkpoint(session,nav,digest)
            new,_,_=load_checkpoint(path)
            self.assertEqual(cp.read_bytes(),raw);self.assertEqual(new['mission'],original['mission']);self.assertEqual(new['coverage'],original['coverage'])
            self.assertEqual(new['navigation_v189']['original_start_pose']['x'],1.2)
    def test_concurrent_checkpoint_write_refused(self):
        with tempfile.TemporaryDirectory() as t:
            session=Path(t);save_checkpoint(session/'checkpoints_v18',dict(context=CTX,mission_session_ids=[FIRST]))
            with self.assertRaisesRegex(ValueError,'CONCURRENTLY'):co.append_checkpoint(session,initial_navigation(FIRST,CTX),'bad')
    def test_pgm_png_matches_dimensions(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'map.pgm';p.write_bytes(b'P5\n# hello\n2 3\n255\n'+bytes([0,255,254,0,205,205]))
            w,h,png=pgm_png(p);self.assertEqual((w,h),(2,3));self.assertTrue(png.startswith(b'\x89PNG'))
    def test_start_dry_run_has_no_extra_capture_or_return(self):
        with patch.object(co,'run_child',return_value=0) as run,patch.object(co,'coordinator_lock',return_value=__import__('contextlib').nullcontext()):
            co.main(['start','--dry-run']);self.assertTrue(run.call_args.args[1]);self.assertIn('dry-run',run.call_args.args[0])
    def test_real_log_projection_when_available(self):
        path=os.environ.get('V189_REAL_SUMMARY')
        if not path:self.skipTest('historical summary not available')
        summary=json.loads(Path(path).read_text());before=copy.deepcopy(summary)
        result=collect(summary)
        from collections import Counter
        counts=Counter(x['source'] for x in result['categories']['blocked_points'])
        self.assertEqual(counts,{'PATH_BLOCKAGE':40,'REROUTE':20})
        self.assertEqual(len(result['reroute_diagnostics_v189']),8)
        self.assertEqual(summary,before)
        self.assertEqual(result['zone_lifecycle_v187'],summary['zone_lifecycle_v187'])

if __name__=='__main__':unittest.main()
