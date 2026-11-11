import copy,gzip,hashlib,json,math,sys,tempfile,unittest
from pathlib import Path
from unittest import mock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts/exploration/navigation/return_home'))
import home_anchor as p
import home_return as run

FIRST='session_20261110_131743_816532'
LAST='session_20261110_163410_816469'
T5='session_20261110_130514_234118'
BUNDLE='a88ec3362a41e7c2c0f5af6bd27bddb15a973f359dd821e32e817e14135246f3'

def fixture(tmp):
    root=Path(tmp)
    session=root/'logs/hospital_zone_mission'/LAST
    cp=session/'checkpoints_v18'
    cp.mkdir(parents=True)
    ctx=dict(bundle_signature=BUNDLE,fingerprints={k:k*64 for k in ('world','reference','zones','nav2')})
    zones={f'R{i:02d}':dict(exploration_status='COMPLETE',phase='TERMINAL',completion_reason='SECOND_PASS_FINISHED',access_status='ACCESSIBLE') for i in range(1,27)}
    zones['CORRIDOR_E']=dict(exploration_status='COMPLETE',phase='TERMINAL',completion_reason='BOUNDED_ACCESS_REVIEW_FINISHED',access_status='INACCESSIBLE')
    payload=dict(context=ctx,coverage=dict(source_session=T5),mission_session_ids=[FIRST,LAST],
        mission=dict(source_execute=True,source_session=LAST,source_state=p.FINISHED,
        counters=dict(sent_goals=58,succeeded_goals=50),scope=list(zones),
        zone_lifecycle_v187=dict(schema='dn012_explicit_zone_lifecycle_v1_8_7',zones=zones)))
    (session/'run_manifest.json').write_text(json.dumps(dict(options=dict(execute=True),zone_bundle_signature=BUNDLE)))
    (session/'summary.json').write_text(json.dumps(dict(execute=True,state=p.FINISHED,stop_reason=None,owned_goal_id=None,zone_bundle_signature=BUNDLE,coverage_session=T5,sent=58,nav_succeeded=50)))
    raw_payload=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    sha=hashlib.sha256(raw_payload).hexdigest()
    file='checkpoint_20261110T075000_000000Z_1234abcd.json.gz'
    binary=gzip.compress(json.dumps(dict(payload=payload,payload_sha256=sha)).encode())
    (cp/file).write_bytes(binary)
    (cp/'LATEST.json').write_text(json.dumps(dict(checkpoint=file,payload_sha256=sha,sha256=hashlib.sha256(binary).hexdigest())))
    return session, payload, zones

class HomePolicyTests(unittest.TestCase):
    def setUp(self):
        self.d=tempfile.TemporaryDirectory();self.addCleanup(self.d.cleanup)
        self.session,self.payload,self.zones=fixture(self.d.name)
    def test_complete_27_zones_verified(self):
        v=p.check_completion(self.session)
        self.assertEqual(v['finished_zones'],27);self.assertEqual(v['initial_session'],FIRST)
        self.assertEqual(v['sent'],58)
    def test_incomplete_zone_blocks_return(self):
        self.zones['CORRIDOR_E']['exploration_status']='IN_PROGRESS'
        self._rewrite();self.assertRaisesRegex(ValueError,'ZONE_NOT_TERMINAL',p.check_completion,self.session)
    def test_bad_source_state_blocks_return(self):
        self.payload['mission']['source_state']='PARTIAL_DEFERRED_REMAIN'
        self._rewrite();self.assertRaisesRegex(ValueError,'TERMINAL_STATE',p.check_completion,self.session)
    def test_stale_summary_blocks_return(self):
        q=self.session/'summary.json'; d=json.loads(q.read_text());d['state']='WAIT_STOP';q.write_text(json.dumps(d))
        self.assertRaisesRegex(ValueError,'NORMALLY_COMPLETE',p.check_completion,self.session)
    def test_dry_run_blocks_return(self):
        q=self.session/'run_manifest.json'; d=json.loads(q.read_text());d['options']['execute']=False;q.write_text(json.dumps(d))
        self.assertRaisesRegex(ValueError,'EXECUTE_NOT_DRY_RUN',p.check_completion,self.session)
    def test_mismatch_lineage_blocks_return(self):
        self.payload['mission_session_ids']=[LAST]
        self._rewrite();v=p.check_completion(self.session)
        a=p.make_anchor(dict(v,initial_session=FIRST),dict(x=1.,y=2.,yaw=0.),'ORIGINAL_START_MAP_POSE_OPERATOR_CONFIRMED')
        self.assertRaisesRegex(ValueError,'OTHER_MISSION',p.validate_anchor,a,v)
    def test_original_home_different_from_resume(self):
        v=p.check_completion(self.session)
        a=p.make_anchor(v,dict(x=1.4,y=20.6,yaw=-1.5),'ORIGINAL_START_MAP_POSE_OPERATOR_CONFIRMED')
        self.assertEqual(p.validate_anchor(a,v)['x'],1.4)
        self.assertNotEqual(p.validate_anchor(a,v)['x'],32.803)
        self.assertTrue(a['not_resume_start'])
    def test_map_changed_refuses_anchor(self):
        v=p.check_completion(self.session)
        a=p.make_anchor(v,dict(x=1.,y=2.,yaw=0.),'ORIGINAL_START_MAP_POSE_OPERATOR_CONFIRMED')
        other=copy.deepcopy(v);other['context']['fingerprints']['world']='X'*64
        self.assertRaisesRegex(ValueError,'OTHER_MISSION_OR_MAP',p.validate_anchor,a,other)
    def test_no_fabricated_legacy_home(self):
        self.assertFalse(hasattr(self.payload['mission'],'home_pose_map'))
    def test_corrupt_pointer_refused(self):
        q=self.session/'checkpoints_v18/LATEST.json';d=json.loads(q.read_text());d['sha256']='0'*64;q.write_text(json.dumps(d))
        self.assertRaisesRegex(ValueError,'FILE_SHA_MISMATCH',p.check_completion,self.session)
    def test_corrupt_payload_digest_refused(self):
        cp=self.session/'checkpoints_v18';q=cp/'LATEST.json';d=json.loads(q.read_text());f=cp/d['checkpoint'];obj=json.loads(gzip.decompress(f.read_bytes()));obj['payload']['mission']['source_state']='FAKE';f.write_bytes(gzip.compress(json.dumps(obj).encode()));d['sha256']=hashlib.sha256(f.read_bytes()).hexdigest();q.write_text(json.dumps(d))
        self.assertRaisesRegex(ValueError,'PAYLOAD_SHA',p.check_completion,self.session)
    def test_invalid_home_coordinates(self):
        v=p.check_completion(self.session)
        with self.assertRaisesRegex(ValueError,'INVALID'):p.make_anchor(v,dict(x=float('nan'),y=0.,yaw=0.),'ORIGINAL_START_MAP_POSE_OPERATOR_CONFIRMED')
    def test_old_DN011_hash_guards(self):
        self.assertEqual(len(run.DN011_HASHES),2)
    def test_first_home_write_immutable(self):
        path=Path(self.d.name)/'anchor.json'
        run.atomic_json(path,dict(x=1),immutable=True)
        self.assertRaises(FileExistsError,run.atomic_json,path,dict(x=2),True)
    def test_resume_dry_run_never_autoreturn(self):
        class Child:
            def __init__(self):self.stdout=iter(['[SAVED] '+str(self.session) if False else 'DRY_RUN: no NavigateToPose was requested\n']);self.returncode=0
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def wait(self):return 0
        with mock.patch.object(run.subprocess,'Popen',return_value=Child()):
            with mock.patch.object(run,'perform_return') as f:
                self.assertEqual(run.resume(['--dry-run']),0);f.assert_not_called()
    def test_resume_finished_dispatches_one_return(self):
        session=str(self.session)
        class Child:
            stdout=iter(['[STATE] ALL_ZONES_EXPLORATION_COMPLETE\n',
                          '[SAVED] '+session+'\n'])
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def wait(self):return 0
        with mock.patch.object(run.subprocess,'Popen',return_value=Child()):
            with mock.patch.object(run,'perform_return') as f:
                with mock.patch.object(run,'session_info',return_value=p.check_completion(self.session)) as v:
                    self.assertEqual(run.resume([]),0)
                    f.assert_called_once();v.assert_called_once_with(session)
    def test_resume_failed_exit_never_returns(self):
        class Child:
            stdout=iter(['[STATE] STOPPED_WITHOUT_COMPLETION\n'])
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def wait(self):return 2
        with mock.patch.object(run.subprocess,'Popen',return_value=Child()):
            with mock.patch.object(run,'perform_return') as f:
                self.assertEqual(run.resume([]),2);f.assert_not_called()
    def test_missing_original_home_does_not_request_goal(self):
        v=p.check_completion(self.session)
        with mock.patch.object(run,'ANCHOR',Path(self.d.name)/'missing_home.json'), \
             mock.patch.object(run,'LOG',Path(self.d.name)/'return_logs'), \
             mock.patch.object(run,'load_supervisor') as f:
            self.assertRaisesRegex(ValueError,'ORIGINAL_HOME_NOT_LOCKED',run.perform_return,v)
            f.assert_not_called()
            self.assertEqual(len(list((Path(self.d.name)/'return_logs').rglob('report.json'))),1)
    def test_bad_anchor_quaternion_refused(self):
        v=p.check_completion(self.session)
        a=p.make_anchor(v,dict(x=1.,y=2.,yaw=1.0),'ORIGINAL_START_MAP_POSE_OPERATOR_CONFIRMED')
        a['pose_map']['yaw']=0.0
        self.assertRaisesRegex(ValueError,'QUAT_INVALID',p.validate_anchor,a,v)
    def _rewrite(self):
        cp=self.session/'checkpoints_v18';q=cp/'LATEST.json';d=json.loads(q.read_text());f=cp/d['checkpoint']
        raw=json.dumps(self.payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        digest=hashlib.sha256(raw).hexdigest();z=gzip.compress(json.dumps(dict(payload=self.payload,payload_sha256=digest)).encode());f.write_bytes(z)
        d.update(sha256=hashlib.sha256(z).hexdigest(),payload_sha256=digest);q.write_text(json.dumps(d))

if __name__=='__main__':unittest.main()
