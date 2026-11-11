"""Mock integration of actual v189 hooks; never imports ROS runtime."""
from __future__ import annotations
import copy,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/exploration/zone_exploration/src'),str(ROOT/'scripts/exploration'),str(ROOT/'scripts/exploration/navigation/return_home')]
from home_state import HomeCheckpoint,initial_navigation,lock_origin
import mission_coordinator as co
FIRST='session_20261110_131743_816532'
CTX=dict(bundle_signature='a'*64,fingerprints={k:k*64 for k in ('world','zones','reference','nav2')})
POSE=dict(x=1.,y=2.,yaw=.4)


class CaptureRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.home=HomeCheckpoint(self.temp.name,CTX,[FIRST])
        self.now=1.;self.wall=100.;self.saved=[]
        self.node=SimpleNamespace(stop_requested=None,sent_goals=0,last_clock=0.,clock_advanced_wall=100.,
             _v183_action_seen=True,foreign_active=lambda:False,local_start_clear=lambda inp:True,
             stop_evidence=SimpleNamespace(ready=lambda a,b:True),ros_s=lambda:self.now,
             info=Mock(),event=Mock(),tf_buffer=SimpleNamespace(lookup_transform=self.tf))
        self.node.inputs=lambda:(None,dict(pose=[1.,2.,.4]))
        self.c=SimpleNamespace(n=self.node,args=SimpleNamespace(execute=True,robot_frame='base_link'),
            resume_started_wall=100.,write_failed=False,last_saved=None,dirty=False)
        self.node.publish_report=self.publish
        tm=ModuleType('rclpy.time');tm.Time=lambda:None
        du=ModuleType('rclpy.duration');du.Duration=lambda **kw:None
        self.modules=patch.dict(sys.modules,{'rclpy.time':tm,'rclpy.duration':du});self.modules.start();self.addCleanup(self.modules.stop)
        self.clock=patch('home_state.time_monotonic',lambda:self.wall);self.clock.start();self.addCleanup(self.clock.stop)
    def tf(self,*args,**kwargs):
        return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=int(self.now),nanosec=0)))
    def publish(self,force=False):
        self.saved.append(copy.deepcopy(self.home.data));self.c.last_saved='new_cp'
    def feed(self):
        result=self.home.before_initial_poll(self.c);self.now+=1.;self.wall+=.4;return result
    def test_initial_home_checkpoint_before_first_goal(self):
        # First already-validated map pose is stored; no extra 5-sample gate.
        self.assertTrue(self.feed());self.assertTrue(self.home.capture_persisted)
        self.assertEqual(len(self.saved),1);self.assertIsNotNone(self.saved[0]['original_start_pose'])
        self.assertEqual(self.saved[0]['original_start_pose']['x'],1.)
        self.assertEqual(self.node.sent_goals,0)
    def test_first_origin_not_recaptured(self):
        self.assertTrue(self.feed())
        before=copy.deepcopy(self.home.data['original_start_pose'])
        self.node.inputs=lambda:(None,dict(pose=[50.,50.,.5]))
        for i in range(4):self.assertTrue(self.feed())
        self.assertEqual(before,self.home.data['original_start_pose']);self.assertEqual(len(self.saved),1)
    def test_dry_run_does_not_set_origin(self):
        self.c.args.execute=False;self.assertTrue(self.feed())
        self.assertEqual(self.saved,[]);self.assertIsNone(self.home.data['original_start_pose'])
    def test_resume_never_recaptures_original(self):
        self.home=HomeCheckpoint(self.temp.name,CTX,[FIRST],dict(coverage={'source_session':'t5'}))
        self.assertTrue(self.feed());self.assertIsNone(self.home.data['original_start_pose'])
    def test_failed_write_blocks_first_nav(self):
        def fail(**kw):self.c.write_failed=True
        self.node.publish_report=fail
        with self.assertRaisesRegex(ValueError,'WRITE_FAILED'):self.feed()
        self.assertFalse(self.home.capture_persisted)
    def test_existing_goals_without_home_refused(self):
        self.node.sent_goals=1
        with self.assertRaisesRegex(ValueError,'AFTER_FIRST_NAV'):self.feed()
    def test_home_capture_does_not_duplicate_nav_status_gate(self):
        self.node._v183_action_seen=False
        self.assertTrue(self.feed())
        self.assertEqual(len(self.saved),1)
    def test_home_capture_does_not_duplicate_stop_or_footprint_gates(self):
        self.node.stop_evidence.ready=lambda a,b:False
        self.node.local_start_clear=lambda inp:False
        self.assertTrue(self.feed())
        self.assertEqual(len(self.saved),1)
    def test_home_capture_does_not_duplicate_tf_lookup(self):
        self.node.tf_buffer.lookup_transform=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('duplicate TF lookup'))
        # The already-validated input's map pose is the single source of HOME.
        self.assertTrue(self.feed())
        self.assertEqual(len(self.saved),1)
    def test_stale_input_blocks_capture(self):
        self.node.inputs=lambda:('STALE_GLOBAL_COSTMAP',None)
        for i in range(6):self.assertFalse(self.feed())
        self.assertEqual(self.saved,[])
    def test_wait_for_inputs_does_not_add_new_60s_abort(self):
        self.wall=300.
        self.node.inputs=lambda:('WAIT_GLOBAL_COSTMAP',None)
        self.assertFalse(self.feed());self.assertEqual(self.saved,[])
    def test_invalid_map_pose_cannot_be_saved(self):
        self.node.inputs=lambda:(None,dict(pose=[float('nan'),2.,.4]))
        with self.assertRaisesRegex(ValueError,'HOME_POSE_NONFINITE'):self.feed()
        self.assertEqual(self.saved,[])
    def test_clock_reset_still_refused(self):
        self.node.last_clock=100.
        with self.assertRaisesRegex(ValueError,'CLOCK_RESET_BEFORE_FIRST_START'):self.feed()


class ReturnRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.session=self.root/'logs/hospital_zone_mission'/FIRST
        self.home=lock_origin(initial_navigation(FIRST,CTX),CTX,POSE,'FIRST_MISSION_FRESH_MAP_TF')
        self.original=copy.deepcopy(self.home['original_start_pose'])
        self.payload=dict(context=CTX,mission_session_ids=[FIRST],navigation_v189=self.home)
        self.snapshots=[]
        self.n=Mock();self.n.tf_pose.return_value=dict(x=40.,y=5.,yaw=.3)
        self.n.plan_home.return_value=dict(already_home=False,poses=10)
        self.n.nav_home.side_effect=self.nav
        self.rc=Mock();self.rc.ok.return_value=False
        self.legacy=Mock();self.legacy.HomeSupervisor.return_value=self.n
        self.patches=[patch.object(co,'ROOT',self.root),patch.object(co,'check_completion',return_value={}),
            patch.object(co,'verify_disk_identity'),patch.object(co,'state_from_session',side_effect=lambda s:(self.payload,'sha',copy.deepcopy(self.home))),
            patch.object(co,'append_checkpoint',side_effect=self.append),patch.object(co,'load_supervisor',return_value=(self.rc,self.legacy)),
            patch.object(co,'ros_init')]
        for p in self.patches:p.start();self.addCleanup(p.stop)
    def append(self,session,nav,digest):
        self.snapshots.append(copy.deepcopy(nav));return 'new','checkpoint'
    def nav(self,pose,timeout):
        self.assertEqual(self.snapshots[-1]['return_home']['status'],'RETURN_IN_PROGRESS')
        self.assertEqual(pose['x'],POSE['x']);self.assertNotEqual(pose['x'],40.)
        self.n.tf_pose.return_value=dict(POSE)
        return dict(arrival_xy_error_m=.02)
    def test_return_goal_is_original_not_resume(self):
        self.assertEqual(co._perform_return(self.session),0);self.n.nav_home.assert_called_once()
        self.assertEqual(self.snapshots[-1]['return_home']['status'],'HOME_REACHED')
        self.assertTrue(all(s['original_start_pose']==self.original for s in self.snapshots))
    def test_return_in_progress_saved_before_navigation(self):
        self.assertEqual(co._perform_return(self.session),0)
        self.assertEqual([s['return_home']['status'] for s in self.snapshots],
            ['RETURN_PREFLIGHT','RETURN_PLAN_OK','RETURN_IN_PROGRESS','HOME_REACHED'])
    def test_plan_only_sends_no_goal_and_does_not_write_checkpoint(self):
        self.assertEqual(co._perform_return(self.session,plan_only=True),0)
        self.n.nav_home.assert_not_called();self.assertEqual(self.snapshots,[])
    def test_missing_home_sends_no_goal(self):
        self.home=initial_navigation(FIRST,CTX)
        self.assertEqual(co._perform_return(self.session),2);self.n.nav_home.assert_not_called()
        co.load_supervisor.assert_not_called()
        self.assertEqual(self.snapshots[-1]['return_home']['status'],'ORIGINAL_START_MISSING')
    def test_already_home_sends_no_goal(self):
        self.n.plan_home.return_value=dict(already_home=True,poses=0)
        self.assertEqual(co._perform_return(self.session),0);self.n.nav_home.assert_not_called()
    def test_return_failure_keeps_home_and_records_state(self):
        self.n.nav_home.side_effect=RuntimeError('blocked')
        self.assertEqual(co._perform_return(self.session),2)
        self.assertEqual(self.snapshots[-1]['return_home']['status'],'RETURN_FAILED_OR_INTERRUPTED')
        self.assertEqual(self.snapshots[-1]['original_start_pose'],self.original)
    def test_plan_failure_sends_no_goal(self):
        self.n.plan_home.side_effect=RuntimeError('no route')
        self.assertEqual(co._perform_return(self.session),2);self.n.nav_home.assert_not_called()
    def test_checkpoint_write_failure_before_send_blocks_goal(self):
        co.append_checkpoint.side_effect=OSError('disk full')
        self.assertEqual(co._perform_return(self.session),2);self.n.nav_home.assert_not_called()
    def test_return_after_interrupt_retries_original_home_only(self):
        self.home['return_home']['status']='RETURN_IN_PROGRESS'
        self.home['return_home']['attempts']=1
        self.assertEqual(co._perform_return(self.session),0)
        self.assertEqual(self.snapshots[-1]['return_home']['attempts'],2)
        self.assertEqual(self.snapshots[-1]['original_start_pose'],self.original)

if __name__=='__main__':unittest.main()
