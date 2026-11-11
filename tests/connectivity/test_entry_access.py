"""DN-012 v2 blocked-entry regression. Fake ROS; no real robot, DDS or Gazebo."""
from __future__ import annotations
import math, os, sys, tempfile, time, unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT/'scripts/exploration/zone_exploration/tests',
          ROOT/'scripts/exploration/zone_exploration/src',
          ROOT/'scripts/exploration/navigation/return_home'):
    sys.path.insert(0,str(p))
import fixtures as fx
import fake_ros
fake_ros.install()
os.environ['ZB_PROJECT_ROOT']=str(ROOT)
from planning_core import Settings
from connectivity_audit import sensor_signature
from connectivity_audit import EntryAccessSettings,entry_probe_sites
import zone_node


def pos(zid,x,y,comp=1):
    pid=f'{zid}:{x}:{y}'
    probe=dict(id='ENTRY_TEST_'+pid,zone_id=zid,goal_kind='OBSERVE',
        x=float(x),y=float(y),yaw=0.,new_cells=0,new_area_m2=0.,
        visible_indices=np.empty(0,dtype=np.int32),new_indices=np.empty(0,dtype=np.int32),
        path_est_m=1.,audit_observation=True,audit_position_id=pid,
        audit_component=comp)
    return dict(position_id=pid,component=comp,probe=probe)


class EntrySamplingTests(unittest.TestCase):
    def test_avoid_repeat_yaw_at_same_blocked_entrance(self):
        settings=EntryAccessSettings(max_probes=3,per_component=2,min_separation_m=.8,failed_radius_m=.5)
        cat=dict(catalog_complete=True,positions=[pos('R14_T',4.26,3.53),pos('R14_T',4.36,3.53),
                                                   pos('R14_T',3.66,2.28),pos('R14_T',3.66,3.53)])
        out=entry_probe_sites(cat,[dict(x=4.26,y=3.53)],settings)
        self.assertTrue(out)
        self.assertTrue(all(math.hypot(p['probe']['x']-4.26,p['probe']['y']-3.53)>=.5 for p in out))
        self.assertLessEqual(len(out),2)
    def test_different_local_components_each_get_a_chance(self):
        cat=dict(catalog_complete=True,positions=[pos('Z',1.,1.,1),pos('Z',2.,1.,1),
                                                   pos('Z',4.,2.,2),pos('Z',5.,2.,2)])
        out=entry_probe_sites(cat,[],EntryAccessSettings(max_probes=2,per_component=2))
        self.assertEqual({p['component'] for p in out},{1,2})
    def test_bounded_even_many_positions(self):
        cat=dict(catalog_complete=True,positions=[pos('Z',x*1.,3.,1) for x in range(80)])
        out=entry_probe_sites(cat,[],EntryAccessSettings(max_probes=6,per_component=3))
        self.assertLessEqual(len(out),3)
        self.assertEqual(len({p['position_id'] for p in out}),len(out))
    def test_incomplete_catalog_cannot_claim_exhaustive(self):
        with self.assertRaisesRegex(ValueError,'INCOMPLETE_CATALOG'):
            entry_probe_sites(dict(catalog_complete=False,positions=[]),[],EntryAccessSettings())
    def test_invalid_bounds_rejected(self):
        for o in [EntryAccessSettings(max_probes=0),EntryAccessSettings(min_separation_m=0),
                  EntryAccessSettings(per_component=0),EntryAccessSettings(failed_radius_m=float('nan'))]:
            with self.assertRaises(ValueError):o.validate()
    def test_no_sites_is_not_accessible_or_unreachable_proof(self):
        cat=dict(catalog_complete=True,positions=[pos('Z',4.3,3.5,1)])
        self.assertEqual(entry_probe_sites(cat,[dict(x=4.3,y=3.5)],EntryAccessSettings()),[])


class EntryRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.b=fx.bundle();self.s=fx.snapshot(self.b)
        self.args=zone_node.parser().parse_args(['--connectivity-aware','--visibility-backend','python',
              '--threshold','.96','--entry-access-max-probes','2','--entry-access-per-component','2'])
        self.n=zone_node.make_node_type(self.args,self.b,Settings(threshold=.96),Path(self.tmp.name))()
        self.n.inputs=lambda:(None,self.s)
        self.n.stop_evidence.ready=lambda *a:True
        self.n.local_start_clear=lambda *a:True
        self.n.book.update(self.s['status']);self.n.book.context('grid1',time.monotonic())
        self.n.boot_wall-=5
        self.rt=self.n.connectivity
        self.zid='R_A'
        self.n.book.activate(self.zid)
        self.n.book.audit_requested[self.zid]='ENTRY_NAV_ABORTED_ACCESS_RECHECK'
        self.rt.note_entry_abort(dict(zone_id=self.zid,goal_kind='ENTRY',x=4.26,y=3.53,yaw=0.,id='failed_real_nav'))
        self.rt.start_zone(self.zid)
        self.rt.catalog=dict(zone_id=self.zid,cost_key='grid1',epoch=1,
             sensor_signature=sensor_signature(self.s['sensor']),start_pose=list(self.s['pose']),
             catalog_complete=True,positions=[pos(self.zid,3.4,2.1),pos(self.zid,5.1,3.5)])
    def tearDown(self):
        self.n.close();self.tmp.cleanup()
    def settle_plan(self,status=6):
        self.n.poll_actions(self.s)
        self.n.plan_client.handles[-1].result_future.set(NS(status=status,result=NS(path=None)))
        self.n.poll_actions(self.s)
    def test_blocked_entry_reopens_only_after_new_evidence_once(self):
        book=self.n.book
        self.rt.finish_partial('ENTRY_ACCESS_UNVERIFIED_ALTERNATES_EXHAUSTED')
        key=book.context_key
        self.assertEqual(book.reopen_changed(),[])
        book.context('new-map',book.wall+60)
        self.assertEqual(book.reopen_changed(),[])  # no subsequent progress
        book.progress+=1
        self.assertEqual(book.reopen_changed(),[self.zid])
        self.assertEqual(book.reopen_changed(),[])
        book.finish_partial(self.zid,'ENTRY_ACCESS_UNVERIFIED_ALTERNATES_EXHAUSTED')
        book.context('another-map',book.wall+60)
        book.progress+=1
        self.assertEqual(book.reopen_changed(),[])  # bounded one revisit

    def test_entry_abort_is_a_zone_lock_audit(self):
        self.assertTrue(self.rt.entry_mode)
        self.assertEqual(self.n.book.active_zone,self.zid)
    def test_blocked_entry_uses_planning_only_and_defers_zone(self):
        # Mirrors R14_T: initial confirmed-stop NAV ABORTED, repeated NAVFN_ABORTED.
        for i in range(2):
            self.assertTrue(self.rt.step(self.s))
            self.assertTrue(self.n.current.get('connectivity_probe'))
            self.assertTrue(self.n.current.get('entry_access_probe'))
            self.assertEqual(self.n.plan_attempts,0)  # probes do not spend global budget
            self.assertEqual(self.n.nav_client.sent,[])  # no blind robot motion
            self.settle_plan(status=6)
        self.assertTrue(self.rt.step(self.s))
        self.assertEqual(self.n.book.records[self.zid].state,'PARTIAL_DEFERRED')
        self.assertEqual(self.n.book.records[self.zid].deferred_reason,
                         'ENTRY_ACCESS_UNVERIFIED_ALTERNATES_EXHAUSTED')
        self.assertFalse(self.n.book.outcomes[self.zid]['normal_termination'])
        self.assertIsNone(self.n.book.active_zone)
        self.assertFalse(self.n.done)
        self.assertIn('R_B',[z['id'] for z in self.n.book.eligible()])
        self.assertFalse(self.n.book.all_resolved())
        self.assertEqual(self.n.nav_client.sent,[])
    def test_good_probe_replans_before_any_real_goal(self):
        # Use an actual safely generated baseline ENTRY candidate, not a guessed
        # geometry. A good planning-only result must lead to fresh preflight.
        zone=next(z for z in self.b[0]['zones'] if z['id']==self.zid)
        candidates=self.n.zone_planner.entry_batch(self.s,[zone],[])['candidates']
        self.assertTrue(candidates)
        c=dict(candidates[0],goal_kind='ENTRY',audit_observation=True,
               audit_position_id='confirmed-entry-site')
        self.rt.entry_sites=[dict(position_id='confirmed-entry-site',component=1,probe=c)]
        self.rt.entry_index=0
        self.assertTrue(self.rt.step(self.s))
        self.assertTrue(self.n.current.get('connectivity_probe'))
        self.n.poll_actions(self.s)
        poses=[]
        for x,y in (self.s['pose'][:2],(c['x'],c['y'])):
            p=fake_ros.PoseStamped();p.header.frame_id='map';p.pose.position.x=x;p.pose.position.y=y
            poses.append(p)
        self.n.plan_client.handles[-1].result_future.set(NS(status=4,result=NS(
            path=NS(header=NS(frame_id='map'),poses=poses))))
        self.n.poll_actions(self.s)
        self.assertIsNotNone(self.rt.entry_verified_goal)
        self.assertEqual(self.n.nav_client.sent,[])
        self.assertTrue(self.rt.step(self.s))
        self.assertEqual(self.n.current['goal_kind'],'ENTRY')
        self.assertFalse(self.n.current.get('connectivity_probe',False))
        self.assertTrue(self.n.current.get('entry_access_recovery'))
        self.assertEqual(self.n.plan_attempts,1)
        self.assertEqual(self.n.nav_client.sent,[])
    def test_successful_confirmed_entry_releases_audit(self):
        self.rt.entry_reached(self.zid)
        self.assertNotIn(self.zid,self.n.book.audit_requested)
        self.assertIsNone(self.rt.zid)
        self.assertFalse(self.rt.entry_mode)
    def test_stale_local_start_stops_before_probe(self):
        self.n.local_start_clear=lambda *a:False
        self.rt.step(self.s)
        self.assertEqual(self.n.stop_requested,'LOCAL_START_NOT_CLEAR_OR_STALE')
        self.assertEqual(self.n.nav_client.sent,[])


if __name__=='__main__':unittest.main()
