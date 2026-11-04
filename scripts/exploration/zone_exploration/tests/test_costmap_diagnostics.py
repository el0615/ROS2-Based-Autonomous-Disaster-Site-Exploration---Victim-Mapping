"""Synthetic audit of diagnostic-only behavior. No ROS network or navigation."""
from fixtures import *
from pathlib import Path
import json, tempfile, time, unittest, os
from types import SimpleNamespace as NS
import fake_ros
fake_ros.install()
os.environ['ZB_PROJECT_ROOT']='/synthetic'
import zone_node
from costmap_diagnostics import snapshot_costmap_freshness
from zb_reference_planning import Settings


class CostmapSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.mon={'received_count':9,'accepted_count':9,'last_receive_wall':98.,
                  'last_valid_receive_wall':98.,'last_header_stamp_s':200.,
                  'last_interarrival_wall_s':2.,'max_interarrival_wall_s':2.5,
                  'invalid_geometry_count':0,'invalid_values_count':0}
    def test_within_limits(self):
        d=snapshot_costmap_freshness({'stamp':200.,'wall':98.},self.mon,201.7,100.)
        self.assertEqual(d['cause'],'WITHIN_LIMITS')
        self.assertEqual(d['ros_age_s'],1.7)
        self.assertFalse(d['ros_stale'])
    def test_ros_old_but_wall_fresh(self):
        d=snapshot_costmap_freshness({'stamp':190.,'wall':99.},self.mon,200.,100.)
        self.assertEqual(d['cause'],'ROS_HEADER_STALE')
        self.assertTrue(d['ros_stale']);self.assertFalse(d['wall_stale'])
    def test_wall_old_but_ros_fresh(self):
        d=snapshot_costmap_freshness({'stamp':200.,'wall':80.},self.mon,201.,100.)
        self.assertEqual(d['cause'],'NO_RECENT_VALID_RECEIPT')
        self.assertTrue(d['wall_stale']);self.assertFalse(d['ros_stale'])
    def test_both_old(self):
        self.assertEqual(snapshot_costmap_freshness({'stamp':190.,'wall':70.},self.mon,201.,100.)['cause'],'ROS_AND_WALL_STALE')
    def test_future_stamp(self):
        self.assertEqual(snapshot_costmap_freshness({'stamp':201.,'wall':98.},self.mon,200.,100.)['cause'],'HEADER_IN_FUTURE')
    def test_no_costmap(self):
        d=snapshot_costmap_freshness(None,self.mon,201.,100.)
        self.assertEqual(d['cause'],'NO_VALID_COSTMAP')
        self.assertIsNone(d['ros_age_s']);self.assertIsNone(d['wall_age_s'])
    def test_exact_thresholds_preserved(self):
        for r,w in [(-.1,15.),(5.,15.)]:
            d=snapshot_costmap_freshness({'stamp':200.-r,'wall':100.-w},self.mon,200.,100.)
            self.assertEqual(d['cause'],'WITHIN_LIMITS')


class TransportDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.b=bundle()
        a=zone_node.parser().parse_args(['--visibility-backend','python'])
        self.node=zone_node.make_node_type(a,self.b,Settings(),Path(self.tmp.name))()
        self.node.reference_ok=True
        self.node.matched={'status':status(self.b,t=10.),'stamp':10.,'wall':time.monotonic(),'observed':self.b[3].copy()}
    def tearDown(self):
        self.node.close();self.tmp.cleanup()
    def rows(self,event):
        path=Path(self.tmp.name)/'events.jsonl'
        return [json.loads(s) for s in path.read_text().splitlines() if json.loads(s).get('event')==event]
    def test_stale_in_inputs_keeps_same_reason_and_once_per_episode(self):
        self.node.costmap_entry={'stamp':4.,'wall':time.monotonic(),'values':None,'key':'fake'}
        for _ in range(10):
            error,_=self.node.inputs()
            self.assertEqual(error,'STALE_GLOBAL_COSTMAP')
        rows=self.rows('GLOBAL_COSTMAP_STALE_DIAG')
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['cause'],'ROS_HEADER_STALE')
        self.assertEqual(len(self.rows('GLOBAL_COSTMAP_CANCEL_DIAG')),0)
    def test_recover_resets_episode(self):
        now=time.monotonic()
        self.node.costmap_entry={'stamp':4.,'wall':now,'values':None,'key':'fake'}
        self.node.inputs()
        self.node.costmap_entry={'stamp':10.,'wall':time.monotonic(),'values':None,'key':'fake'}
        self.node.inputs()  # will continue to other inputs; sufficient to mark recovery
        self.assertEqual(len(self.rows('GLOBAL_COSTMAP_RECOVERED_DIAG')),1)
        self.node.costmap_entry={'stamp':4.,'wall':time.monotonic(),'values':None,'key':'fake'}
        self.node.inputs()
        self.assertEqual(len(self.rows('GLOBAL_COSTMAP_STALE_DIAG')),2)
    def test_cancel_event_does_not_change_cancel_policy(self):
        self.node.costmap_entry={'stamp':4.,'wall':time.monotonic(),'values':None,'key':'fake'}
        self.node.inputs()
        # No action, just diagnostic stop event to avoid fake ROS navigation side effects.
        self.node.request_stop('DATA_LOSS_STALE_GLOBAL_COSTMAP')
        self.assertEqual(self.node.stop_requested,'DATA_LOSS_STALE_GLOBAL_COSTMAP')
        d=self.rows('GLOBAL_COSTMAP_CANCEL_DIAG')
        self.assertEqual(len(d),1)
        self.assertEqual(d[0]['cause'],'ROS_HEADER_STALE')
        self.assertEqual(len(self.node.nav_client.sent),0)
    def test_normal_stale_cancel_delay_unchanged(self):
        self.node.costmap_entry={'stamp':4.,'wall':time.monotonic(),'values':None,'key':'fake'}
        self.node.nav_uuid='a'*32
        self.node.nav_terminal=None
        self.node.poll_actions=lambda inp:None
        self.node.poll()
        self.assertIsNone(self.node.stop_requested)  # first detection, 2s grace remains
        self.node.bad_since=time.monotonic()-2.1
        self.node.poll()
        self.assertEqual(self.node.stop_requested,'DATA_LOSS_STALE_GLOBAL_COSTMAP')
        self.assertEqual(len(self.rows('GLOBAL_COSTMAP_CANCEL_DIAG')),1)
    def test_raw_received_counter_and_gap(self):
        # Stub ONLY geometry check to feed synthetic costmap values to the callback.
        old=zone_node.geometry_matches
        try:
            zone_node.geometry_matches=lambda *_:True
            msg=NS(header=NS(stamp=NS(sec=10,nanosec=0)),data=self.b[3].astype(int).ravel().tolist())
            self.node.on_costmap(msg)
            self.node.on_costmap(msg)
            m=self.node.costmap_monitor
            self.assertEqual(m['received_count'],2)
            self.assertEqual(m['accepted_count'],2)
            self.assertIsNotNone(m['last_interarrival_wall_s'])
        finally:
            zone_node.geometry_matches=old

if __name__=='__main__':unittest.main()
