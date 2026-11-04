#!/usr/bin/env python3
"""Entrance override is navigation-only and fail-closed; no real ROS used."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/exploration'))
from hospital_coverage_core import load_bundle
from entrance_navigation_overlay import apply_entrance_overlay
from observation_planning_core import Settings, conservative_safe, oriented_footprint_clear, conservative_start_bridge

class EntranceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.geom,cls.labels,cls.free,cls.signature=load_bundle(ROOT/'config/exploration/hospital_zones.json',ROOT/'config/exploration/hospital_coverage_reference.json.gz')
        cls.path=ROOT/'config/exploration/entrance_nav_only_overlay_v1.json'
        cls.pose=(.302,.003,.007)
        cls.cost=np.zeros(cls.free.shape,dtype=np.int16)

    def apply(self,cost=None,pose=None,free=None,labels=None,path=None):
        return apply_entrance_overlay(self.free if free is None else free,
            self.labels if labels is None else labels,self.geom,self.cost if cost is None else cost,
            self.pose if pose is None else pose,self.path if path is None else path)

    def test_reference_start_blocked_prior(self):
        self.assertFalse(oriented_footprint_clear(self.free,self.geom,self.pose,Settings()))

    def test_single_small_entrance_override_restores_start_footprint(self):
        nav,info=self.apply()
        self.assertEqual(info['applied_cells'],81)
        self.assertTrue(oriented_footprint_clear(nav,self.geom,self.pose,Settings()))
        safe,_=conservative_safe(nav,self.geom,Settings())
        self.assertIsNotNone(conservative_start_bridge(nav,safe,self.geom,self.pose,Settings()))

    def test_original_coverage_and_free_untouched(self):
        prior=self.free.copy();zones=self.labels.copy();nav,info=self.apply()
        self.assertTrue(np.array_equal(prior,self.free))
        self.assertTrue(np.array_equal(zones,self.labels))
        self.assertTrue(np.all(self.labels[nav & ~prior]==0))
        self.assertEqual(np.sum(nav & ~prior),81)

    def test_costmap_blocks_unknown_or_inflated_cells(self):
        for value in (-1,50,99,100):
            with self.subTest(value=value):
                cost=self.cost.copy();cost[278,11]=value
                nav,info=self.apply(cost=cost)
                self.assertEqual(info['applied_cells'],80)
                self.assertFalse(oriented_footprint_clear(nav,self.geom,self.pose,Settings()))

    def test_pose_outside_entrance_unchanged(self):
        nav,info=self.apply(pose=(3.66,3.53,0.))
        self.assertFalse(info['at_entrance'])
        self.assertTrue(np.array_equal(nav,self.free))

    def test_reference_change_fails_closed(self):
        change=self.free.copy();change[278,17]=False
        with self.assertRaisesRegex(ValueError,'ENTRANCE_REFERENCE_MAP_MISMATCH'):
            self.apply(free=change)

    def test_zone_mask_change_fails_closed(self):
        change=self.labels.copy();change[278,11]=1
        with self.assertRaisesRegex(ValueError,'ENTRANCE_OVERRIDE_TOUCHES_ZONE'):
            self.apply(labels=change)

    def test_unexpected_config_fails_closed(self):
        import tempfile
        config=json.loads(self.path.read_text());config['cell_count']=88
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json';p.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'ENTRANCE_APPROVED_CELL_COUNT_CHANGED'):
                self.apply(path=p)

    def test_build_batch_preserves_zone_coverage_but_uses_nav_overlay(self):
        from observation_sequence_core import build_batch
        from observation_planning_core import SensorModel
        # Synthetic scan geometry; no actual robot test.
        snapshot=dict(pose=self.pose,observed=np.zeros_like(self.free,dtype=bool),
            cost=dict(values=self.cost),sensor=SensorModel(-3.14,.02,314,.2,8.))
        zone=next(z for z in self.data['zones'] if z['id']=='R14_T')
        result=build_batch((self.data,self.geom,self.labels,self.free,self.signature),snapshot,
            [zone],Settings(max_candidates=5),[],shortlist=1,entrance_overlay=self.path)
        self.assertNotEqual(result['fatal'],'START_CLEARANCE_BLOCKED')
        self.assertEqual(result['entrance']['applied_cells'],81)

if __name__=='__main__': unittest.main()
