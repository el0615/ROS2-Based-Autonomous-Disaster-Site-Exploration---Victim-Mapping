#!/usr/bin/env python3
"""Deterministic tests: no ROS server and no robot required."""
from __future__ import annotations
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/exploration'))
from hospital_coverage_core import Coverage, Geometry, decode_runs, planar_pose


def fixture(w=20,h=20,r=.1):
    g=Geometry(w,h,r,(0.,0.,0.))
    labels=np.ones((h,w),dtype=np.uint16);labels[:,10:]=2
    free=np.ones((h,w),bool)
    zones=[dict(id='A',kind='room',mask_id=1),dict(id='B',kind='corridor',mask_id=2)]
    return g,labels,free,zones


def scan(c,t=1.,ranges=None,p=(.25,1.05,0.),end=None,**kw):
    return c.integrate([1.3] if ranges is None else ranges,kw.pop('angle_min',0.),kw.pop('angle_increment',0.),
                       kw.pop('range_min',.05),kw.pop('range_max',5.),p,p if end is None else end,t,**kw)


class CoverageTests(unittest.TestCase):
    def test_top_left_to_ros(self):
        x=decode_runs(4,3,[[0,1,3,1],[2,0,1,2]],True)
        self.assertEqual(x[2,1],1);self.assertEqual(x[0,0],2);self.assertEqual(np.count_nonzero(x),3)
    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):decode_runs(4,3,[[0,0,3,1],[0,2,4,2]],True)
    def test_no_initial_observation(self):
        c=Coverage(*fixture());self.assertEqual(c.observed_total,0)
        self.assertEqual(c.statistics()[0]['observation_state'],'NO_VALID_SCAN')
    def test_wall_stops_ray(self):
        g,l,f,z=fixture();f[:,10]=False;l[~f]=0;c=Coverage(g,l,f,z)
        scan(c);self.assertGreater(c.observed_total,0);self.assertEqual(c.observed_counts[2],0)
    def test_open_door_sees_other_zone(self):
        g,l,f,z=fixture();f[:,10]=False;f[10,10]=True;l[~f]=0;c=Coverage(g,l,f,z)
        scan(c);self.assertGreater(c.observed_counts[1],0);self.assertGreater(c.observed_counts[2],0)
    def test_live_obstacle_endpoint_excluded(self):
        c=Coverage(*fixture());scan(c,ranges=[.30])
        self.assertTrue(c.observed[10,4]);self.assertFalse(c.observed[10,5]);self.assertFalse(c.observed[10,6:].any())
    def test_gray_unknown_is_blocker(self):
        g,l,f,z=fixture();f[:,8]=False;l[:,8]=0;c=Coverage(g,l,f,z)
        scan(c);self.assertFalse(c.observed[:,8:].any())
    def test_non_survey_can_be_traversed_by_rays(self):
        g,l,f,z=fixture();l[:,7:12]=0;c=Coverage(g,l,f,z)
        scan(c);self.assertEqual(int(c.observed[:,7:12].sum()),0);self.assertGreater(c.observed_counts[2],0)
    def test_corner_no_leak(self):
        g,l,f,z=fixture();f[0,1]=False;l[0,1]=0;c=Coverage(g,l,f,z)
        c.trace(.05,.05,math.pi/4,1.)
        self.assertFalse(c.observed[1:,1:].any())
    def test_repeated_no_inflation(self):
        c=Coverage(*fixture());a=scan(c);n=c.observed_total
        b=scan(c,t=2.);self.assertEqual(b['new_cells'],0);self.assertEqual(c.observed_total,n)
        self.assertFalse(scan(c,t=2.)['processed'])
    def test_inf_default_no_observation(self):
        c=Coverage(*fixture());scan(c,ranges=[math.inf])
        self.assertEqual(c.observed_total,0);self.assertEqual(c.evidence_scan_count,0)
    def test_inf_optin(self):
        c=Coverage(*fixture());scan(c,ranges=[math.inf],use_infinite=True,max_range=.5)
        self.assertGreater(c.observed_total,0);self.assertFalse(c.observed[:,8:].any())
    def test_invalid_ranges(self):
        c=Coverage(*fixture());r=scan(c,ranges=[float('nan'),-math.inf,-1.,0.,6.])
        self.assertEqual(c.observed_total,0);self.assertEqual(r['invalid_beams'],5)
    def test_range_cap(self):
        c=Coverage(*fixture());scan(c,ranges=[4.],max_range=.5)
        self.assertFalse(c.observed[10,8:].any());self.assertTrue(c.observed[10,3])
    def test_rotated_map(self):
        g,l,f,z=fixture();g=Geometry(g.width,g.height,g.resolution,(2.,-3.,math.pi/2));c=Coverage(g,l,f,z)
        x,y=g.cell_to_world(2,10);scan(c,p=(x,y,math.pi/2))
        self.assertTrue(c.observed[10,3]);self.assertFalse(c.observed[9].any())
    def test_translation_interpolated(self):
        c=Coverage(*fixture());scan(c,ranges=[.2,.2],p=(.25,.25,0),end=(.25,.45,0),time_increment=.1)
        self.assertTrue(c.observed[2,3]);self.assertTrue(c.observed[4,3]);self.assertFalse(c.observed[3,3])
    def test_threshold_is_not_mission_complete(self):
        c=Coverage(*fixture(),threshold=.005);scan(c);s=c.statistics()[0]
        self.assertEqual(s['observation_state'],'PARTIAL');scan(c,t=2.2)
        s=c.statistics()[0];self.assertEqual(s['observation_state'],'THRESHOLD_REACHED')
        self.assertFalse(s['victim_search_complete'])
    def test_zero_timestamp_rejected(self):
        with self.assertRaises(ValueError):scan(Coverage(*fixture()),t=0.)
    def test_sensor_in_wall_no_fake_seed(self):
        g,l,f,z=fixture();f[10,2]=False;l[10,2]=0;c=Coverage(g,l,f,z)
        out=scan(c);self.assertEqual(c.observed_total,0);self.assertEqual(out['sensor_blocked_beams'],1)
    def test_observed_runs_roundtrip(self):
        c=Coverage(*fixture());scan(c);recovered=np.zeros_like(c.observed)
        for row,left,right in c.observed_runs():recovered[row,left:right]=True
        self.assertTrue(np.array_equal(c.observed,recovered))
    def test_no_wedge_fill(self):
        c=Coverage(*fixture());scan(c,ranges=[1.3])
        self.assertEqual(int(c.observed[0:10].sum()),0);self.assertEqual(int(c.observed[11:].sum()),0)
    def test_pose_tilt_rejected(self):
        tr=NS(translation=NS(x=0,y=0,z=1),rotation=NS(x=math.sin(.2),y=0,z=0,w=math.cos(.2)))
        with self.assertRaises(ValueError):planar_pose(tr,.05)
    def test_rays_origin_blind_disk_not_counted(self):
        c=Coverage(*fixture());scan(c,range_min=.25)
        self.assertFalse(c.observed[10,2]);self.assertFalse(c.observed[10,3])
    def test_grid_semantics(self):
        g,l,f,z=fixture();l[:,0]=0;c=Coverage(g,l,f,z);scan(c)
        v=c.grid_values();self.assertTrue((v[:,0]==-1).all())
        self.assertEqual(np.count_nonzero(v==100),c.observed_total)


def run_tests(data,geom,labels,known_free):
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(CoverageTests))
    if not result.wasSuccessful():raise SystemExit(1)
    assert len(data['zones'])==27 and data['room_count']==20 and data['nonroom_count']==7
    assert int(np.count_nonzero(labels))==499983
    assert int(known_free.sum())==503186
    assert not np.any((labels>0)&~known_free)
    assert all(z['known_free_cells']==np.count_nonzero(labels==z['mask_id']) for z in data['zones'])
    # PGM rows 197:317, cols 0:37: exterior not in survey denominator.
    assert not labels[geom.height-317:geom.height-197,:37].any()
    g=geom
    for yaw in (0.,.4,-1.5):
        rot=Geometry(g.width,g.height,g.resolution,(g.origin[0],g.origin[1],yaw))
        for i,j in [(0,0),(100,150),(g.width-1,g.height-1)]:
            x,y=rot.cell_to_world(i,j);a,b=rot.world_to_grid_float(x,y)
            assert abs(a-(i+.5))<1e-8 and abs(b-(j+.5))<1e-8
    print('[BUNDLE PASS] exact approved 27-zone counts; source-white mask; exterior exclusion; rotated coordinates')

if __name__=='__main__':
    from hospital_coverage_core import load_bundle
    root=Path(__file__).resolve().parents[1]
    data,g,l,f,_=load_bundle(root/'config/exploration/hospital_zones.json',root/'config/exploration/hospital_coverage_reference.json.gz')
    run_tests(data,g,l,f)
