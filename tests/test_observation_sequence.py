#!/usr/bin/env python3
"""Pure policy regression tests. No robot/ROS2 execution."""
from __future__ import annotations
import math
from pathlib import Path
import sys
import unittest
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts/exploration'))
from hospital_coverage_core import Geometry
from observation_planning_core import Settings,SensorModel,validate_coverage
from observation_sequence_core import (MissionBook,StopEvidence,pose_repeated,
    validate_nav_path,refresh_candidate,build_batch,zone_order,safe_early_stop)

ZONES=[dict(id='A',mask_id=1,known_free_cells=100),dict(id='B',mask_id=2,known_free_cells=100)]
def status(t=0,n=1,a=0,b=0,sid='mission'):
    return dict(session_id=sid,last_evidence_scan_end_s=float(t),evidence_scans=n,
                zones=[dict(id='A',reference_free_cells=100,observed_cells=a),dict(id='B',reference_free_cells=100,observed_cells=b)])
def candidate(zone='A',x=2.,y=2.,yaw=.2):
    return dict(zone_id=zone,id='P',x=x,y=y,yaw=yaw)

class SequencePolicyTests(unittest.TestCase):
    def test_initial_high_requires_second_scan(self):
        b=MissionBook(ZONES);b.update(status(a=95));self.assertEqual(b.records['A'].state,'THRESHOLD_PENDING')
    def test_confirmed_threshold(self):
        b=MissionBook(ZONES);b.update(status(a=90));self.assertEqual(b.update(status(1,2,90)),['A']);self.assertTrue(b.is_complete('A'))
    def test_duplicate_scan_not_confirmation(self):
        b=MissionBook(ZONES);b.update(status(a=90));b.update(status(3,1,90));self.assertFalse(b.is_complete('A'))
    def test_short_interval_not_confirmation(self):
        b=MissionBook(ZONES);b.update(status(a=90));b.update(status(.2,2,90));self.assertFalse(b.is_complete('A'))
    def test_already_complete_skipped(self):
        b=MissionBook(ZONES);b.update(status(a=99));b.update(status(1,2,99));self.assertEqual([z['id'] for z in b.eligible()],['B'])
    def test_session_change_requires_restart(self):
        b=MissionBook(ZONES);b.update(status());self.assertRaises(ValueError,b.update,status(sid='other'))
    def test_regressing_counts_refused(self):
        b=MissionBook(ZONES);b.update(status(a=20));self.assertRaises(ValueError,b.update,status(2,2,10))
    def test_wrong_denominator_refused(self):
        b=MissionBook(ZONES);s=status();s['zones'][0]['reference_free_cells']=99
        self.assertRaises(ValueError,b.update,s)
    def test_nav_success_not_room_completion(self):
        b=MissionBook(ZONES);b.update(status(a=82));b.record_visit(candidate(),0,.82,'SUCCEEDED',None)
        self.assertFalse(b.is_complete('A'));self.assertEqual(b.locked_zone,'A')
    def test_view_limit_is_partial_not_unreachable(self):
        b=MissionBook(ZONES,max_views=1);b.update(status(a=50));b.record_visit(candidate(),0,.5,'SUCCEEDED',None)
        self.assertEqual(b.records['A'].state,'PARTIAL_DEFERRED');self.assertNotIn('UNREACHABLE',str(b.summary()))
    def test_two_low_gain_visits_defer(self):
        b=MissionBook(ZONES);b.update(status(a=50));b.record_visit(candidate(),.5,.5,'SUCCEEDED',None)
        b.record_visit(candidate(x=3),.5,.5,'SUCCEEDED',None)
        self.assertEqual(b.records['A'].deferred_reason,'LOW_MEASURED_GAIN_TWICE')
    def test_good_gain_resets_low_gain(self):
        b=MissionBook(ZONES);b.update(status(a=60));b.record_visit(candidate(),.5,.5,'SUCCEEDED',None)
        b.record_visit(candidate(x=3),.5,.6,'SUCCEEDED',None);self.assertEqual(b.records['A'].low_gain_count,0)
    def test_no_candidate_never_complete(self):
        b=MissionBook(ZONES);b.update(status());b.defer('A','NO_VIEW');self.assertFalse(b.is_complete('A'))
    def test_real_coverage_pattern(self):
        b=MissionBook(ZONES);b.update(status());b.update(status(2,2,82));b.record_visit(candidate(),0,.82,'SUCCEEDED',None)
        b.update(status(4,4,99));b.update(status(5,5,99));b.record_visit(candidate(x=3),.82,.99,'SUCCEEDED',None)
        self.assertTrue(b.is_complete('A'));self.assertIsNone(b.locked_zone)
        self.assertEqual(b.visits[-1]['nav_status'],'SUCCEEDED')
    def test_all_complete_scope(self):
        b=MissionBook(ZONES);b.update(status(a=99,b=92));b.update(status(2,2,99,92));self.assertTrue(b.all_complete())
    def test_same_pose_repeat(self):
        self.assertTrue(pose_repeated(candidate(),[candidate(x=2.1)]))
    def test_different_yaw_allowed(self):
        self.assertFalse(pose_repeated(candidate(yaw=2),[candidate(yaw=.2)]))
    def test_other_room_not_same_view(self):
        self.assertFalse(pose_repeated(candidate(),[candidate(zone='B')]))
    def test_wrap_yaw_repeat(self):
        self.assertTrue(pose_repeated(candidate(yaw=-math.pi+.05),[candidate(yaw=math.pi-.05)]))

class StopTests(unittest.TestCase):
    def stopped(self,s,start=0,after=0):
        for i in range(1,7):s.update(start+i*.25,after+i*.25,0,0,0,0)
    def test_stopped(self):
        s=StopEvidence();self.stopped(s);self.assertTrue(s.ready(1.5,1.5))
    def test_result_reset_rejects_old(self):
        s=StopEvidence();self.stopped(s);s.reset(after=2);self.stopped(s)
        self.assertFalse(s.ready(2,2))
    def test_motion_resets_window(self):
        s=StopEvidence();self.stopped(s);s.update(1.6,1.6,.5,0,0,0);self.assertFalse(s.ready(1.6,1.6))
    def test_rotation_not_stopped(self):
        s=StopEvidence()
        for i in range(10):s.update(i*.2,i*.2,0,0,0,.1)
        self.assertFalse(s.ready(2,2))
    def test_nonfinite_not_stopped(self):
        s=StopEvidence();self.stopped(s);s.update(1.7,1.7,float('nan'),0,0,0);self.assertFalse(s.ready(1.7,1.7))
    def test_duplicate_stamp_not_multiple_samples(self):
        s=StopEvidence()
        for i in range(20):s.update(1,i*.1,0,0,0,0)
        self.assertFalse(s.ready(1,2))
    def test_stale_odom(self):
        s=StopEvidence();self.stopped(s);self.assertFalse(s.ready(3,3))
    def test_wall_stale(self):
        s=StopEvidence();self.stopped(s);self.assertFalse(s.ready(1.5,10))
    def test_odom_gap(self):
        s=StopEvidence();self.stopped(s);s.update(3,3,0,0,0,0);self.assertFalse(s.ready(3,3))

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.g=Geometry(50,50,.1,(0.,0.,0.));self.free=np.ones((50,50),bool);self.cost=np.zeros((50,50),int)
        self.goal=candidate(x=3,y=2)
    def path(self,pts=None,frame='map'):
        return validate_nav_path(pts or [(2.,2.),(3.,2.)],frame,self.goal,(2.,2.,0.),self.g,self.free,self.cost)
    def test_path_length(self):self.assertAlmostEqual(self.path(),1.)
    def test_bad_frame(self):self.assertRaises(ValueError,self.path,None,'odom')
    def test_bad_endpoint(self):self.assertRaises(ValueError,self.path,[(2.,2.),(2.5,2.)])
    def test_bad_startpoint(self):self.assertRaises(ValueError,self.path,[(1.,2.),(3.,2.)])
    def test_cross_wall(self):
        self.free[:,25]=False;self.assertRaises(ValueError,self.path)
    def test_cross_live_inscribed_cost(self):
        self.cost[:,25]=99;self.assertRaises(ValueError,self.path)
    def test_unknown_path(self):
        self.cost[:,25]=-1;self.assertRaises(ValueError,self.path)
    def test_nan_path(self):self.assertRaises(ValueError,self.path,[(2,2),(float('nan'),2)])
    def test_clear_early_cancel_area(self):self.assertTrue(safe_early_stop(self.free,self.cost,self.g,(2,2,0),Settings()))
    def test_early_cancel_near_wall_refused(self):
        self.free[:,22]=False;self.assertFalse(safe_early_stop(self.free,self.cost,self.g,(2,2,0),Settings()))
    def test_prediction_not_added_to_observed(self):
        observed=np.zeros((5,5),bool);c=dict(visible_indices=np.arange(25),path_est_m=2.)
        r=refresh_candidate(c,observed,Geometry(5,5,.1,(0.,0.,0.)),Settings())
        self.assertIsNotNone(r);self.assertEqual(int(observed.sum()),0)


def run_tests(bundle=None):
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (SequencePolicyTests,StopTests,GeometryTests))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    if bundle:
        data,g,labels,free,signature=bundle
        assert len(data['zones'])==27 and int((labels>0).sum())==499983
        assert {'R14_T','R15_T'} <= {z['id'] for z in data['zones']}
        print('[BUNDLE PASS] approved 27-zone masks; denominator=499983; original coverage remains external')
    print('[SELF-TEST PASS] policies/path checks/stop evidence only; NOT real ROS2/DDS/Nav2 or physical stop validation')
    return result.testsRun

if __name__=='__main__':run_tests()
