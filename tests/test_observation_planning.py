#!/usr/bin/env python3
"""Deterministic unit and approved-map smoke tests; not live ROS testing."""
from __future__ import annotations
import ast
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/exploration'))
from hospital_coverage_core import Geometry,Coverage
from observation_planning_core import *


def fixture():
    g=Geometry(70,60,.1,(0.,0.,0.))
    free=np.ones((60,70),bool);free[[0,-1],:]=False;free[:,[0,-1]]=False
    labels=np.zeros(free.shape,np.uint16);labels[free]=2
    labels[20:40,25:50]=1
    zone=dict(id='ROOM',mask_id=1,known_free_cells=500,kind='room')
    return g,labels,free,zone


def example_sensor():return SensorModel(-math.pi,math.tau/180,180,.05,25.)


def status_for(labels,zones,observed,signature='test',session='s1',t=10.,seq=2):
    return dict(schema='hospital_room_coverage_v1',session_id=session,map_validated=True,fault=None,
                zone_bundle_signature=signature,observed_cells=int(observed.sum()),data_health='LIVE',
                evidence_scans=seq,last_evidence_scan_end_s=t,ros_time_s=t,
                max_observation_range_m=8.,zones=[dict(id=z['id'],reference_free_cells=z['known_free_cells'],
                observed_cells=int(np.count_nonzero(observed&(labels==z['mask_id'])))) for z in zones])


class PlanningTests(unittest.TestCase):
    def test_01_geometry_roundtrip_rotated(self):
        g=Geometry(50,30,.05,(-.3,1.2,.7));x,y=g.cell_to_world(11,17)
        self.assertTrue(np.allclose(g.world_to_grid_float(x,y),(11.5,17.5)))
    def test_02_sensor_limits(self):
        for s in (SensorModel(0,0,360,0,25),SensorModel(0,.01,360,30,25),SensorModel(0,.01,99999,0,25)):
            with self.assertRaises(ValueError):s.validate()
    def test_03_footprint_cannot_shrink(self):
        with self.assertRaises(ValueError):Settings(robot_radius_m=.2).validate()
    def test_04_nonfinite_config(self):
        with self.assertRaises(ValueError):Settings(max_range_m=float('nan')).validate()
    def test_05_diagonal_corner_not_passable(self):
        a=np.array([[1,0],[0,1]],bool);ids,flat,graph=path_graph(a,.1)
        dist=dijkstra(graph,indices=0);self.assertTrue(np.isinf(dist[1]))
    def test_06_diagonal_open(self):
        ids,flat,graph=path_graph(np.ones((2,2),bool),.1)
        self.assertAlmostEqual(dijkstra(graph,indices=0)[3],math.sqrt(2)*.1)
    def test_07_boundary_clearance(self):
        g,l,f,z=fixture();safe,clr=conservative_safe(f,g,Settings())
        self.assertFalse(safe[1,1]);self.assertTrue(safe[10,10])
    def test_08_wall_occlusion(self):
        g,l,f,z=fixture();f[:,35]=False
        visible=visibility_cells(g,f,l==1,2.5,3.,0.,example_sensor(),8.)
        self.assertTrue(len(visible)>0);self.assertTrue(np.all(visible%g.width<35))
    def test_09_only_target_label(self):
        g,l,f,z=fixture();v=visibility_cells(g,f,l==1,2.5,3.,0.,example_sensor(),8.)
        self.assertTrue(np.all(l.ravel()[v]==1))
    def test_10_range_cap(self):
        g,l,f,z=fixture();v=visibility_cells(g,f,f,2.55,3.05,0.,example_sensor(),.5)
        yy,xx=np.divmod(v,g.width)
        self.assertTrue(np.all(np.hypot((xx+.5)*.1-2.55,(yy+.5)*.1-3.05)<=.5+.1*math.sqrt(2)))
    def test_11_partial_fov(self):
        g,l,f,z=fixture();s=SensorModel(-.3,.01,61,.05,25)
        v=visibility_cells(g,f,f,2.55,3.05,0.,s,2.)
        self.assertTrue(np.all(v%g.width>=25))
    def test_12_sensor_offset(self):
        g,l,f,z=fixture();s=SensorModel(-.3,.01,61,.05,25,.3,0,0)
        a=visibility_cells(g,f,f,2.25,3.05,0.,s,2.)
        b=visibility_cells(g,f,f,2.55,3.05,0.,replace(s,offset_x=0),2.)
        self.assertTrue(np.array_equal(a,b))
    def test_13_vector_DDA_matches_existing_coverage_trace(self):
        g,l,f,z=fixture();f[10:45,36]=False;f[27:31,36]=True
        f[40,12:28]=False
        for x,y,yaw in [(1.25,1.55,.4),(2.95,3.15,-.7),(4.55,4.05,1.1)]:
            labels=np.where(f,1,0).astype(np.uint16)
            s=SensorModel(-math.pi,math.tau/300,300,.1,25.,.12,.03,.05)
            pred=visibility_cells(g,f,f,x,y,yaw,s,3.)
            c=Coverage(g,labels,f,[dict(id='r',mask_id=1,kind='room')])
            sx=x+math.cos(yaw)*s.offset_x-math.sin(yaw)*s.offset_y
            sy=y+math.sin(yaw)*s.offset_x+math.cos(yaw)*s.offset_y
            pose=(sx,sy,yaw+s.offset_yaw)
            c.integrate([25.]*s.beams,s.angle_min,s.angle_increment,s.range_min,s.range_max,pose,pose,1.,max_range=3.)
            self.assertTrue(np.array_equal(pred,np.flatnonzero(c.observed)))
    def test_14_input_immutability(self):
        g,l,f,z=fixture();obs=np.zeros(f.shape,bool);a=l.copy();b=f.copy();c=obs.copy()
        p=build_plan(g,l,f,z,obs,(1.5,1.5,0),example_sensor(),Settings(max_candidates=10))
        self.assertEqual(p['status'],'SELECTED');self.assertTrue(np.array_equal(l,a)&np.array_equal(f,b)&np.array_equal(obs,c))
    def test_15_external_viewpoints_allowed(self):
        g,l,f,z=fixture();safe,clr=conservative_safe(f,g,Settings())
        pts=generate_candidates(safe,clr,l==1,l,g,Settings(max_candidates=100))
        self.assertTrue(any(l[j,i]==2 for i,j in pts));self.assertTrue(any(l[j,i]==1 for i,j in pts))
    def test_16_blocked_seed_not_snapped(self):
        g,l,f,z=fixture();p=build_plan(g,l,f,z,np.zeros(f.shape,bool),(.05,.05,0),example_sensor(),Settings())
        self.assertEqual(p['status'],'START_CLEARANCE_BLOCKED')
    def test_start_local_bridge_near_wall_stays_read_only(self):
        g,l,f,z=fixture()
        cfg=Settings(max_candidates=15)
        safe,clr=conservative_safe(f,g,cfg)
        # A start inside original free space but outside circular-grid erosion.
        x,y=.41,3.05
        self.assertTrue(f[int(y/.1),int(x/.1)])
        self.assertFalse(safe[int(y/.1),int(x/.1)])
        self.assertTrue(oriented_footprint_clear(f,g,(x,y,0.),cfg))
        bridge=conservative_start_bridge(f,safe,g,(x,y,0.),cfg)
        self.assertIsNotNone(bridge)
        self.assertLessEqual(bridge[2],.35)
        p=build_plan(g,l,f,z,np.zeros(f.shape,bool),(x,y,0.),example_sensor(),cfg)
        self.assertEqual(p['start_mode'],'FOOTPRINT_LOCAL_BRIDGE_DISPLAY_ONLY')
        self.assertFalse(p['navigation_commands_sent'])

    def test_start_bridge_never_crosses_wall_or_unknown(self):
        g,l,f,z=fixture();f[:,20:25]=False
        cfg=Settings()
        safe,clr=conservative_safe(f,g,cfg)
        self.assertFalse(oriented_footprint_clear(f,g,(2.1,3.0,0.),cfg))
        self.assertIsNone(conservative_start_bridge(f,safe,g,(2.1,3.0,0.),cfg))

    def test_start_bridge_rejects_out_of_map_pose(self):
        g,l,f,z=fixture();safe,clr=conservative_safe(f,g,Settings())
        self.assertIsNone(conservative_start_bridge(f,safe,g,(0.0,0.,0.),Settings()))

    def test_17_grid_disconnection(self):
        g,l,f,z=fixture();f[:,18:22]=False
        p=build_plan(g,l,f,z,np.zeros(f.shape,bool),(1.,1.,0),example_sensor(),Settings(max_candidates=15))
        self.assertIsNone(p['selected'])
    def test_18_live_costmap_obstacle_excludes_target(self):
        g,l,f,z=fixture();cost=np.zeros(l.shape,np.int16);cost[l==1]=100
        p=build_plan(g,l,f,z,np.zeros(f.shape,bool),(1.5,1.5,0),example_sensor(),Settings(max_candidates=10),cost)
        self.assertIsNone(p['selected'])
    def test_19_observed_gain_zero(self):
        g,l,f,z=fixture();p=build_plan(g,l,f,z,np.zeros(f.shape,bool),(1.5,1.5,0),example_sensor(),Settings(max_candidates=5))
        q=rerank_plan(p,l>0,g,Settings());self.assertIsNone(q['selected']);self.assertEqual(q['status'],'NO_USEFUL_GAIN')
    def test_20_rerank_removes_seen_cells(self):
        g,l,f,z=fixture();obs=np.zeros(f.shape,bool)
        p=build_plan(g,l,f,z,obs,(1.5,1.5,0),example_sensor(),Settings(max_candidates=5))
        c=p['selected'];obs.ravel()[c['visible_indices']]=True
        q=rerank_plan(p,obs,g,Settings())
        self.assertTrue(all(not np.any(obs.ravel()[d['new_indices']]) for d in q['candidates']))
    def test_21_threshold_waits_two_evidence_scans(self):
        tr=ThresholdTracker(.9);s={'session_id':'x','last_evidence_scan_end_s':10.,'evidence_scans':1}
        self.assertEqual(tr.update(s,.9),'THRESHOLD_PENDING');self.assertEqual(tr.update(s,.9),'THRESHOLD_PENDING')
        s.update(last_evidence_scan_end_s=11.1,evidence_scans=2)
        self.assertEqual(tr.update(s,.9),'OBSERVATION_SUFFICIENT')
    def test_22_threshold_reset_session(self):
        tr=ThresholdTracker(.9);s={'session_id':'x','last_evidence_scan_end_s':10.,'evidence_scans':1}
        tr.update(s,.95);s.update(last_evidence_scan_end_s=12.,evidence_scans=2);tr.update(s,.95)
        s.update(session_id='y');self.assertEqual(tr.update(s,.1),'NEEDS_OBSERVATION')
    def test_23_prediction_does_not_satisfy_actual_threshold(self):
        tr=ThresholdTracker(.9);s={'session_id':'x','last_evidence_scan_end_s':10.,'evidence_scans':1}
        self.assertEqual(tr.update(s,.641),'NEEDS_OBSERVATION')
    def test_24_stale_and_clock(self):
        s={'data_health':'LIVE','ros_time_s':10.,'last_evidence_scan_end_s':9.,'evidence_scans':1}
        self.assertIsNone(current_input_error(s,10.,0));self.assertEqual(current_input_error(s,30.,0),'STALE_COVERAGE')
        self.assertEqual(current_input_error(s,1.,0),'CLOCK_MISMATCH');self.assertEqual(current_input_error(s,10.,20),'STALE_COVERAGE')
    def test_25_status_grid_signature(self):
        g,l,f,z=fixture();zones=[z,dict(id='other',mask_id=2,known_free_cells=int(np.sum(l==2)))];obs=np.zeros(l.shape,bool)
        s=status_for(l,zones,obs);grid=np.where(l>0,0,-1)
        validate_coverage(s,grid,l,'test',zones)
        with self.assertRaises(ValueError):validate_coverage(s,grid,l,'WRONG',zones)
    def test_26_status_grid_counts(self):
        g,l,f,z=fixture();zones=[z,dict(id='other',mask_id=2,known_free_cells=int(np.sum(l==2)))];obs=np.zeros(l.shape,bool)
        s=status_for(l,zones,obs);grid=np.where(l>0,0,-1);grid[25,30]=100
        with self.assertRaises(ValueError):validate_coverage(s,grid,l,'test',zones)
    def test_27_no_motion_APIs(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/'scripts/exploration/observation_goal_planner.py').read_text()
        tree=ast.parse(source,feature_version=(3,10))
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(calls)&{'send_goal_async','cancel_goal_async','create_client','create_service','set_parameters'})
        pubcalls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='create_publisher']
        topics={n.args[1].value for n in pubcalls}
        self.assertEqual(topics,{'/observation_planner/markers','/observation_planner/status'})
    def test_28_output_JSON_finite(self):
        g,l,f,z=fixture();p=build_plan(g,l,f,z,np.zeros(f.shape,bool),(1.5,1.5,0),example_sensor(),Settings(max_candidates=5))
        text=json.dumps(reportable(p),allow_nan=False);self.assertNotIn('visible_indices',text)
    def test_29_exterior_not_candidate(self):
        g,l,f,z=fixture();l[:,0:20]=0
        safe,clr=conservative_safe(f,g,Settings());pts=generate_candidates(safe,clr,l==1,l,g,Settings())
        self.assertTrue(all(l[j,i]>0 for i,j in pts))
    def test_30_grid_boundary_unknown_blocked(self):
        g,l,f,z=fixture();f[:,35]=False;v=visibility_cells(g,f,f,3.,3.,0,example_sensor(),8)
        self.assertTrue(np.all(v%g.width<35))


def run_tests(bundle=None):
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PlanningTests)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise AssertionError('Observation planner unit tests failed')
    if bundle is not None:
        data,g,labels,free,signature=bundle
        assert len(data['zones'])==27 and np.count_nonzero(labels)==499983
        z=next(z for z in data['zones'] if z['id']=='R14_T')
        assert z['known_free_cells']==11798
        original=array_fingerprint(labels)+array_fingerprint(free)
        p=build_plan(g,labels,free,z,np.zeros(free.shape,bool),(2.,0.,0.),example_sensor(),Settings(max_candidates=24))
        assert p['selected'] is not None
        assert all(np.all(labels.ravel()[c['visible_indices']]==z['mask_id']) for c in p['candidates'])
        assert original==array_fingerprint(labels)+array_fingerprint(free)
        print('[BUNDLE PASS] approved 27 zones; R14_T=11798 cells; reference-only synthetic scenario produced candidates')
    return result.testsRun

if __name__=='__main__':run_tests()
