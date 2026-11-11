from fixtures import *
import unittest, ast, time
from dataclasses import replace
from zone_policy import ZoneMission, repeated, room_priority
from zone_planner import ZonePlanner, refresh, is_inside, arrival_diagnostic
from planning_core import Settings, SensorModel, current_input_error, validate_coverage
from sequence_core import StopEvidence, validate_nav_path, validate_oriented_swept_path

class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.b=bundle();self.book=ZoneMission(self.b[0]['zones'],max_views=3,revisit_cooldown_s=5.)
        self.s=status(self.b)
    def candidate(self,kind='ENTRY',zone='R_A'):
        return dict(id='c',zone_id=zone,x=6.,y=2.,yaw=0.,goal_kind=kind)
    def test_threshold_requires_fresh_second_evidence(self):
        seen=self.b[2]==2
        self.book.update(status(self.b,seen,seq=1,t=1))
        self.assertFalse(self.book.is_complete('R_A'))
        self.book.update(status(self.b,seen,seq=1,t=2))
        self.assertFalse(self.book.is_complete('R_A'))
        self.book.update(status(self.b,seen,seq=2,t=2.1))
        self.assertTrue(self.book.is_complete('R_A'))
    def test_no_completion_from_navigation(self):
        self.book.record_visit(self.candidate(),0.,.4,'SUCCEEDED',None)
        self.assertFalse(self.book.is_complete('R_A'))
    def test_entry_zero_gain_not_penalized(self):
        self.book.record_visit(self.candidate(),.4,.4,'SUCCEEDED',None)
        self.book.record_visit(self.candidate(),.4,.4,'SUCCEEDED',None)
        self.assertEqual(self.book.visit_state['R_A'].observation_goals,0)
        self.assertEqual(self.book.visit_state['R_A'].low_gain,0)
        self.assertNotEqual(self.book.records['R_A'].state,'PARTIAL_DEFERRED')
    def test_local_low_gain_is_bounded(self):
        for _ in range(2):self.book.record_visit(self.candidate('OBSERVE'),.4,.4,'SUCCEEDED',None)
        self.assertEqual(self.book.records['R_A'].deferred_reason,'LOW_MEASURED_GAIN_TWICE_PARTIAL')
    def test_local_view_budget(self):
        for _ in range(3):self.book.record_visit(self.candidate('OBSERVE'),.3,.4,'SUCCEEDED',None)
        self.assertEqual(self.book.records['R_A'].state,'PARTIAL_DEFERRED')
    def test_no_permanent_unreachable(self):
        self.book.defer('R_A','NO_SAMPLED_POSE')
        self.assertNotIn('UNREACHABLE',self.book.records['R_A'].state)
    def test_no_revisit_on_clock_only(self):
        self.book.context('a',0);self.book.defer('R_A','x');self.book.context('a',100)
        self.assertEqual(self.book.reopen_changed(),[])
    def test_revisit_on_cost_change_after_cooldown(self):
        self.book.context('a',0);self.book.defer('R_A','x');self.book.context('b',2)
        self.assertEqual(self.book.reopen_changed(),[])
        self.book.context('b',6);self.assertEqual(self.book.reopen_changed(),['R_A'])
    def test_revisit_on_other_progress(self):
        self.book.context('a',0);self.book.defer('R_A','x');self.book.progress+=1;self.book.context('a',6)
        self.assertEqual(self.book.reopen_changed(),['R_A'])
    def test_revisit_cap(self):
        self.book.context('a',0);self.book.defer('R_A','x');self.book.context('b',6);self.book.reopen_changed()
        self.book.defer('R_A','x');self.book.context('c',12)
        self.assertEqual(self.book.reopen_changed(),[])
    def test_session_change_is_fatal(self):
        self.book.update(self.s);s=dict(self.s,session_id='new')
        with self.assertRaisesRegex(ValueError,'SESSION_CHANGED'):self.book.update(s)
    def test_coverage_regression_is_fatal(self):
        seen=self.b[2]==2;self.book.update(status(self.b,seen))
        with self.assertRaisesRegex(ValueError,'REGRESSION'):self.book.update(self.s)
    def test_rooms_first_not_hardcoded_ids(self):
        self.assertEqual(room_priority(dict(id='ROOM_NEW',kind='room'),'rooms-first'),0)
        self.assertEqual(room_priority(dict(id='LOBBY',kind='lobby'),'rooms-first'),1)
    def test_entry_history_distinct_from_observation(self):
        c=self.candidate();p=dict(c,goal_kind='OBSERVE')
        self.assertFalse(repeated(c,[p]));self.assertTrue(repeated(c,[c]))
    def test_completion_releases_active(self):
        self.book.activate('R_A');seen=self.b[2]==2
        self.book.update(status(self.b,seen,1,1));self.book.update(status(self.b,seen,2,2.1))
        self.assertIsNone(self.book.active_zone)

class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.b=bundle();self.cfg=Settings();self.s=snapshot(self.b)
        self.p=ZonePlanner(self.b,self.cfg,backend='python',local_candidates=6,heading_samples=2,budget_s=15)
        self.zone=self.b[0]['zones'][1]
    def test_entry_is_internal_and_zero_gain_allowed(self):
        batch=self.p.entry_batch(self.s,self.b[0]['zones'],[])
        self.assertTrue(batch['candidates']);self.assertNotEqual(batch['zone_id'],'LOBBY')
        for c in batch['candidates']:
            self.assertEqual(c['goal_kind'],'ENTRY');self.assertEqual(c['new_area_m2'],0)
            self.assertEqual(c['viewpoint_zone_mask_id'],self.zone['mask_id'])
            self.assertIsNotNone(refresh(c,self.s['observed'],self.b[1],self.cfg))
    def test_entry_depth_is_recorded_and_prefers_buffer(self):
        batch=self.p.entry_batch(self.s,[self.zone],[])
        self.assertTrue(batch['candidates'])
        values=[c['entry_boundary_depth_m'] for c in batch['candidates']]
        self.assertTrue(all(d >= self.p.entry_inset for d in values))
        self.assertGreaterEqual(values[0],self.p.entry_inset+self.p.entry_arrival_buffer)
    def test_failed_entry_excludes_same_position_even_different_yaw(self):
        first=self.p.entry_batch(self.s,[self.zone],[])['candidates'][0]
        retry=self.p.entry_batch(self.s,[self.zone],[dict(first,yaw=first['yaw']+1)])
        for c in retry['candidates']:
            self.assertGreaterEqual(math.hypot(c['x']-first['x'],c['y']-first['y']),.60-1e-9)
    def test_arrival_diagnostic_contains_measured_center_and_footprint(self):
        goal=self.p.entry_batch(self.s,[self.zone],[])['candidates'][0]
        diag=arrival_diagnostic(self.zone,(goal['x'],goal['y'],goal['yaw']),
                                (2.,2.,0.),self.b,self.cfg)
        self.assertFalse(diag['actual']['center_inside'])
        self.assertFalse(diag['actual']['footprint_inside'])
        self.assertTrue(diag['goal']['footprint_inside'])
        self.assertGreater(diag['xy_error_m'],0)
    def test_entry_has_no_visibility_cost(self):
        self.p.entry_batch(self.s,[self.zone],[])
        self.assertEqual(self.p.visibility.calls,0)
    def test_entry_does_not_change_inputs(self):
        original=self.b[3].copy();seen=self.s['observed'].copy()
        self.p.entry_batch(self.s,[self.zone],[])
        np.testing.assert_array_equal(original,self.b[3]);np.testing.assert_array_equal(seen,self.s['observed'])
    def test_entry_hole_not_centroid(self):
        self.s['cost']['values'][25:30,65:70]=100
        batch=self.p.entry_batch(self.s,[self.zone],[])
        for c in batch['candidates']:
            i,j=c['cell'];self.assertNotEqual(self.s['cost']['values'][j,i],100)
    def test_no_safe_interior_is_partial_reason(self):
        self.s['cost']['values'][self.b[2]==2]=100
        b=self.p.entry_batch(self.s,[self.zone],[])
        self.assertEqual(b['candidates'],[]);self.assertIn('NOT_UNREACHABLE_PROOF',b['failures']['R_A'])
    def test_disconnected_graph_offers_navfn_probe_not_safe_claim(self):
        self.s['cost']['values'][:,44]=100
        b=self.p.entry_batch(self.s,[self.zone],[])
        self.assertTrue(b['candidates'])
        self.assertEqual(b['candidates'][0]['route_mode'],'NAVFN_ENTRY_PROBE')
    def test_local_catalog_only_target_zone(self):
        c,_,_=self.p.catalog(self.s,self.zone)
        self.assertTrue(c)
        self.assertTrue(all(x['viewpoint_zone_mask_id']==2 for x in c))
    def test_catalog_reused_when_only_observed_changes(self):
        self.p.catalog(self.s,self.zone);calls=self.p.visibility.calls
        self.s['observed'][20:22,50:55]=True
        _,_,hit=self.p.catalog(self.s,self.zone)
        self.assertTrue(hit);self.assertEqual(self.p.visibility.calls,calls)
    def test_costmap_change_invalidates_catalog(self):
        self.p.catalog(self.s,self.zone);n=self.p.visibility.calls
        self.s['cost']['values'][20,60]=100
        _,_,hit=self.p.catalog(self.s,self.zone)
        self.assertFalse(hit);self.assertGreater(self.p.visibility.calls,n)
    def test_sensor_change_invalidates_catalog(self):
        self.p.catalog(self.s,self.zone)
        self.s['sensor']=replace(self.s['sensor'],offset_yaw=.1)
        _,_,hit=self.p.catalog(self.s,self.zone);self.assertFalse(hit)
    def test_prefetch_does_not_create_executable_plan(self):
        r=self.p.prefetch(self.s,self.zone)
        self.assertEqual(r['kind'],'PREFETCH');self.assertNotIn('candidates',r)
    def test_fresh_coverage_reranks(self):
        r=self.p.local_batch(self.s,self.zone,[]);self.assertTrue(r['candidates'])
        self.s['observed']|=self.b[2]==2
        r=self.p.local_batch(self.s,self.zone,[]);self.assertEqual(r['candidates'],[])
    def test_cancel_cooperative(self):
        self.p.stop()
        with self.assertRaisesRegex(RuntimeError,'CANCELLED'):self.p.entry_batch(self.s,[self.zone],[])
    def test_budget_exhaustion_not_cached(self):
        self.p.budget_s=0
        with self.assertRaises(TimeoutError):self.p.catalog(self.s,self.zone)
        self.assertEqual(len(self.p.catalogs),0)
    def test_rotated_map_membership(self):
        b=list(self.b);b[1]=Geometry(100,80,.1,(4.,-3.,.7))
        x,y=b[1].cell_to_world(65,25)
        self.assertTrue(is_inside(self.zone,(x,y,.7),b,self.cfg))
    def test_enclosed_free_start_without_conservative_bridge_delegates_to_navfn(self):
        # Center is known free, with a clear oriented footprint, but strict
        # circular clearance and the short straight bridge are impossible.
        # This must not become a fabricated connected strict-grid route.
        cost = self.s['cost']['values']
        cost[16:25,16] = 100; cost[16:25,24] = 100
        cost[16,16:25] = 100; cost[24,16:25] = 100
        maps = self.p.maps(self.s)
        self.assertTrue(maps['free'][20,20])
        self.assertFalse(maps['safe'][20,20])
        from planning_core import conservative_start_bridge
        self.assertIsNone(conservative_start_bridge(maps['free'],maps['safe'],self.b[1],self.s['pose'],self.cfg))
        batch = self.p.entry_batch(self.s,[self.zone],[])
        self.assertEqual(batch['start_route_mode'],'NAVFN_START_PROBE_NO_CONSERVATIVE_BRIDGE')
        self.assertTrue(batch['candidates'])
        self.assertTrue(all(c['route_mode']=='NAVFN_ENTRY_PROBE' for c in batch['candidates']))
        self.assertTrue(all(is_inside(self.zone,(c['x'],c['y'],c['yaw']),self.b,self.cfg)
                            for c in batch['candidates']))
        self.assertTrue(all(c['path_est_m']>=0 for c in batch['candidates']))

    def test_enclosed_free_start_observation_uses_navfn_probe(self):
        cost=self.s['cost']['values']
        cost[16:25,16]=100;cost[16:25,24]=100
        cost[16,16:25]=100;cost[24,16:25]=100
        batch=self.p.local_batch(self.s,self.zone,[])
        self.assertEqual(batch['start_route_mode'],'NAVFN_START_PROBE_NO_CONSERVATIVE_BRIDGE')
        self.assertTrue(batch['candidates'])
        self.assertTrue(all(c['route_mode']=='NAVFN_ENTRY_PROBE' for c in batch['candidates']))

    def test_lethal_start_is_still_rejected_without_delegation(self):
        i,j=20,20
        self.s['cost']['values'][j,i]=100
        with self.assertRaisesRegex(ValueError,'START_NOT_CURRENT_FREE'):
            self.p.entry_batch(self.s,[self.zone],[])

    def test_unknown_start_is_still_rejected_without_delegation(self):
        i,j=20,20
        self.s['cost']['values'][j,i]=-1
        with self.assertRaisesRegex(ValueError,'START_NOT_CURRENT_FREE'):
            self.p.entry_batch(self.s,[self.zone],[])

    def test_strict_start_still_uses_connected_graph(self):
        batch=self.p.entry_batch(self.s,[self.zone],[])
        self.assertEqual(batch['start_route_mode'],'STRICT_CONSERVATIVE_GRID')
        self.assertTrue(any(c['route_mode']=='STRICT_GRID' for c in batch['candidates']))

    def test_invalid_start_rejected(self):
        self.s['pose']=(-1.,-1.,0)
        with self.assertRaisesRegex(ValueError,'START_'):self.p.entry_batch(self.s,[self.zone],[])

class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.b=bundle();self.g=self.b[1];self.s=snapshot(self.b);self.cfg=Settings()
        self.pose=(2.,2.,0);self.goal=dict(x=2.2,y=2.,yaw=0.)
    def sweep(self):return validate_oriented_swept_path([(2.,2.),(2.2,2.)],self.pose,self.goal,self.g,self.b[3],self.s['cost']['values'],self.cfg)
    def test_open_sweep(self):self.assertGreater(self.sweep(),0)
    def test_lethal_rejected(self):
        self.s['cost']['values'][20,22]=100
        with self.assertRaises(ValueError):self.sweep()
    def test_unknown_rejected(self):
        self.s['cost']['values'][20,22]=-1
        with self.assertRaises(ValueError):self.sweep()
    def test_99_strict_body_rejected(self):
        self.s['cost']['values'][20,22]=99
        with self.assertRaises(ValueError):self.sweep()
    def test_bad_frame_rejected(self):
        with self.assertRaises(ValueError):validate_nav_path([(2,2),(2.2,2)],'odom',self.goal,self.pose,self.g,self.b[3],self.s['cost']['values'])
    def test_stop_requires_fresh_samples(self):
        e=StopEvidence();e.reset(after=1)
        for t in np.arange(1.1,2.3,.1):e.update(float(t),10.,0,0,0,0)
        self.assertTrue(e.ready(2.2,10.));self.assertFalse(e.ready(5.,14.))
    def test_cancel_ack_not_stop_evidence(self):self.assertFalse(StopEvidence().ready(1.,1.))
    def test_future_coverage_rejected(self):
        self.assertEqual(current_input_error(self.s['status'],1.,0),'CLOCK_MISMATCH')
    def test_invalid_coverage_grid_rejected(self):
        grid=np.where(self.b[2]>0,0,-1)
        grid[10,10]=100
        with self.assertRaises(ValueError):validate_coverage(self.s['status'],grid,self.b[2],self.b[4],self.b[0]['zones'])

class SourceTests(unittest.TestCase):
    def test_python310_parse_all(self):
        for p in SRC.glob('*.py'):ast.parse(p.read_text(),feature_version=(3,10))
    def test_no_velocity_map_tf_publishers(self):
        s=(SRC/'zone_node.py').read_text()
        self.assertNotIn('create_publisher(Twist',s)
        self.assertNotIn('create_publisher(OccupancyGrid',s)
        self.assertNotIn('TransformBroadcaster',s)
    def test_actual_execution_requires_validated_navfn_goal_without_full_sweep(self):
        s=(SRC/'zone_node.py').read_text()
        self.assertIn("self.validated_plan_mode!='NAVFN_GOAL_VALIDATED'",s)
        self.assertNotIn('validate_oriented_swept_path(',s)
        self.assertNotIn('STRICT_SWEEP_OK',s)
        self.assertIn('GOAL_FOOTPRINT_BLOCKED',s)
        self.assertIn('GOAL_NOT_INSIDE_TARGET_ZONE',s)
    def test_no_v4_guard_bypass(self):
        s=(SRC/'zone_node.py').read_text()
        self.assertNotIn('DN011_VISIBILITY_EXECUTE_V4',s)
        self.assertNotIn('os.environ[\'DN011_VISIBILITY_FAST\']',s)

if __name__=='__main__':unittest.main(verbosity=2)
