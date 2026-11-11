"""Focused v1.7 regressions. ROS/TF/scan callbacks are mocked, not Gazebo.

Set DN012_V17_SOURCE_DIR to staged/installed source. Optional
DN012_V17_REAL_PROJECT enables checks of the user's actual 27-zone bundle.
"""
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
from dataclasses import asdict, dataclass
import numpy as np

SRC = Path(os.environ.get('DN012_V17_SOURCE_DIR', Path(__file__).resolve().parent))
sys.path.insert(0, str(SRC))
from reroute_core import (RerouteSettings, RerouteDetector, PointStore,
    compare_paths, geometry_dict, context_for, json_hash, rasterize_points)
from reroute_evidence import EvidenceJournal, ZoneLookup, PATCH_VERSION
from reroute_runtime import RerouteRuntime

# This Geometry copy follows the uploaded hospital_coverage_core.py.
@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    resolution: float
    origin: tuple
    def world_to_grid_float(self, x, y):
        ox,oy,yaw=self.origin;c,s=math.cos(yaw),math.sin(yaw);dx,dy=x-ox,y-oy
        return ((c*dx+s*dy)/self.resolution,(-s*dx+c*dy)/self.resolution)
    def cell_to_world(self,i,j):
        ox,oy,yaw=self.origin;x,y=(i+.5)*self.resolution,(j+.5)*self.resolution
        c,s=math.cos(yaw),math.sin(yaw)
        return ox+c*x-s*y,oy+s*x+c*y
    def contains(self,i,j):return 0<=i<self.width and 0<=j<self.height

ZONE_NAMES = ['R01','R02A','R02B','R03','R04','R05','R06A','R06B','R07A','R07B',
    'R08A','R08B','R09','R10A','R10B','R11','R12','R13','R14_T','R15_T','LOBBY',
    'CORRIDOR_B','CONNECTOR_1','CONNECTOR_2','CORRIDOR_C','CORRIDOR_D','CORRIDOR_E']

def bundle():
    g=Geometry(80,80,.25,(-5.,-5.,0.))
    labels=np.full((80,80),22,dtype=np.uint16)
    labels[0,:27]=np.arange(1,28)
    zones=[dict(id=name,mask_id=i+1) for i,name in enumerate(ZONE_NAMES)]
    return dict(zones=zones),g,labels,np.ones(labels.shape,dtype=bool),'fixture-signature'


def event(confirmed=False, stamp=10., anchor=(0.,0.)):
    s=RerouteSettings()
    old=[(0.,0.),(6.,0.)];new=[(0.,0.),(0.,3.),(6.,3.),(6.,0.)]
    metric=compare_paths(old,new,(0.,0.),(6.,0.),s)
    return dict(goal_id='a'*32,goal_xy=[6.,0.],anchor_xy=list(anchor),anchor_ros_s=stamp,
        confirmed_ros_s=stamp+.6 if confirmed else None,confirmation_samples=2 if confirmed else 1,
        sequence=1 if confirmed else 0,plan_change_confirmed=confirmed,metric=metric,
        confirm_metric=copy.deepcopy(metric),physical_blockage_proven=False,path_execution_confirmed=False,
        semantics='large_detour_operational_candidate_NOT_passage_cut_proof')


def snapshot(confirmed=False, stamp=10.):
    return dict(event(confirmed,stamp),point_xy=[.7,0.],spatial_zone='CORRIDOR_B',
        spatial_lookup={'zone_id':'CORRIDOR_B','mask_id':22,'cell_ros':[22,20],'status':'MATCHED'},
        anchor_spatial_lookup={'zone_id':'CORRIDOR_B','mask_id':22},destination_zone='R12',
        correction_source='OLD_PATH_FORWARD_OFFSET_ESTIMATE',mask_eligible=confirmed,
        mask_hold_reason=None if confirmed else 'PENDING_PLAN_CONFIRMATION',mask_radius_m=.15)


def fake_modules():
    std=types.ModuleType('std_msgs');msg=types.ModuleType('std_msgs.msg');msg.String=type('String',(),{})
    qos=types.ModuleType('rclpy.qos')
    qos.QoSProfile=lambda **kw: types.SimpleNamespace(**kw)
    qos.ReliabilityPolicy=types.SimpleNamespace(RELIABLE=1)
    qos.DurabilityPolicy=types.SimpleNamespace(VOLATILE=0)
    path=types.ModuleType('path_blockage');path.path_obstacle_witness=lambda *a,**k:None
    return {'std_msgs':std,'std_msgs.msg':msg,'rclpy':types.ModuleType('rclpy'),
            'rclpy.qos':qos,'path_blockage':path}

class Node:
    def __init__(self):
        self.nav_uuid='a'*32;self.current={'x':6.,'y':0.,'yaw':0.,'zone_id':'R12'}
        self.zones_by_id={'R12':{'id':'R12','mask_id':17}}
        self.nav_handle=object();self.nav_terminal=None;self.stop_requested=None
        self.cancel_reason=None;self.coverage_epoch=1;self.sim=10.;self.events=[];self.stops=[]
        self.foreign=False
    def ros_s(self):return self.sim
    def event(self,name,**kw):self.events.append((name,copy.deepcopy(kw)))
    def info(self,*a):pass
    def create_subscription(self,*a):return object()
    def foreign_active(self):return ['other'] if self.foreign else []
    def request_stop(self,reason):self.stops.append(reason);self.stop_requested=reason

class Harness:
    def __init__(self, cls=RerouteRuntime):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.b=bundle();self.n=Node();self.s=RerouteSettings()
        conf=self.root/'config/exploration/navigation';conf.mkdir(parents=True)
        (conf/'dn012_reroute_v1_4.json').write_text(json.dumps(dict(scene_id='test',detector=asdict(self.s))))
        (conf/'hospital_zones.json').write_text('{}')
        (conf/'hospital_coverage_reference.json.gz').write_bytes(b'context-only-fixture')
        world=self.root/'gazebo/worlds/disaster/disaster_hospital.sdf';world.parent.mkdir(parents=True);world.write_text('<sdf/>')
        self.out=self.root/'session';self.out.mkdir()
        self.monitor=types.SimpleNamespace(hits=None,settings=object())
        self.inp=dict(epoch=1,pose=(0.,0.,0.),cost={'values':np.zeros((80,80)), 'key':'cost1','stamp':10.})
        self.env=patch.dict(os.environ,{'ZB_PROJECT_ROOT':str(self.root),'DN012_REROUTE_MODE':'observe'},clear=False)
        self.env.start()
        self.mods=fake_modules();self.modules=patch.dict(sys.modules,self.mods);self.modules.start()
        self.runtime=cls(self.monitor,self.n,self.b,object(),types.SimpleNamespace(namespace='/j100_0001',robot_frame='base_link'),self.out)
        self.runtime._pose_at=lambda stamp:(0.,0.,0.)
    def close(self):
        self.runtime.close();self.modules.stop();self.env.stop();self.tmp.cleanup()
    def record(self,confirmed=False,stamp=10.):
        e=event(confirmed,stamp);self.n.sim=stamp+(.6 if confirmed else 0.)
        self.runtime._record(e,self.inp,e['confirm_metric']['new_remaining'])
        return self.runtime.evidence.summary()['records'][0]
    def feed(self,points,stamp):
        self.n.sim=stamp;self.runtime.on_plan(points,stamp);self.runtime.tick(self.inp)

class SpatialTests(unittest.TestCase):
    def setUp(self):self.b=bundle();self.lookup=ZoneLookup(self.b)
    def test_all_27_ids(self):
        for i,name in enumerate(ZONE_NAMES):
            with self.subTest(name=name):
                self.assertEqual(self.lookup.locate(self.b[1].cell_to_world(i,0))['zone_id'],name)
    def test_r12_scope_does_not_hide_corridor(self):
        self.assertEqual(self.lookup.locate((0.,0.))['zone_id'],'CORRIDOR_B')
    def test_unassigned_not_nearest_or_target(self):
        self.b[2][20,20]=0
        d=self.lookup.locate((0.,0.));self.assertIsNone(d['zone_id']);self.assertEqual(d['status'],'UNASSIGNED_CELL')
    def test_outside_left(self):self.assertEqual(self.lookup.locate((-5.001,0))['status'],'OUTSIDE_MAP')
    def test_right_edge_is_outside(self):self.assertEqual(self.lookup.locate((15.,0))['status'],'OUTSIDE_MAP')
    def test_top_edge_is_outside(self):self.assertEqual(self.lookup.locate((0.,15.))['status'],'OUTSIDE_MAP')
    def test_negative_coordinate_uses_floor(self):self.assertEqual(self.lookup.locate((-5.001,0))['cell_ros'][0],-1)
    def test_internal_boundary_uses_existing_grid(self):
        g=self.b[1];self.b[2][20,19]=17
        self.assertEqual(self.lookup.locate((-.00001,0))['zone_id'],'R12')
        self.assertEqual(self.lookup.locate((0,0))['zone_id'],'CORRIDOR_B')
    def test_nonfinite_is_unassigned(self):
        for v in (math.nan,math.inf,-math.inf):self.assertEqual(self.lookup.locate((v,0))['status'],'INVALID_COORDINATE')
    def test_unknown_mask_is_not_target(self):
        self.b[2][20,20]=99;self.assertEqual(self.lookup.locate((0,0))['status'],'MASK_ID_NOT_IN_ZONE_CATALOG')
    def test_duplicate_catalog_refused(self):
        self.b[0]['zones'].append(dict(id='duplicate',mask_id=22))
        with self.assertRaises(ValueError):ZoneLookup(self.b)
    def test_rotated_map_transform(self):
        g=Geometry(80,80,.25,(-5.,-5.,math.pi/2))
        b=(self.b[0],g,*self.b[2:]);q=ZoneLookup(b)
        self.assertEqual(q.locate(g.cell_to_world(16,0))['zone_id'],'R12')
    def test_two_historical_coordinates_from_uploaded_mask_runs(self):
        # TWO actual uploaded mask row runs, not a reconstructed full map.
        g=Geometry(1127,507,.05,(-.561,-13.9,0.));labels=np.zeros((507,1127),np.uint16)
        for row,left,right,mid in [(123,775,939,22),(159,526,939,22)]:labels[506-row,left:right]=mid
        q=ZoneLookup((self.b[0],g,labels,None,'two-exact-uploaded-runs'))
        for point in ((40.564,5.275),(41.564,3.475)):
            d=q.locate(point);self.assertEqual((d['mask_id'],d['zone_id']),(22,'CORRIDOR_B'))

class JournalTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.j=EvidenceJournal(Path(self.tmp.name)/'e','context');self.ident='a'*20
    def tearDown(self):self.tmp.cleanup()
    def test_pending_written_immediately(self):
        r=self.j.record(self.ident,snapshot());self.assertTrue(Path(r['pending_file']).exists());self.assertEqual(r['status'],'PENDING')
    def test_confirmation_cannot_replace_first_pending(self):
        p=self.j.record(self.ident,snapshot());before=Path(p['pending_file']).read_bytes()
        s=snapshot(True);s['point_xy']=[1.2,0.];s['spatial_zone']='R12';s['correction_source']='LIDAR_COSTMAP_OLD_PATH'
        c=self.j.record(self.ident,s)
        self.assertEqual(Path(p['pending_file']).read_bytes(),before)
        self.assertEqual(c['first_detection']['point_xy'],[.7,0.]);self.assertEqual(c['confirmation']['point_xy'],[1.2,0.])
        self.assertEqual(c['first_detection']['spatial_zone'],'CORRIDOR_B')
    def test_duplicate_pending_is_no_clobber(self):
        a=self.j.record(self.ident,snapshot());s=snapshot();s['point_xy']=[2.,0.]
        b=self.j.record(self.ident,s);self.assertEqual(a,b)
    def test_duplicate_confirmation_is_no_clobber(self):
        a=self.j.record(self.ident,snapshot(True));b=snapshot(True);b['point_xy']=[2.,0.]
        self.assertEqual(a,self.j.record(self.ident,b))
    def test_no_fabricated_pending_for_confirmation_only(self):
        d=self.j.record(self.ident,snapshot(True));self.assertFalse(d['first_pending_available']);self.assertIsNone(d['pending_file'])
    def test_snapshot_detached_from_input_mutation(self):
        s=snapshot();self.j.record(self.ident,s);s['metric']['extra_m']=999
        self.assertEqual(self.j.summary()['records'][0]['first_detection']['metric']['extra_m'],6.)
    def test_summary_detached_from_external_mutation(self):
        self.j.record(self.ident,snapshot());d=self.j.summary();d['records'][0]['status']='BAD'
        self.assertEqual(self.j.summary()['records'][0]['status'],'PENDING')
    def test_new_goal_has_separate_evidence(self):
        self.j.record(self.ident,snapshot());self.j.record('b'*20,snapshot(stamp=20.))
        self.assertEqual(self.j.summary()['counts']['total'],2)
    def test_recover_after_restart_without_index(self):
        a=self.j.record(self.ident,snapshot());(self.j.folder/'index.json').unlink()
        again=EvidenceJournal(self.j.folder,'context');self.assertEqual(again.summary()['records'],[a])
    def test_wrong_context_refused(self):
        self.j.record(self.ident,snapshot())
        with self.assertRaises(ValueError):EvidenceJournal(self.j.folder,'other')
    def test_index_has_hash_and_first_and_confirmation_links(self):
        self.j.record(self.ident,snapshot());self.j.record(self.ident,snapshot(True))
        d=json.loads((self.j.folder/'index.json').read_text());r=d['records'][0]
        self.assertEqual(len(r['pending_sha256']),64);self.assertTrue(r['confirmation_file']);self.assertFalse(d['controls_masks_or_navigation'])
    def test_proof_escalation_refused(self):
        s=snapshot();s['physical_blockage_proven']=True
        with self.assertRaises(ValueError):self.j.record(self.ident,s)
    def test_pending_mask_activation_refused(self):
        s=snapshot();s['mask_eligible']=True
        with self.assertRaises(ValueError):self.j.record(self.ident,s)
    def test_nan_refused(self):
        s=snapshot();s['metric']['extra_m']=math.nan
        with self.assertRaises(ValueError):self.j.record(self.ident,s)
    def test_symlink_refused(self):
        target=self.j.folder/'real';target.write_text('{}');(self.j.folder/(self.ident+'.pending.json')).symlink_to(target)
        with self.assertRaises(ValueError):self.j.record(self.ident,snapshot())
    def test_inconsistent_confirmation_never_published(self):
        self.j.record(self.ident,snapshot());bad=snapshot(True);bad['anchor_xy']=[5.,5.]
        with self.assertRaises(ValueError):self.j.record(self.ident,bad)
        self.assertFalse((self.j.folder/(self.ident+'.confirmed.json')).exists())
        self.assertEqual(self.j.summary()['counts']['pending'],1)
    def test_path_traversal_id_refused(self):
        with self.assertRaises(ValueError):self.j.record('../bad',snapshot())

class DetectorUnchangedTests(unittest.TestCase):
    def test_and_boundary(self):
        m=compare_paths([(0,0),(6,0)],[(0,0),(0,1.5),(6,1.5),(6,0)],(0,0),(6,0),RerouteSettings())
        self.assertTrue(m['triggered']);self.assertEqual(m['extra_m'],3.)
    def test_absolute_alone_does_not_trigger(self):
        m=compare_paths([(0,0),(10,0)],[(0,0),(0,1.5),(10,1.5),(10,0)],(0,0),(10,0),RerouteSettings())
        self.assertFalse(m['triggered'])
    def test_ratio_alone_does_not_trigger(self):
        m=compare_paths([(0,0),(2,0)],[(0,0),(0,.6),(2,.6),(2,0)],(0,0),(2,0),RerouteSettings())
        self.assertFalse(m['triggered'])
    def test_no_confirm_until_time_and_sample_gates(self):
        d=RerouteDetector();old=[(0,0),(6,0)];new=[(0,0),(0,3),(6,3),(6,0)]
        d.feed(old,(0,0),1,'g',(6,0));p,_=d.feed(new,(0,0),2,'g',(6,0))
        self.assertFalse(p['plan_change_confirmed'])
        self.assertIsNone(d.feed(new,(0,0),2.1,'g',(6,0))[0])
        c,_=d.feed(new,(0,0),2.6,'g',(6,0));self.assertTrue(c['plan_change_confirmed']);self.assertEqual(c['anchor_ros_s'],2)
    def test_pending_is_not_rasterized(self):
        b=bundle();s=RerouteSettings();g=geometry_dict(b[1])
        rec={'id':'x','enabled':True,'mask_eligible':False,'point_xy':[.7,0.],'mask_radius_m':s.mask_radius(.25)}
        m,st=rasterize_points(g,[rec],(4.,4.),s);self.assertEqual(st['active_ids'],[]);self.assertFalse((m==100).any())
    def test_world_hash_changes_context_even_same_scene_id(self):
        h=Harness()
        try:
            a=context_for(h.root,'same');(h.root/'gazebo/worlds/disaster/disaster_hospital.sdf').write_text('<sdf>new</sdf>')
            b=context_for(h.root,'same');self.assertNotEqual(json_hash(a),json_hash(b))
        finally:h.close()

class RuntimeTests(unittest.TestCase):
    def setUp(self):self.h=Harness();self.r=self.h.runtime
    def tearDown(self):self.h.close()
    def test_actual_runtime_single_zone_mapping(self):
        self.h.record();self.assertEqual(self.r.store.data['records'][0]['spatial_zone'],'CORRIDOR_B')
        self.assertEqual(list(self.h.n.zones_by_id),['R12'])
    def test_no_navigation_commands_from_record(self):
        self.h.record();self.h.record(True);self.assertEqual(self.h.n.stops,[]);self.assertIsNone(self.h.n.cancel_reason)
    def test_pending_then_changed_sensor_point_preserves_first(self):
        a=self.h.record();raw=Path(a['pending_file']).read_bytes()
        self.h.monitor.hits={'stamp':10.6,'wall':time.monotonic(),'points':np.zeros((1,2))}
        self.h.mods['path_blockage'].path_obstacle_witness=lambda *a,**k:{'xy':[1.2,0.],'scan_supported':True}
        self.h.record(True);d=self.r.summary()['evidence_v1_7']['records'][0]
        self.assertEqual(Path(d['pending_file']).read_bytes(),raw)
        self.assertEqual(d['first_detection']['point_xy'],[.7,0.]);self.assertEqual(self.r.store.data['records'][0]['point_xy'],[1.2,0.])
    def test_summary_keeps_old_interface_and_adds_links(self):
        self.h.record();d=self.r.summary();self.assertIn('points',d);self.assertIn('context_hash',d)
        self.assertEqual(d['spatial_catalog_size'],27);self.assertIn('pending_file',d['evidence_v1_7']['records'][0])
    def test_unconfirmed_stays_in_summary_after_new_goal(self):
        self.h.record();self.h.n.nav_uuid='b'*32;self.r.begin(self.h.inp)
        self.assertEqual(self.r.summary()['evidence_v1_7']['counts']['pending'],1)
    def test_pending_record_survives_store_capacity_failure(self):
        self.r.store.max_records=0
        with self.assertRaises(ValueError):self.h.record()
        self.assertEqual(self.r.evidence.summary()['counts']['pending'],1)
    def test_tick_records_first_accepted_pending(self):
        self.h.feed([(0,0),(6,0)],1.)
        self.h.feed([(0,0),(0,3),(6,3),(6,0)],2.)
        self.assertEqual(self.r.evidence.summary()['counts']['pending'],1)
    def test_replaced_route_retains_both_pending_events(self):
        self.h.feed([(0,0),(6,0)],1.)
        self.h.feed([(0,0),(0,3),(6,3),(6,0)],2.)
        self.h.feed([(0,0),(0,-3),(6,-3),(6,0)],2.6)
        self.assertEqual(self.r.evidence.summary()['counts'],{'total':2,'pending':2,'confirmed':0})
    def test_below_threshold_cannot_erase_pending_evidence(self):
        self.h.feed([(0,0),(6,0)],1.);self.h.feed([(0,0),(0,3),(6,3),(6,0)],2.)
        self.h.feed([(0,0),(6,0)],2.6)
        self.assertIsNone(self.r.detector.pending);self.assertEqual(self.r.evidence.summary()['counts']['pending'],1)
    def test_duplicate_plan_stamp_not_confirmed(self):
        self.h.feed([(0,0),(6,0)],1.);p=[(0,0),(0,3),(6,3),(6,0)]
        self.h.feed(p,2.);self.h.feed(p,2.)
        self.assertEqual(self.r.evidence.summary()['counts']['confirmed'],0)
    def test_exact_time_tf_missing_is_not_replaced_by_current_pose(self):
        self.r._pose_at=lambda stamp:(_ for _ in ()).throw(RuntimeError('no TF'))
        self.h.n.sim=10.;self.r.on_plan([(0,0),(6,0)],10.);self.r.queue[0]['received']=time.monotonic()-1.2
        self.r.tick(self.h.inp);self.assertEqual(self.r.comparisons,0)
        self.assertIn('REROUTE_SAMPLE_SKIPPED',[x[0] for x in self.h.n.events])
    def test_foreign_goal_is_ignored(self):
        self.h.n.foreign=True;self.r.on_plan([(0,0),(6,0)],10.);self.assertEqual(len(self.r.queue),0)
    def test_canceling_goal_is_ignored(self):
        self.h.n.cancel_reason='USER';self.r.on_plan([(0,0),(6,0)],10.);self.assertEqual(len(self.r.queue),0)
    def test_stale_path_is_ignored(self):
        self.h.n.sim=10.;self.r.on_plan([(0,0),(6,0)],7.);self.r.tick(self.h.inp);self.assertEqual(self.r.comparisons,0)
    def test_epoch_mismatch_is_ignored(self):
        self.r.on_plan([(0,0),(6,0)],10.);self.r.tick(dict(self.h.inp,epoch=2));self.assertEqual(self.r.comparisons,0)
    def test_observe_readiness_and_apply_gate_are_unchanged(self):
        self.assertEqual(self.r.readiness(self.h.inp),(True,'OBSERVE_ONLY'))
        self.r.mode='apply';self.assertEqual(self.r.readiness(self.h.inp),(False,'WAIT_REROUTE_MASK_SERVER'))

@unittest.skipUnless(os.environ.get('DN012_V17_ORIGINAL_RUNTIME'),'original runtime not staged')
class OriginalParityTests(unittest.TestCase):
    def test_latest_operational_memory_and_mask_same_except_zone_label(self):
        spec=importlib.util.spec_from_file_location('dn012_original_runtime',os.environ['DN012_V17_ORIGINAL_RUNTIME'])
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        values=[]
        for cls in (original.RerouteRuntime,RerouteRuntime):
            h=Harness(cls)
            try:
                for confirmed in (False,True):
                    e=event(confirmed);h.n.sim=10.+(.6 if confirmed else 0.)
                    h.runtime._record(e,h.inp,e['confirm_metric']['new_remaining'])
                record=copy.deepcopy(h.runtime.store.data['records'][0])
                record.pop('first_seen_utc');record.pop('spatial_zone')
                mask,status=rasterize_points(geometry_dict(h.b[1]),h.runtime.store.data['records'],(4.,4.),h.s)
                values.append((record,mask,status))
            finally:h.close()
        self.assertEqual(values[0][0],values[1][0]);np.testing.assert_array_equal(values[0][1],values[1][1])
        self.assertEqual(values[0][2],values[1][2])

# These cases run on the user's machine with the full actual data bundle;
# they are explicitly skipped when that project has not been supplied locally.
@unittest.skipUnless(os.environ.get('DN012_V17_REAL_PROJECT'),'full local project not mounted')
class ActualBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project=Path(os.environ['DN012_V17_REAL_PROJECT']);sys.path.insert(0,str(project/'scripts/exploration'))
        from hospital_coverage_core import load_bundle
        cls.loaded=[]
        for rel in ('config/exploration','config/exploration/navigation'):
            base=project/rel;zp=base/'hospital_zones.json';rp=base/'hospital_coverage_reference.json.gz'
            if zp.is_file() and rp.is_file():cls.loaded.append((rel,load_bundle(zp,rp)))
        if not cls.loaded:raise AssertionError('NO_ACTUAL_ZONE_BUNDLE')
    def test_original_bundle_validation_and_27_catalog(self):
        for rel,b in self.loaded:
            self.assertEqual(len(b[0]['zones']),27,rel)
            self.assertEqual((b[1].width,b[1].height),(1127,507),rel)
    def test_each_actual_zone_representative_cell_maps_to_same_name(self):
        for rel,b in self.loaded:
            q=ZoneLookup(b)
            for z in b[0]['zones']:
                js,is_=np.nonzero(b[2]==z['mask_id']);self.assertGreater(len(js),0)
                k=len(js)//2;point=b[1].cell_to_world(int(is_[k]),int(js[k]))
                self.assertEqual(q.locate(point)['zone_id'],z['id'],(rel,z['id']))
    def test_both_actual_historical_corridor_coordinates(self):
        for rel,b in self.loaded:
            q=ZoneLookup(b)
            for p in ((40.564,5.275),(41.564,3.475)):
                self.assertEqual(q.locate(p)['zone_id'],'CORRIDOR_B',(rel,p))

class SessionMemoryTests(unittest.TestCase):
    def setUp(self):
        from region_runtime import make_runtime
        from resume_runtime import _restore_legacy_detector_memories, capture_mission
        self.restore=_restore_legacy_detector_memories
        self.capture=capture_mission
        self.region_factory=make_runtime
        self.h=Harness()
        self.addCleanup(self.h.close)

    def region(self, output):
        class Base:
            def __init__(self,node,bundle,cfg,args,output):
                self.n=node;self.bundle=bundle;self.cfg=cfg;self.args=args;self.output=output
        with patch.dict(os.environ,{'DN012_REGION_TWO_PASS':'1','DN012_BLOCKAGE_MEMORY':'1'}):
            return self.region_factory(Base)(self.h.n,self.h.b,object(),self.h.runtime.args,output)

    def next_run(self):
        output=self.h.root/'next_session';output.mkdir()
        runtime=RerouteRuntime(self.h.monitor,self.h.n,self.h.b,object(),self.h.runtime.args,output)
        self.addCleanup(runtime.close)
        return runtime

    def test_new_run_ignores_previous_and_global_reroute_records(self):
        self.h.record(confirmed=True)
        old_bytes=self.h.runtime.store.path.read_bytes()
        legacy=self.h.root/'.dn012_reroute_memory'
        legacy.mkdir();(legacy/('points_'+self.h.runtime.store.context_hash[:24]+'.json')).write_bytes(old_bytes)
        second=self.next_run()
        self.assertEqual(second.store.data['records'],[])
        self.assertEqual(second.store.path,second.output/'memory/reroute_points.json')
        self.assertEqual(self.h.runtime.store.path.read_bytes(),old_bytes)

    def test_new_run_ignores_previous_and_global_blockage_records(self):
        first=self.region(self.h.out).blockage_memory
        first.detect({'xy':[0.,0.]},goal_id='first',ros_time_s=10.)
        old_bytes=first.path.read_bytes()
        legacy=self.h.root/'.dn012_blockage_memory';legacy.mkdir()
        name='map_'+str(self.h.b[4])[:16]+'_domain_'+os.environ.get('ROS_DOMAIN_ID','0')+'_j100.json'
        (legacy/name).write_bytes(old_bytes)
        second=self.region(self.h.root/'next_session').blockage_memory
        self.assertEqual(second.records,[])
        self.assertTrue(second.path.is_file())
        self.assertEqual(first.path.read_bytes(),old_bytes)

    def test_checkpoint_capture_and_restore_both_memories(self):
        self.h.record(confirmed=True)
        first=self.region(self.h.out).blockage_memory
        first.detect({'xy':[0.,0.]},goal_id='original-goal',ros_time_s=10.)
        n=self.h.n
        n.connectivity=types.SimpleNamespace(blockage_memory=first)
        n.path_blockage=types.SimpleNamespace(reroute=self.h.runtime)
        n.book=types.SimpleNamespace(records={},visit_state={},visits=[],rejected=[],progress=0,threshold=.96)
        n.mission_guard=None
        from zone_lifecycle import ZoneLifecycle
        lifecycle=ZoneLifecycle(n.zones_by_id)
        n._v187_lifecycle=types.SimpleNamespace(snapshot=lambda node:lifecycle.snapshot())
        for key in ('sent_goals','plan_attempts','succeeded_goals','aborted_goals','coverage_cancels'):
            setattr(n,key,0)
        mission=self.capture(n,{'execute':True,'source_session_v18':self.h.out.name})
        self.assertEqual(mission['detector_memories']['blockage'],first.records)
        second=self.next_run();second_memory=self.region(second.output).blockage_memory
        node=types.SimpleNamespace(connectivity=types.SimpleNamespace(blockage_memory=second_memory),
            path_blockage=types.SimpleNamespace(reroute=second))
        before=(first.path.read_bytes(),self.h.runtime.store.path.read_bytes())
        self.restore(node,mission)
        self.assertEqual(second_memory.records,first.records)
        self.assertEqual(second.store.data['records'],self.h.runtime.store.data['records'])
        self.assertEqual(second_memory.records[0]['goal_ids'],['original-goal'])
        self.assertEqual(before,(first.path.read_bytes(),self.h.runtime.store.path.read_bytes()))
        self.assertEqual(json.loads(second_memory.path.read_text())['records'],first.records)
        self.assertEqual(json.loads(second.store.path.read_text())['records'],second.store.data['records'])
        self.assertIsNone(second.detector.goal_id)

    def test_legacy_checkpoint_restores_only_its_embedded_snapshot(self):
        self.h.record(confirmed=True)
        first=self.region(self.h.out).blockage_memory
        first.detect({'xy':[0.,0.]},goal_id='old-goal',ros_time_s=10.)
        mission={'original_summary':{'region_review':{'blockage_memory':first.summary()},
                                    'reroute_memory_v1_4':self.h.runtime.summary()}}
        second=self.next_run();memory=self.region(second.output).blockage_memory
        node=types.SimpleNamespace(connectivity=types.SimpleNamespace(blockage_memory=memory),
            path_blockage=types.SimpleNamespace(reroute=second))
        self.restore(node,mission)
        self.assertEqual([p['id'] for p in memory.records],[p['id'] for p in first.records])
        self.assertEqual(memory.records[0]['goal_ids'],[])
        self.assertEqual(second.store.data['records'],self.h.runtime.store.data['records'])

    def test_invalid_checkpoint_point_is_rejected_without_writing(self):
        self.h.record(confirmed=True)
        invalid=copy.deepcopy(self.h.runtime.store.data['records'])
        invalid[0]['point_xy']=[9999.,9999.]
        second=self.next_run();before=second.store.path.read_bytes()
        with self.assertRaises(ValueError):second.store.restore_records(invalid)
        self.assertEqual(second.store.path.read_bytes(),before)
        memory=self.region(second.output).blockage_memory
        before=memory.path.read_bytes()
        with self.assertRaises(ValueError):memory.restore_records([{'id':'bad','xy':[9999.,9999.]}])
        self.assertEqual(memory.path.read_bytes(),before)

    def test_mask_status_from_another_session_is_ignored(self):
        runtime=self.h.runtime
        status=dict(schema='dn012_reroute_mask_status_v1',context_hash=runtime.store.context_hash,
                    memory_file=str(self.h.root/'other/memory/reroute_points.json'))
        runtime.on_status(types.SimpleNamespace(data=json.dumps(status)))
        self.assertIsNone(runtime.status)
        status['memory_file']=str(runtime.store.path.resolve())
        runtime.on_status(types.SimpleNamespace(data=json.dumps(status)))
        self.assertEqual(runtime.status,status)

    def test_inspector_defaults_to_latest_without_global_fallback(self):
        from contextlib import redirect_stdout
        path=Path(__file__).resolve().parents[2]/'scripts/exploration/navigation/reports/inspect_results.py'
        spec=importlib.util.spec_from_file_location('session_memory_inspector',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.h.record(confirmed=True)
        parent=self.h.root/'logs/hospital_zone_mission'
        old=parent/'session_20260101';old.mkdir(parents=True)
        (old/'memory').mkdir()
        (old/'memory/reroute_points.json').write_bytes(self.h.runtime.store.path.read_bytes())
        new=parent/'session_20260102';new.mkdir()
        text=io.StringIO()
        with redirect_stdout(text):module.memory_main(['--project',str(self.h.root)])
        self.assertIn(str(new),text.getvalue())
        self.assertIn('NO_REROUTE_POINT_FILE_YET',text.getvalue())
        text=io.StringIO()
        with redirect_stdout(text):module.memory_main(['--project',str(self.h.root),'--session',str(old)])
        self.assertIn('POINTS: 1',text.getvalue())

if __name__=='__main__':unittest.main(verbosity=2)
