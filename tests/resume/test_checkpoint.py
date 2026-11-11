"""Pure and simulated-interface regressions. NOT ROS/DDS/Gazebo or motion tests."""
import copy
from dataclasses import dataclass
import gzip
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from scipy import ndimage

import checkpoint as cp
import exploration_results as un
from resume_runtime import restore_mission, prepare_mission_args

# This standalone fallback is used ONLY in the author's isolated pure tests.
# The installer's --check stage imports the user's real regions module.
try:
    import regions
    REAL_REGIONS = True
except ImportError:
    if os.environ.get('DN012_REQUIRE_REAL_REGIONS') == '1':raise
    REAL_REGIONS = False
    mod = types.ModuleType('regions')
    def components(mask, g, minimum=0.):
        labels, _ = ndimage.label(mask,ndimage.generate_binary_structure(2,1))
        result=[]
        for ident in range(1,int(labels.max())+1):
            ids=np.flatnonzero(labels==ident).astype(np.int32)
            area=len(ids)*g.resolution**2
            if area+1e-9<minimum:continue
            yy,xx=np.divmod(ids,g.width)
            result.append(dict(region_id=cp.sha(ids.tobytes())[:16],cells=len(ids),area_m2=area,
                indices=ids,centroid_map_m=list(g.cell_to_world(float(xx.mean()),float(yy.mean()))),
                bbox_cells_ros=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]))
        return sorted(result,key=lambda x:(-x['cells'],x['region_id']))
    def row_runs(ids,width):
        out=[]
        for idx in sorted(ids.tolist()):
            j,i=divmod(idx,width)
            if out and out[-1][0]==j and out[-1][2]==i:out[-1][2]=i+1
            else:out.append([j,i,i+1])
        return out
    mod.components=components
    mod.public_region=lambda r:{k:v for k,v in r.items() if k!='indices'}
    mod.row_runs=row_runs
    mod.zone_polygon_domain=lambda z,g,f:(np.asarray(z.get('_domain',f),dtype=bool),'SEMANTIC_ZONE_POLYGON')
    sys.modules['regions']=mod

@dataclass
class G:
    width:int=100
    height:int=60
    resolution:float=.1
    origin:tuple=(0.,0.,0.)
    def contains(self,i,j):return 0<=i<self.width and 0<=j<self.height
    def cell_to_world(self,i,j):return ((i+.5)*self.resolution,(j+.5)*self.resolution)
    def world_to_grid_float(self,x,y):return (x/self.resolution,y/self.resolution)

def fixture():
    g=G();known=np.ones((g.height,g.width),bool)
    known[[0,-1],:]=False;known[:,[0,-1]]=False
    labels=np.zeros(known.shape,np.uint16);labels[known]=1
    labels[10:50,60:95]=2
    zones=[dict(id='LOBBY',mask_id=1,kind='lobby'),dict(id='ROOM',mask_id=2,kind='room')]
    for z in zones:z['known_free_cells']=int((labels==z['mask_id']).sum())
    return (dict(zones=zones),g,labels,known,'verified-test-bundle')

def status(b,mask,sid='old-session'):
    zones=[dict(id=z['id'],reference_free_cells=int((b[2]==z['mask_id']).sum()),
                observed_cells=int(((b[2]==z['mask_id'])&mask).sum())) for z in b[0]['zones']]
    return dict(session_id=sid,zone_bundle_signature=b[4],reference_free_survey_cells=int((b[2]>0).sum()),
                observed_cells=int(mask.sum()),zones=zones,data_health='LIVE',map_validated=True,fault=None,
                evidence_scans=10,last_evidence_scan_end_s=10.)

class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.b=fixture();self.mask=np.zeros_like(self.b[3]);self.mask[12:40,61:80]=True
        self.status=status(self.b,self.mask)
        self.cov=cp.coverage_payload(self.b,self.mask,self.status)
        self.payload=dict(schema=cp.SCHEMA,context={'verified':'yes'},coverage=self.cov,mission=None)
        self.tmp=tempfile.TemporaryDirectory();self.dir=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_mask_roundtrip(self):np.testing.assert_array_equal(cp.validate_coverage(self.cov,self.b),self.mask)
    def test_empty_mask_roundtrip(self):np.testing.assert_array_equal(cp.mask_from_runs([],5,5),np.zeros((5,5),bool))
    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):cp.mask_from_runs([[1,1,4],[1,3,5]],6,6)
    def test_out_of_grid_rejected(self):
        with self.assertRaises(ValueError):cp.mask_from_runs([[6,1,4]],6,6)
    def test_unsorted_rejected(self):
        with self.assertRaises(ValueError):cp.mask_from_runs([[2,1,2],[1,1,2]],6,6)
    def test_float_run_rejected(self):
        with self.assertRaises(ValueError):cp.mask_from_runs([[1.,1,4]],6,6)
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):cp.encode({'x':float('nan')})
    def test_unknown_type_rejected(self):
        with self.assertRaises(TypeError):cp.encode({'x':object()})
    def test_counts_disagree_rejected(self):
        s=copy.deepcopy(self.status);s['observed_cells']+=1
        with self.assertRaises(ValueError):cp.coverage_payload(self.b,self.mask,s)
    def test_zone_counts_disagree_rejected(self):
        s=copy.deepcopy(self.status);s['zones'][0]['observed_cells']+=1
        with self.assertRaises(ValueError):cp.coverage_payload(self.b,self.mask,s)
    def test_coverage_outside_mask_rejected(self):
        a=self.mask.copy();a[0,0]=True
        with self.assertRaises(ValueError):cp.coverage_payload(self.b,a,status(self.b,a))
    def test_bundle_mismatch_rejected(self):
        p=copy.deepcopy(self.cov);p['zone_bundle_signature']='wrong'
        with self.assertRaises(ValueError):cp.validate_coverage(p,self.b)
    def test_geometry_mismatch_rejected(self):
        p=copy.deepcopy(self.cov);p['geometry']['resolution']=.2
        with self.assertRaises(ValueError):cp.validate_coverage(p,self.b)
    def test_atomic_roundtrip_and_pointer(self):
        path,digest=cp.save_checkpoint(self.dir,self.payload)
        p,h,_=cp.load_checkpoint(self.dir);self.assertEqual(h,digest);self.assertEqual(p['coverage'],self.cov)
    def test_hash_corruption_refused(self):
        path,_=cp.save_checkpoint(self.dir,self.payload);env=cp.read_json(path)
        env['payload']['coverage']['observed_cells']+=1
        path.write_bytes(gzip.compress(cp.encode(env)))
        with self.assertRaises(ValueError):cp.load_checkpoint(path)
        with self.assertRaises(ValueError):cp.load_checkpoint(self.dir)
    def test_pointer_traversal_refused(self):
        (self.dir/'LATEST.json').write_text(json.dumps({'checkpoint':'../x','sha256':'x'}))
        with self.assertRaises(ValueError):cp.load_checkpoint(self.dir)
    def test_retains_previous_complete_snapshots(self):
        for _ in range(5):cp.save_checkpoint(self.dir,self.payload)
        self.assertEqual(len(list(self.dir.glob('checkpoint_*.json.gz'))),3)
        cp.load_checkpoint(self.dir)
    def test_context_change_refused(self):
        with self.assertRaises(ValueError):cp.assert_context(self.payload,{'verified':'changed'})
    def test_superset_allowed(self):
        live=self.mask.copy();live[20,30]=True;cp.compatible_live_coverage(self.mask,live,status(self.b,live))
    def test_lost_saved_cells_refused(self):
        live=self.mask.copy();live[20,65]=False
        with self.assertRaises(ValueError):cp.compatible_live_coverage(self.mask,live,status(self.b,live))
    def test_stale_status_not_restorable(self):
        s=dict(self.status,data_health='STALE')
        with self.assertRaises(ValueError):cp.compatible_live_coverage(self.mask,self.mask,s)
    def test_restore_does_not_reuse_old_clock_or_completion(self):
        c=types.SimpleNamespace(denominators=np.bincount(self.b[2].ravel()),scan_count=100,
            evidence_scan_count=100,last_scan_end_s=9000.,threshold_since={1:9000.},
            threshold_scan_count={1:80},threshold_confirmed={1})
        cp.restore_coverage(c,self.cov,self.b)
        self.assertEqual(c.observed_total,int(self.mask.sum()))
        self.assertIsNone(c.last_scan_end_s);self.assertFalse(c.threshold_confirmed)
        self.assertEqual(c.evidence_scan_count,0)
    def test_legacy_import_session_and_count(self):
        d=self.dir/'old-session';d.mkdir()
        (d/'coverage_status.json').write_bytes(cp.encode(self.status))
        mask=dict(schema='observed_mask_ros_row_runs_v1',row_order='ROS bottom-up',
            zone_bundle_signature=self.b[4],width=100,height=60,observed_cells=int(self.mask.sum()),runs=cp.runs_from_mask(self.mask))
        (d/'observed_mask.json.gz').write_bytes(gzip.compress(cp.encode(mask)))
        self.assertEqual(cp.import_legacy_coverage(d,self.b)['observed_cells'],int(self.mask.sum()))
    def test_area_policy_not_other_parameters(self):
        with patch.dict(os.environ,{'DN012_REGION_MIN_AREA_M2':'.5','DN012_SECOND_GOAL_MARGIN_M':'.1'}):
            cp.enforce_area_policy();self.assertEqual(os.environ['DN012_REGION_MIN_AREA_M2'],'1.0')
            self.assertEqual(os.environ['DN012_SECOND_GOAL_MARGIN_M'],'.1')

class BlockageTests(unittest.TestCase):
    def setUp(self):
        self.b=fixture();self.g=self.b[1];self.cost=np.zeros_like(self.b[2],dtype=np.int16)
        self.cfg=types.SimpleNamespace(robot_radius_m=math.hypot(.229,.208),safety_margin_m=.04)
        self.inp=dict(cost={'values':self.cost,'key':'c1'},pose=(2.,3.,0.),observed=np.zeros_like(self.b[3]))
        self.s=types.SimpleNamespace(min_area_m2=1.,max_access_regions=64)
    def assess(self,zone=None):return un.assess_access(self.b,self.inp,zone or self.b[0]['zones'][1],np.zeros_like(self.b[3]),self.cfg,self.s,True)
    def test_full_barrier_is_inaccessible(self):
        self.cost[:,49:52]=100
        r=self.assess();self.assertEqual(r['zone_access_status'],'INACCESSIBLE');self.assertTrue(r['global_regions'])
    def test_no_invented_exterior_detour(self):
        self.cost[:,49:52]=100;r=self.assess()
        self.assertFalse(r['unmapped_exterior_detours_allowed']);self.assertEqual(r['status'],'INACCESSIBLE')
    def test_global_detour_not_zone_inaccessible(self):
        self.cost[18:42,49:52]=100
        self.assertNotEqual(self.assess()['status'],'INACCESSIBLE')
    def test_inflation_not_obstacle(self):
        self.cost[:,49:52]=99
        self.assertFalse(self.assess()['global_regions'])
    def test_unknown_not_obstacle(self):
        self.cost[:,49:52]=-1
        self.assertFalse(self.assess()['global_regions'])
    def test_unsafe_start_not_inaccessible_zone(self):
        self.cost[25:35,15:25]=100
        self.assertEqual(self.assess()['status'],'ACCESS_NOT_ASSESSED')
    def test_disabled_fixed_topology_not_a_finding(self):
        r=un.assess_access(self.b,self.inp,self.b[0]['zones'][1],np.zeros_like(self.b[3]),self.cfg,self.s,False)
        self.assertEqual(r['reason'],'FIXED_PRIOR_NOT_ENABLED')
    def test_sub_1m2_regions_not_selected(self):
        self.cost[:,49:52]=100
        # 0.5 m2 target on the disconnected side, independent from total room.
        self.b[2][self.b[2]==2]=1;self.b[2][20:25,65:75]=2
        self.assertEqual(self.assess()['status'],'BELOW_SIGNIFICANT_AREA')
        self.assertFalse(self.assess()['global_regions'])
    def test_1m2_inclusive(self):
        self.cost[:,49:52]=100;self.b[2][self.b[2]==2]=1;self.b[2][20:30,65:75]=2
        self.assertEqual(self.assess()['status'],'INACCESSIBLE')
    def test_pending_and_confirmation_preserved(self):
        r=un.BlockageRegistry(self.b)
        r.observe('REROUTE','id',[2.,3.],{'stage':0},'ROOM','PENDING')
        r.observe('REROUTE','id',[2.1,3.],{'stage':1},'ROOM','CONFIRMED')
        self.assertEqual(len(r.public()),1);self.assertEqual(len(r.public()[0]['observations']),2)
        self.assertEqual(r.public()[0]['xy'],[2.,3.])
    def test_reroute_and_path_one_collection(self):
        r=un.BlockageRegistry(self.b)
        r.observe('REROUTE','r',[2.,3.],{},'ROOM')
        r.observe('PATH_BLOCKAGE','p',[2.05,3.],{},'ROOM')
        self.assertEqual(len(r.public()),1)
        self.assertEqual(set(r.public()[0]['sources']),{'REROUTE','PATH_BLOCKAGE'})
        self.assertEqual(r.public()[0]['display_label'],'차단지역')
    def test_repeat_summary_does_not_duplicate(self):
        r=un.BlockageRegistry(self.b)
        for _ in range(5):r.observe('REROUTE','r',[2.,3.],{'a':1},'ROOM')
        self.assertEqual(len(r.public()[0]['observations']),1)
    def test_point_does_not_become_area(self):
        r=un.BlockageRegistry(self.b);r.observe('REROUTE','r',[2.,3.],{},'ROOM')
        p=r.public()[0];self.assertEqual(p['geometry_kind'],'POINT');self.assertNotIn('area_m2',p)
    def test_all_zone_lookup_not_mission_subset(self):
        self.assertEqual(un.locate(self.b,[7.,3.]),'ROOM')
        self.assertEqual(un.locate(self.b,[2.,3.]),'LOBBY')
    def test_nonfinite_point_rejected(self):
        r=un.BlockageRegistry(self.b);r.observe('REROUTE','bad',[float('nan'),0.],{})
        self.assertFalse(r.public())
    def test_pending_in_public_projection(self):
        def legacy(*a,**k):return dict(categories={x:[] for x in ['blocked_points','inaccessible_regions','inaccessible_zones','robot_unexplorable_regions']},category_labels={})
        result=un._project_results_with_baseline(legacy,{}, {},reroute_summary={'points':[{'id':'r','point_xy':[2.,3.],'plan_change_confirmed':False}]})
        self.assertEqual(len(result['blocked_areas']),1)
        self.assertEqual(result['blocked_areas'][0]['detection_stage'],'PENDING')
    def test_current_zone_inaccessible_in_projection(self):
        self.cost[:,49:52]=100;access=self.assess()
        def legacy(*a,**k):return dict(categories={x:[] for x in ['blocked_points','inaccessible_regions','inaccessible_zones','robot_unexplorable_regions']},category_labels={})
        result=un._project_results_with_baseline(legacy,{'zones':{'ROOM':{'access':access}}},{})
        self.assertEqual(result['categories']['inaccessible_zones'][0]['access_status'],'INACCESSIBLE')

class ResumeInterfaceTests(unittest.TestCase):
    def make(self):
        b=fixture();mask=b[3].copy();s=status(b,mask,'new-session')
        records={z['id']:types.SimpleNamespace(state='UNOBSERVED',measured=0.,views=0,low_gain_count=0,deferred_reason=None,tracker=object()) for z in b[0]['zones']}
        visits={z['id']:types.SimpleNamespace(entered=False,entry_goals=0,observation_goals=0,low_gain=0,retries=0,entry_arrival_retries=0) for z in b[0]['zones']}
        book=types.SimpleNamespace(records=records,visit_state=visits,session=None,visits=[],rejected=[],progress=0,
            active_zone=None,locked_zone=None,deferred_this_mission={'ROOM'},deferral_history={},outcomes={},nav_abort_counts={},threshold=.96)
        n=types.SimpleNamespace(book=book,zones_by_id={z['id']:z for z in b[0]['zones']},nav_uuid=None,nav_handle=None,plan_handle=None,connectivity=None,mission_guard=None)
        m=dict(scope=list(n.zones_by_id),threshold=.96,records={'ROOM':{'views':3,'state':'OBSERVATION_SUFFICIENT'}},
            counters=dict(sent_goals=43,plan_attempts=88,succeeded_goals=40,aborted_goals=1,coverage_cancels=0),
            visits=[{'id':'past-goal'}],rejected=[{'id':'bad-goal'}],deferral_history={'ROOM':[{'reason':'old'}]},
            interrupted_goal_id='must-not-replay')
        return b,n,m,dict(status=s,observed=mask)
    def test_restores_counts_not_goal(self):
        b,n,m,inp=self.make();restore_mission(n,m,inp,b)
        self.assertEqual(n.sent_goals,43);self.assertIsNone(n.nav_uuid);self.assertIsNone(n.current)
        self.assertEqual(n.book.visits[0]['id'],'past-goal')
    def test_completion_requires_fresh_tracker_again(self):
        b,n,m,inp=self.make();restore_mission(n,m,inp,b)
        self.assertEqual(n.book.records['ROOM'].state,'THRESHOLD_PENDING')
        self.assertEqual(n.book.records['ROOM'].measured,1.)
    def test_explicit_resume_preserves_old_scheduling_lock(self):
        b,n,m,inp=self.make()
        m['records']['ROOM'].update(state='PARTIAL_DEFERRED',deferred_reason='PATH_BLOCKED_NO_CONFIRMED_DETOUR')
        m['deferred_this_mission']=['ROOM']
        restore_mission(n,m,inp,b)
        self.assertEqual(n.book.deferred_this_mission,{'ROOM'})
        self.assertEqual(n.book.records['ROOM'].state,'PARTIAL_DEFERRED')
        self.assertEqual(n.book.records['ROOM'].deferred_reason,'PATH_BLOCKED_NO_CONFIRMED_DETOUR')
        self.assertEqual(n.book.deferral_history['ROOM'][0]['reason'],'old')
    def test_failed_goals_retained_as_history_not_stale_holds(self):
        b,n,m,inp=self.make();restore_mission(n,m,inp,b)
        self.assertFalse(n.book.rejected);self.assertEqual(n._v18_historical_rejected[0]['id'],'bad-goal')
    def test_active_goal_blocks_restore(self):
        b,n,m,inp=self.make();n.nav_uuid='still-running'
        with self.assertRaises(ValueError):restore_mission(n,m,inp,b)
    def test_scope_mismatch_blocks_restore(self):
        b,n,m,inp=self.make();m['scope']=['ROOM']
        with self.assertRaises(ValueError):restore_mission(n,m,inp,b)
    def test_bad_counter_blocks_restore(self):
        b,n,m,inp=self.make();m['counters']['sent_goals']=-1
        with self.assertRaises(ValueError):restore_mission(n,m,inp,b)



class ResumeNoRepeatRegressionTests(unittest.TestCase):
    """Synthetic reproduction of the reported Nov-10 15/11/1 checkpoint."""
    ZONES = ('R01','R02A','R02B','R03','R04','R05','R06A','R06B','R07A','R07B',
        'R08A','R08B','R09','R10A','R10B','R11','R12','R13','R14_T','R15_T',
        'LOBBY','CORRIDOR_B','CONNECTOR_1','CONNECTOR_2','CORRIDOR_C','CORRIDOR_D','CORRIDOR_E')
    DEFERRED = {'R02B','R04','R06A','R07B','R08A','R09','R10A','R10B','R11','R12','R14_T'}
    REVIEWED = {'R01','R02A','R02B','R03','R04','R05','R06A','R06B','R07A','R07B',
        'R08A','R08B','R09','R10A','R10B','R11','R12','R13','R14_T','R15_T'}
    def make(self):
        s=[];saved={};records={};visit={};zone_map={}
        for zid in self.ZONES:
            complete=zid not in self.DEFERRED and zid!='CORRIDOR_E'
            state=('OBSERVATION_SUFFICIENT' if complete else
                'PARTIAL_DEFERRED' if zid in self.DEFERRED else 'PARTIAL')
            value=99 if complete else 2 if zid in self.DEFERRED else 70
            s.append(dict(id=zid,observed_cells=value,reference_free_cells=100))
            saved[zid]=dict(state=state,measured=value/100,views=2,low_gain_count=0,
                deferred_reason='PATH_BLOCKED_NO_CONFIRMED_DETOUR' if zid in self.DEFERRED else None)
            records[zid]=types.SimpleNamespace(state='UNOBSERVED',measured=0.,views=0,
                low_gain_count=0,deferred_reason=None)
            visit[zid]=types.SimpleNamespace(entered=False,entry_goals=0,observation_goals=0,
                low_gain=0,retries=0,entry_arrival_retries=0)
            zone_map[zid]={'id':zid}
        book=types.SimpleNamespace(records=records,visit_state=visit,session=None,visits=[],
            rejected=[],progress=0,active_zone=None,locked_zone=None,deferred_this_mission=set(),
            deferral_history={},outcomes={},nav_abort_counts={},threshold=.96)
        runtime=types.SimpleNamespace(tasks=None,failed_entries={},local_reviewed=set(),total_probes=0)
        node=types.SimpleNamespace(book=book,zones_by_id=zone_map,nav_uuid=None,nav_handle=None,
            nav_send=None,plan_handle=None,plan_send=None,plan_result=None,connectivity=runtime,
            mission_guard=None)
        mission=dict(scope=list(self.ZONES),records=saved,
            counters=dict(sent_goals=50,plan_attempts=127,succeeded_goals=43,
                aborted_goals=0,coverage_cancels=0),visits=[],rejected=[],
            deferred_this_mission=sorted(self.DEFERRED),
            connectivity=dict(local_reviewed=sorted(self.REVIEWED),stage='PRIMARY'))
        inputs=dict(status=dict(session_id='live',zones=s),cost={'key':'fresh-cost'},
            observed=np.zeros((2,2),dtype=bool))
        return node,mission,inputs
    def test_saved_15_sufficient_11_deferred_1_partial(self):
        n,m,inp=self.make();restore_mission(n,m,inp,(None,None,None,None,None))
        observed={z:r.state for z,r in n.book.records.items()}
        from collections import Counter
        self.assertEqual(Counter(observed.values()),{
            'THRESHOLD_PENDING':15,'PARTIAL_DEFERRED':11,'PARTIAL':1})
        self.assertEqual(n.book.deferred_this_mission,self.DEFERRED)
        self.assertEqual(n.connectivity.local_reviewed,self.REVIEWED)
        self.assertEqual((n.sent_goals,n.plan_attempts,n.succeeded_goals),(50,127,43))
        self.assertEqual(n.book.records['R10B'].deferred_reason,'PATH_BLOCKED_NO_CONFIRMED_DETOUR')
        # The legacy no-auto-revisit policy can only select CORRIDOR_E here.
        eligible=[z for z,r in n.book.records.items()
                 if r.state not in ('OBSERVATION_SUFFICIENT','THRESHOLD_PENDING',
                                    'PARTIAL_DEFERRED','OBSERVATION_LIMITED')
                 and z not in n.book.deferred_this_mission]
        self.assertEqual(eligible,['CORRIDOR_E'])
    def test_inconsistent_deferred_state_fail_closed_before_mutation(self):
        n,m,inp=self.make();m['records']['R10B']['deferred_reason']=None
        with self.assertRaisesRegex(ValueError,'RESUME_DEFERRED_REASON_MISSING'):
            restore_mission(n,m,inp,(None,None,None,None,None))
        self.assertIsNone(n.book.session)
    def test_complete_without_live_96_percent_refused(self):
        n,m,inp=self.make();inp['status']['zones'][0]['observed_cells']=10
        with self.assertRaisesRegex(ValueError,'RESUME_SAVED_COMPLETE_ZONE_LOST_COVERAGE'):
            restore_mission(n,m,inp,(None,None,None,None,None))
        self.assertIsNone(n.book.session)
    def test_unknown_reviewed_zone_rejected(self):
        n,m,inp=self.make();m['connectivity']['local_reviewed'].append('NEVER_A_ZONE')
        with self.assertRaisesRegex(ValueError,'RESUME_REVIEWED_UNKNOWN_ZONE'):
            restore_mission(n,m,inp,(None,None,None,None,None))

class AdditionalRegressionTests(unittest.TestCase):
    def test_duplicate_zone_records_rejected(self):
        b=fixture();mask=b[3].copy();s=status(b,mask);s['zones'].append(copy.deepcopy(s['zones'][0]))
        with self.assertRaises(ValueError):cp.coverage_payload(b,mask,s)
    def test_saved_source_session_mismatch_rejected(self):
        b=fixture();mask=b[3].copy();p=cp.coverage_payload(b,mask,status(b,mask));p['source_session']='other'
        with self.assertRaises(ValueError):cp.validate_coverage(p,b)
    def test_new_pending_evidence_not_lost_without_store_point(self):
        def legacy(*a,**k):return dict(categories={x:[] for x in ['blocked_points','inaccessible_regions','inaccessible_zones','robot_unexplorable_regions']},category_labels={})
        ev={'records':[dict(record_id='pending',status='PENDING',first_detection={'point_xy':[2.,3.]},confirmation=None)]}
        r=un._project_results_with_baseline(legacy,{}, {},reroute_summary={'points':[],'evidence_v1_7':ev})
        self.assertEqual(r['blocked_areas'][0]['id'],'pending')
    def test_historical_path_memory_retained(self):
        b=fixture();r=un.BlockageRegistry(b)
        r.absorb_summary({'region_review':{'blockage_memory':{'points':[{'id':'old','xy':[2.,3.],'destination_zone':'ROOM'}]}}})
        self.assertEqual(len(r.public()),1);self.assertTrue(r.public()[0]['historical'])
    def test_nav_domain_is_not_coverage_mask(self):
        b=fixture();b[3][:,49:52]=False;b[2][:,49:52]=0
        cost=np.zeros_like(b[2]);cfg=types.SimpleNamespace(robot_radius_m=math.hypot(.229,.208),safety_margin_m=.04)
        reference=np.zeros_like(cost);reference[[0,-1],:]=100;reference[:,[0,-1]]=100
        inp=dict(cost={'values':cost,'key':'test'},pose=(2.,3.,0.),reference_occupancy=reference)
        r=un.assess_access(b,inp,b[0]['zones'][1],np.zeros_like(b[3]),cfg,object(),True)
        self.assertNotEqual(r['zone_access_status'],'INACCESSIBLE')
    def test_invalid_reference_rejected(self):
        b=fixture();cost=np.zeros_like(b[2]);cfg=types.SimpleNamespace(robot_radius_m=math.hypot(.229,.208),safety_margin_m=.04)
        inp=dict(cost={'values':cost,'key':'test'},pose=(2.,3.,0.),reference_occupancy=np.ones_like(cost)*999)
        r=un.assess_access(b,inp,b[0]['zones'][1],np.zeros_like(b[3]),cfg,object(),True)
        self.assertEqual(r['reason'],'INVALID_VERIFIED_REFERENCE')
    def test_counter_error_does_not_mutate_live_session(self):
        b,n,m,inp=ResumeInterfaceTests().make();m['counters']['sent_goals']=-1
        with self.assertRaises(ValueError):restore_mission(n,m,inp,b)
        self.assertIsNone(n.book.session)
    def test_missing_old_history_has_all_zone_defaults(self):
        b,n,m,inp=ResumeInterfaceTests().make();restore_mission(n,m,inp,b)
        self.assertEqual(set(n.book.deferral_history),set(n.zones_by_id))
        self.assertEqual(n.book.nav_abort_counts['LOBBY'],0)
    def test_pending_plan_send_blocks_restore(self):
        b,n,m,inp=ResumeInterfaceTests().make();n.plan_send=object()
        with self.assertRaises(ValueError):restore_mission(n,m,inp,b)
    def test_scope_inferred_before_factory(self):
        b,n,m,inp=ResumeInterfaceTests().make();cov=cp.coverage_payload(b,inp['observed'],inp['status'])
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{}):
            path,_=cp.save_checkpoint(d,dict(schema=cp.SCHEMA,coverage=cov,mission=m,context={}))
            a=types.SimpleNamespace(resume_from=path,resume_confirm_scene=True,zone_ids=None,threshold=.96)
            prepare_mission_args(a,b);self.assertEqual(set(a.zone_ids),set(m['scope']))
    def test_resume_needs_scene_confirmation(self):
        b,n,m,inp=ResumeInterfaceTests().make()
        a=types.SimpleNamespace(resume_from=Path('irrelevant'),resume_confirm_scene=False)
        with self.assertRaises(ValueError):prepare_mission_args(a,b)
    def test_waiting_resume_still_advances_clock_health(self):
        from resume_runtime import MissionCheckpoint
        c=MissionCheckpoint.__new__(MissionCheckpoint);c.parent={};c.restored=False
        n=types.SimpleNamespace(stop_requested=None,last_clock=1.,clock_advanced_wall=0.,fault=None,
            ros_s=lambda:20.,info=lambda *a:None,inputs=lambda:('WAIT_SCAN',None))
        c.n=n;self.assertFalse(c.before_poll());self.assertEqual(n.last_clock,20.)
        self.assertGreater(n.clock_advanced_wall,0.)

class InstalledIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(REAL_REGIONS,'requires full installed project sources')
    def test_real_policy_defaults(self):
        from regions import RegionSettings
        from two_phase import TwoPassSettings
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(RegionSettings.from_environment().min_area_m2,1.)
            self.assertEqual(TwoPassSettings.from_environment().min_area_m2,1.)
    @unittest.skipUnless(REAL_REGIONS,'requires full installed project sources')
    def test_real_projection_entry(self):
        import exploration_results
        r=exploration_results.project_results({}, {},reroute_summary={'points':[{'id':'r','point_xy':[2.,3.]}]})
        self.assertEqual(r['version'],cp.VERSION);self.assertEqual(len(r['blocked_areas']),1)
    @unittest.skipUnless(REAL_REGIONS,'requires full installed project sources')
    def test_real_factories_have_resume_flags(self):
        import hospital_zone_coverage
        import zone_node
        self.assertIn('--resume-from',hospital_zone_coverage.parser().format_help())
        self.assertIn('--resume-from',zone_node.parser().format_help())
    @unittest.skipUnless(REAL_REGIONS,'requires full installed project sources')
    def test_real_region_factory_import(self):
        from region_runtime import make_runtime
        from connectivity_runtime import ConnectivityRuntime
        self.assertTrue(callable(make_runtime))
        self.assertTrue(callable(ConnectivityRuntime))

# Retain the distinct legacy fixture and historical expectations.
class LegacyResumeInterfaceTests(unittest.TestCase):
    def make(self):
        b=fixture();mask=b[3].copy();s=status(b,mask,'new-session')
        records={z['id']:types.SimpleNamespace(state='UNOBSERVED',measured=0.,views=0,low_gain_count=0,deferred_reason=None,tracker=object()) for z in b[0]['zones']}
        visits={z['id']:types.SimpleNamespace(entered=False,entry_goals=0,observation_goals=0,low_gain=0,retries=0,entry_arrival_retries=0) for z in b[0]['zones']}
        book=types.SimpleNamespace(records=records,visit_state=visits,session=None,visits=[],rejected=[],progress=0,
            active_zone=None,locked_zone=None,deferred_this_mission={'ROOM'},deferral_history={},outcomes={},nav_abort_counts={})
        n=types.SimpleNamespace(book=book,zones_by_id={z['id']:z for z in b[0]['zones']},nav_uuid=None,nav_handle=None,plan_handle=None,connectivity=None,mission_guard=None)
        m=dict(scope=list(n.zones_by_id),threshold=.96,records={'ROOM':{'views':3,'state':'OBSERVATION_SUFFICIENT'}},
            counters=dict(sent_goals=43,plan_attempts=88,succeeded_goals=40,aborted_goals=1,coverage_cancels=0),
            visits=[{'id':'past-goal'}],rejected=[{'id':'bad-goal'}],deferral_history={'ROOM':[{'reason':'old'}]},
            interrupted_goal_id='must-not-replay')
        return b,n,m,dict(status=s,observed=mask)
    def test_restores_counts_not_goal(self):
        b,n,m,inp=self.make();restore_mission(n,m,inp,b)
        self.assertEqual(n.sent_goals,43);self.assertIsNone(n.nav_uuid);self.assertIsNone(n.current)
        self.assertEqual(n.book.visits[0]['id'],'past-goal')
    def test_completion_requires_fresh_tracker_again(self):
        b,n,m,inp=self.make();restore_mission(n,m,inp,b)
        self.assertEqual(n.book.records['ROOM'].state,'PARTIAL')
        self.assertEqual(n.book.records['ROOM'].measured,1.)
    def test_explicit_resume_releases_old_scheduling_lock(self):
        b,n,m,inp=self.make();restore_mission(n,m,inp,b)
        self.assertFalse(n.book.deferred_this_mission)
        self.assertEqual(n.book.deferral_history['ROOM'][0]['reason'],'old')
    def test_failed_goals_retained_as_history_not_stale_holds(self):
        b,n,m,inp=self.make();restore_mission(n,m,inp,b)
        self.assertFalse(n.book.rejected);self.assertEqual(n._v18_historical_rejected[0]['id'],'bad-goal')
    def test_active_goal_blocks_restore(self):
        b,n,m,inp=self.make();n.nav_uuid='still-running'
        with self.assertRaises(ValueError):restore_mission(n,m,inp,b)
    def test_scope_mismatch_blocks_restore(self):
        b,n,m,inp=self.make();m['scope']=['ROOM']
        with self.assertRaises(ValueError):restore_mission(n,m,inp,b)
    def test_bad_counter_blocks_restore(self):
        b,n,m,inp=self.make();m['counters']['sent_goals']=-1
        with self.assertRaises(ValueError):restore_mission(n,m,inp,b)

class LegacyAdditionalRegressionTests(unittest.TestCase):
    def test_counter_error_does_not_mutate_live_session(self):
        b,n,m,inp=LegacyResumeInterfaceTests().make();m['counters']['sent_goals']=-1
        with self.assertRaises(ValueError):restore_mission(n,m,inp,b)
        self.assertIsNone(n.book.session)
    def test_missing_old_history_has_all_zone_defaults(self):
        b,n,m,inp=LegacyResumeInterfaceTests().make();restore_mission(n,m,inp,b)
        self.assertEqual(set(n.book.deferral_history),set(n.zones_by_id))
        self.assertEqual(n.book.nav_abort_counts['LOBBY'],0)
    def test_pending_plan_send_blocks_restore(self):
        b,n,m,inp=LegacyResumeInterfaceTests().make();n.plan_send=object()
        with self.assertRaises(ValueError):restore_mission(n,m,inp,b)
    def test_scope_inferred_before_factory(self):
        b,n,m,inp=LegacyResumeInterfaceTests().make();cov=cp.coverage_payload(b,inp['observed'],inp['status'])
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{}):
            path,_=cp.save_checkpoint(d,dict(schema=cp.SCHEMA,coverage=cov,mission=m,context={}))
            a=types.SimpleNamespace(resume_from=path,resume_confirm_scene=True,zone_ids=None,threshold=.96)
            prepare_mission_args(a,b);self.assertEqual(set(a.zone_ids),set(m['scope']))
    def test_resume_needs_scene_confirmation(self):
        b,n,m,inp=LegacyResumeInterfaceTests().make()
        a=types.SimpleNamespace(resume_from=Path('irrelevant'),resume_confirm_scene=False)
        with self.assertRaises(ValueError):prepare_mission_args(a,b)

if __name__=='__main__':
    print('REGION_HELPERS:', 'actual project module' if REAL_REGIONS else 'isolated synthetic test helpers')
    unittest.main(verbosity=2)
