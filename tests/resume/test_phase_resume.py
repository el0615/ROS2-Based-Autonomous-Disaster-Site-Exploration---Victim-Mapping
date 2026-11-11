"""DN012 v1.8.5 regression: phase-resume without replaying an already run PRIMARY.

ROS-free; actual T6/TF/Nav2 safety must still be validated on the user's J100.
"""
from __future__ import annotations
import copy
import math
from types import SimpleNamespace as NS
import unittest
from phase_resume import second_phase_candidates, install_second_phase_continuation


def state():
    zones=['R01','R04','CORRIDOR_E']
    rows={z:dict(measured=.98 if z=='R01' else .9371 if z=='R04' else .1665,
                 state='EXPLORATION_COMPLETE',views=0 if z=='CORRIDOR_E' else 6,
                 deferred_reason=None) for z in zones}
    task=lambda z:dict(task_id=z+':c1',zone_id=z,status='UNVERIFIED',
                         reason='SECOND_PASS_PREFLIGHT_BUDGET_UNTESTED_CANDIDATES',
                         centroid_map_m=[42.0,7.0] if z=='R04' else [54.,-1.],checks=24)
    return dict(scope=zones,threshold=.96,records=rows,
        visit_states=dict(R01=dict(entered=True), R04=dict(entered=True),
                          CORRIDOR_E=dict(entered=False)),
        connectivity=dict(stage='PRIMARY',local_second_zone=None,
                          local_reviewed=list(zones),
                          local_reviews={z:dict(results=[dict(task_id=z+':c1',status='UNVERIFIED',
                              reason='SECOND_PASS_PREFLIGHT_BUDGET_UNTESTED_CANDIDATES')]) for z in zones}),
        task_ledger=dict(tasks=[task('R04'),task('CORRIDOR_E')]))


def node(m):
    events=[]; stops=[]; selected=[]
    records={z:NS(state='PARTIAL' if z!='R01' else 'OBSERVATION_SUFFICIENT',measured=r['measured'])
             for z,r in m['records'].items()}
    book=NS(records=records,active_zone=None,locked_zone=None)
    book.is_complete=lambda z:records[z].state=='OBSERVATION_SUFFICIENT'
    st=NS(ready=lambda *a: True)
    tasks={t['task_id']:copy.deepcopy(t) for t in m['task_ledger']['tasks']}
    for t in tasks.values():t['status']='PENDING';t['checks']=0
    ledger=NS(tasks=tasks)
    ledger.refresh=lambda inp:None
    rt=NS(exploration_stage='PRIMARY',local_second_zone=None,primary_zone=None,
          local_reviews=copy.deepcopy(m['connectivity']['local_reviews']),
          local_reviewed=set(),tasks=ledger,
          _drain_workers=lambda:True, clear_round=lambda:None,zid=None,
          second_task=None,second_catalog=None,second_context=None,
          second_future=None,second_future_task=None,second_attempted={},second_last_account=None)
    def hook(inp):
        selected.append((rt.exploration_stage,rt.local_second_zone))
        return True
    rt.selection_hook=hook
    def end(inp):
        z=rt.local_second_zone
        rt.local_reviews[z]=dict(results=[dict(task_id=z+':c1',status='OBSERVED',reason='FRESH_COVERAGE_RESOLVED')])
        rt.local_reviewed.add(z)
        rt.local_second_zone=None;rt.exploration_stage='PRIMARY'
    rt._end_local=end
    n=NS(connectivity=rt,book=book,zones_by_id={z:{} for z in records},
         _v183_finished_zones={'R01'},_v184_review_reopened={'R04','CORRIDOR_E'},
         nav_uuid=None,nav_send=None,nav_handle=None,plan_send=None,plan_handle=None,
         plan_result=None,current=None,done=False,stop_requested=None,
         stop_evidence=st,local_start_clear=lambda inp:True,ros_s=lambda:55.0,
         foreign_active=lambda:[], request_stop=lambda v:stops.append(v),
         event=lambda typ,**kwargs:events.append((typ,kwargs)),options=[],batch_zone=None,
         options_context=None)
    return n,selected,events,stops


class PhaseClassification(unittest.TestCase):
    def test_entered_r04_second_but_unentered_corridor_e_primary(self):
        p=second_phase_candidates(state(),['R04','CORRIDOR_E'])
        self.assertEqual(list(p),['R04'])
        self.assertEqual(p['R04']['phase'],'SECOND')
    def test_explicit_second_phase_even_if_entered_false(self):
        m=state();m['connectivity'].update(stage='SECOND',local_second_zone='CORRIDOR_E')
        p=second_phase_candidates(m,['CORRIDOR_E'])
        self.assertIn('CORRIDOR_E',p)
    def test_empty_task_does_not_fabricate_second(self):
        m=state();m['task_ledger']['tasks']=[t for t in m['task_ledger']['tasks'] if t['zone_id']!='R04']
        with self.assertRaisesRegex(ValueError,'TASK_HISTORY_MISSING'):
            second_phase_candidates(m,['R04'])
    def test_actual_task_terminal_never_resumed(self):
        m=state();m['task_ledger']['tasks'][0]['status']='OBSERVED'
        with self.assertRaisesRegex(ValueError,'TASK_ALREADY_TERMINAL'):
            second_phase_candidates(m,['R04'])
    def test_invalid_unknown_reopened_refused(self):
        with self.assertRaisesRegex(ValueError,'UNKNOWN_REOPENED'):
            second_phase_candidates(state(),['R99'])
    def test_coverage_96_exclusion_is_external(self):
        m=state();p=second_phase_candidates(m,['CORRIDOR_E'])
        self.assertFalse(p)
    def test_finished_review_with_no_visit_starts_primary(self):
        m=state();m['records']['R04']['views']=0;m['visit_states']['R04']['entered']=False
        self.assertNotIn('R04',second_phase_candidates(m,['R04']))


class HookRegression(unittest.TestCase):
    def test_second_phase_not_primary_and_original_hook_called(self):
        m=state();n,seen,events,stops=node(m)
        pending=install_second_phase_continuation(n,m)
        self.assertIn('R04',pending)
        out=n.connectivity.selection_hook(dict(pose=(40.,7.,0.)))
        self.assertTrue(out)
        self.assertEqual(seen[-1],('SECOND','R04'))
        self.assertEqual(n.connectivity._v185_current_second,'R04')
        self.assertNotIn('R04',n.connectivity.local_reviews)
        self.assertEqual(len(n.connectivity.local_review_history['R04']),1)
        self.assertFalse(stops)
        self.assertTrue(any(k=='RESUME_SECOND_PHASE_V185' for k,_ in events))
    def test_old_review_archived_before_new_result(self):
        m=state();n,seen,events,_=node(m)
        install_second_phase_continuation(n,m)
        n.connectivity.selection_hook(dict(pose=(40.,7.,0.)))
        n.connectivity._end_local(dict(pose=(40.,7.,0.)))
        self.assertNotIn('R04',n.connectivity._v185_pending_second)
        self.assertEqual(n.connectivity.local_reviews['R04']['results'][0]['status'],'OBSERVED')
        self.assertEqual(n.connectivity.local_review_history['R04'][0]['results'][0]['status'],'UNVERIFIED')
        self.assertTrue(any(k=='RESUME_SECOND_EPISODE_FINISHED_V185' for k,_ in events))
        n.connectivity.selection_hook(dict(pose=(40.,7.,0.)))
        self.assertEqual(seen[-1],('PRIMARY',None))
    def test_same_goal_counters_not_modified(self):
        m=state();n,_,_,_=node(m)
        n.sent_goals=52;n.succeeded_goals=45;n.plan_attempts=127
        install_second_phase_continuation(n,m)
        n.connectivity.selection_hook(dict(pose=(40.,7.,0.)))
        self.assertEqual((n.sent_goals,n.succeeded_goals,n.plan_attempts),(52,45,127))
    def test_active_nav_cannot_switch_phase(self):
        m=state();n,seen,events,_=node(m)
        install_second_phase_continuation(n,m)
        n.nav_uuid='active';n.connectivity.selection_hook(dict(pose=(40.,7.,0.)))
        self.assertEqual(seen[-1],('PRIMARY',None))
        self.assertFalse(any(k=='RESUME_SECOND_PHASE_V185' for k,_ in events))
    def test_active_primary_not_preempted(self):
        m=state();n,seen,events,_=node(m)
        install_second_phase_continuation(n,m)
        n.connectivity.primary_zone='CORRIDOR_E'
        n.connectivity.selection_hook(dict(pose=(40.,7.,0.)))
        self.assertEqual(seen[-1],('PRIMARY',None))
        self.assertFalse(any(k=='RESUME_SECOND_PHASE_V185' for k,_ in events))
    def test_unsafe_local_start_blocks_second(self):
        m=state();n,seen,events,stops=node(m)
        install_second_phase_continuation(n,m)
        n.local_start_clear=lambda inp:False
        n.connectivity.selection_hook(dict(pose=(40.,7.,0.)))
        self.assertEqual(stops,['RESUME_SECOND_LOCAL_START_NOT_CLEAR'])
        self.assertEqual(n.connectivity.exploration_stage,'PRIMARY')
        self.assertFalse(seen)
    def test_unsafe_foreign_goal_blocks_second(self):
        m=state();n,seen,events,stops=node(m)
        install_second_phase_continuation(n,m)
        n.foreign_active=lambda:['other-uuid']
        n.connectivity.selection_hook(dict(pose=(40.,7.,0.)))
        self.assertEqual(stops,['RESUME_SECOND_OTHER_GOAL_ACTIVE'])
        self.assertFalse(seen)
    def test_stop_evidence_wait_no_phase_and_no_nav(self):
        m=state();n,seen,events,stops=node(m)
        n.stop_evidence=NS(ready=lambda *args:False)
        install_second_phase_continuation(n,m)
        self.assertTrue(n.connectivity.selection_hook(dict(pose=(40.,7.,0.))))
        self.assertFalse(seen)
        self.assertEqual(n.connectivity.exploration_stage,'PRIMARY')
    def test_already_finished_zone_excluded_from_second(self):
        m=state();n,seen,events,stops=node(m)
        n._v183_finished_zones.add('R04')
        install_second_phase_continuation(n,m)
        n.connectivity.selection_hook(dict(pose=(40.,7.,0.)))
        self.assertEqual(seen[-1],('PRIMARY',None))
        self.assertNotIn('R04',n.connectivity._v185_pending_second)
    def test_double_install_refused(self):
        m=state();n,seen,events,stops=node(m)
        install_second_phase_continuation(n,m)
        with self.assertRaisesRegex(ValueError,'DUPLICATE_PHASE'):
            install_second_phase_continuation(n,m)
    def test_no_tasks_does_not_issue_second_phase(self):
        m=state();n,seen,events,stops=node(m)
        n.connectivity.tasks.tasks={}
        install_second_phase_continuation(n,m)
        n.connectivity.selection_hook(dict(pose=(40.,7.,0.)))
        self.assertFalse(seen)
        self.assertTrue(any(k=='RESUME_SECOND_NO_PENDING_TASK_V185' for k,_ in events))
        self.assertEqual(stops,['RESUME_SECOND_NO_RUNNABLE_TASK'])
        self.assertEqual(n.connectivity.exploration_stage,'PRIMARY')


if __name__=='__main__':unittest.main(verbosity=2)
