"""DN012 v1.8.6: closed Zone must never auto-reopen due to subregion diagnostics.

Synthetic only. No ROS, no navigation/velocity publishing, no project data writes.
"""
from __future__ import annotations
import unittest,copy,types
from resume_selection import completion_facts,apply_terminal_history,current_pose_decision
from zone_lifecycle import unresolved_zones
from resume_runtime import install_simple_completion_policy,EXPLORATION_COMPLETE
try:
    from phase_resume import second_phase_candidates
except ImportError:
    second_phase_candidates = None

SCOPE=['R01','R04','R09','CORRIDOR_E','R14_T','R10B']

def mission():
    vals={'R01':.995,'R04':.9371,'R09':.001305,'CORRIDOR_E':.16650,'R14_T':.00339,'R10B':.0589}
    records={z:dict(state=('OBSERVATION_SUFFICIENT' if z=='R01' else 'EXPLORATION_COMPLETE'),
                    measured=vals[z],views=(5 if z=='R04' else 0),low_gain_count=0,deferred_reason=None)
             for z in SCOPE}
    conn=dict(stage='PRIMARY',local_second_zone=None,local_reviewed=SCOPE[1:],
              local_reviews={z:dict(ended_ros_s=1200.,measured_percent=100*vals[z],
                                  results=[dict(task_id=z+':0',status='UNVERIFIED',
                                      reason='SECOND_PASS_PREFLIGHT_BUDGET_UNTESTED_CANDIDATES')])
                             for z in SCOPE[1:]})
    tasks=[dict(zone_id=z,task_id=z+':0',status='UNVERIFIED',
                reason='SECOND_PASS_PREFLIGHT_BUDGET_UNTESTED_CANDIDATES',
                checks=24,attempts=[dict(reason='PLAN_TIMEOUT')]) for z in SCOPE[1:]]
    return dict(scope=list(SCOPE),threshold=.96,records=records,connectivity=conn,
                task_ledger=dict(tasks=tasks),counters=dict(sent_goals=52,plan_attempts=127,
                    succeeded_goals=45,aborted_goals=0,coverage_cancels=0),
                completion_v183=dict(finished_zones=SCOPE[1:],lineage_reconciled=True),
                outcomes={z:dict(classification='INACCESSIBLE',reason='NO_GLOBAL_ROBOT_ROUTE') for z in SCOPE[1:]})

def node_fake(m):
    records={z:types.SimpleNamespace(**copy.deepcopy(r)) for z,r in m['records'].items()}
    book=types.SimpleNamespace(records=records,threshold=.96,
        outcomes=copy.deepcopy(m['outcomes']),audit_requested={},deferred_this_mission=set(),
        update=lambda _:[],is_complete=lambda z:records[z].state=='OBSERVATION_SUFFICIENT',
        eligible=lambda:[dict(id=z) for z,r in records.items() if r.state not in ('OBSERVATION_SUFFICIENT','EXPLORATION_COMPLETE')],
        all_resolved=lambda:all(r.state=='OBSERVATION_SUFFICIENT' for r in records.values()))
    ledger=types.SimpleNamespace(tasks={t['task_id']:copy.deepcopy(t) for t in m['task_ledger']['tasks']})
    rt=types.SimpleNamespace(local_reviewed=set(m['connectivity']['local_reviewed']),
        local_reviews=copy.deepcopy(m['connectivity']['local_reviews']),exploration_stage='PRIMARY',
        local_second_zone=None,tasks=ledger,primary_zone=None,second_task=None,
        second_zone_preflights={z:24 for z in SCOPE[1:]})
    def end_local(inp):
        rt.local_reviewed.add(rt.local_second_zone)
        rt.local_reviews[rt.local_second_zone]=dict(results=[dict(status='UNVERIFIED',reason='BUDGET')])
        rt.local_second_zone=None;rt.exploration_stage='PRIMARY'
    rt._end_local=end_local;rt._maybe_begin_local=lambda inp:False;rt.selection_hook=lambda inp:False
    logs=[]
    n=types.SimpleNamespace(book=book,connectivity=rt,zones_by_id={z:dict(id=z) for z in SCOPE},
                            event=lambda *a,**kw:logs.append((a,kw)),
                            _v183_finished_zones=set(),_v184_review_reopened=set(),
                            nav_uuid=None,nav_handle=None,nav_send=None,plan_send=None,plan_handle=None,
                            plan_result=None,worker=None)
    return n,logs

class DurableCompletionTests(unittest.TestCase):
    def test_saved_terminal_state_is_authoritative(self):
        m=mission();m['connectivity']['local_reviewed']=[]
        m['connectivity']['local_reviews']={}
        self.assertIn('CORRIDOR_E',completion_facts(m))
    def test_all_27_like_complete_even_subregions_unverified(self):
        m=mission();facts=completion_facts(m)
        self.assertEqual(set(facts),set(SCOPE))
        self.assertEqual(facts['CORRIDOR_E'],'SAVED_ZONE_EXPLORATION_COMPLETE')
    def test_local_review_without_explicit_saved_state_is_complete(self):
        m=mission();m['records']['R04']['state']='PARTIAL'
        m['completion_v183']['finished_zones']=[]
        self.assertEqual(completion_facts(m)['R04'],'LOCAL_ZONE_REVIEW_FINISHED')
    def test_local_review_terminal_even_budget(self):
        m=mission();m['records']['CORRIDOR_E']['state']='PARTIAL'
        m['completion_v183']['finished_zones']=[]
        self.assertIn('CORRIDOR_E',completion_facts(m))
    def test_pending_zone_not_closed(self):
        m=mission();m['records']['R10B']['state']='PARTIAL'
        m['completion_v183']['finished_zones']=[]
        m['connectivity']['local_reviewed'].remove('R10B')
        self.assertNotIn('R10B',completion_facts(m))
    def test_mid_second_not_closed_on_review_label_alone(self):
        m=mission();m['records']['R09']['state']='PARTIAL'
        m['completion_v183']['finished_zones']=[]
        m['connectivity']['stage']='SECOND';m['connectivity']['local_second_zone']='R09'
        self.assertNotIn('R09',completion_facts(m))
    def test_96_enough_without_review(self):
        m=mission();m['records']['R01']['state']='PARTIAL'
        self.assertEqual(completion_facts(m)['R01'],'COVERAGE_96_REACHED')
    def test_no_automatic_reopening(self):
        m=mission();self.assertEqual(unresolved_zones(m),[])
        n,logs=node_fake(m)
        apply_terminal_history(n,m)
        self.assertEqual(n._v184_review_reopened,set())
        self.assertEqual(n._v183_finished_zones,set(SCOPE))
        self.assertEqual(n.connectivity.second_zone_preflights['CORRIDOR_E'],24)
        self.assertEqual(n.connectivity.tasks.tasks['CORRIDOR_E:0']['checks'],24)
        self.assertEqual(len(n.connectivity.tasks.tasks['CORRIDOR_E:0']['attempts']),1)
    def test_no_secondary_resume_of_finished(self):
        if second_phase_candidates is None:
            self.skipTest('v1.8.4 baseline has no phase adapter')
        self.assertEqual(second_phase_candidates(mission(),[]),{})
    def test_live_local_review_finishes_even_unverified(self):
        m=mission();n,logs=node_fake(m)
        for z in SCOPE[1:]:n.book.records[z].state='PARTIAL'
        install_simple_completion_policy(n)
        for z in SCOPE[1:]:
            self.assertEqual(n.book.records[z].state,EXPLORATION_COMPLETE)
            self.assertNotIn(z,[q['id'] for q in n.book.eligible()])
    def test_live_secondary_in_progress_stays_open(self):
        m=mission();n,logs=node_fake(m)
        n.book.records['R04'].state='PARTIAL'
        n.connectivity.exploration_stage='SECOND';n.connectivity.local_second_zone='R04'
        install_simple_completion_policy(n)
        self.assertNotEqual(n.book.records['R04'].state,EXPLORATION_COMPLETE)
    def test_after_end_local_closes_budget_zone(self):
        m=mission();n,logs=node_fake(m)
        n.connectivity.local_reviewed.remove('R09')
        n.book.records['R09'].state='PARTIAL';n.connectivity.exploration_stage='SECOND';n.connectivity.local_second_zone='R09'
        install_simple_completion_policy(n)
        n.connectivity._end_local({})
        self.assertEqual(n.book.records['R09'].state,EXPLORATION_COMPLETE)
        self.assertTrue(any(e[0][0]=='ZONE_EXPLORATION_COMPLETE' for e in logs))
    def test_report_preserves_coverage_count_and_outcome_history(self):
        m=mission();n,_=node_fake(m)
        n.book.records['R14_T'].state='PARTIAL'
        install_simple_completion_policy(n)
        self.assertAlmostEqual(n.book.records['R14_T'].measured,.00339)
        outcome=n.book.outcomes['R14_T']
        self.assertEqual(outcome['classification'],'EXPLORATION_COMPLETE')
        self.assertEqual(outcome['historical_diagnosis']['classification'],'INACCESSIBLE')
    def test_reported_27_zone_case_does_not_reopen_11(self):
        names=['R01','R02A','R02B','R03','R04','R05','R06A','R06B','R07A','R07B','R08A',
               'R08B','R09','R10A','R10B','R11','R12','R13','R14_T','R15_T','LOBBY',
               'CORRIDOR_B','CONNECTOR_1','CONNECTOR_2','CORRIDOR_C','CORRIDOR_D','CORRIDOR_E']
        low={'R02B','R04','R06A','R07B','R09','R10A','R10B','R11','R12','R14_T','CORRIDOR_E'}
        m=mission();m['scope']=names;m['records']={}
        m['completion_v183']['finished_zones']=list(low)
        m['connectivity']['local_reviewed']=list(low)
        m['connectivity']['local_reviews']={}
        m['task_ledger']['tasks']=[]
        for zid in names:
            is_low=zid in low
            m['records'][zid]=dict(state='EXPLORATION_COMPLETE' if is_low else 'OBSERVATION_SUFFICIENT',
                measured=.1665 if zid=='CORRIDOR_E' else .03 if is_low else .98,
                views=0,low_gain_count=0,deferred_reason=None)
            if is_low:
                m['connectivity']['local_reviews'][zid]=dict(results=[dict(task_id=zid+':0',
                    status='UNVERIFIED',reason='SECOND_PASS_PREFLIGHT_BUDGET_UNTESTED_CANDIDATES')])
                m['task_ledger']['tasks'].append(dict(zone_id=zid,task_id=zid+':0',
                    status='UNVERIFIED',reason='SECOND_PASS_PREFLIGHT_BUDGET_UNTESTED_CANDIDATES',checks=24))
        facts=completion_facts(m)
        self.assertEqual(len(facts),27)
        self.assertEqual(len(unresolved_zones(m)),0)
        self.assertEqual(facts['CORRIDOR_E'],'SAVED_ZONE_EXPLORATION_COMPLETE')
        self.assertEqual(facts['R04'],'SAVED_ZONE_EXPLORATION_COMPLETE')

    def test_unknown_zone_strict(self):
        m=mission();m['connectivity']['local_reviewed'].append('UNKNOWN')
        with self.assertRaisesRegex(ValueError,'UNKNOWN_ZONE'):completion_facts(m)
    def test_empty_review_unfinished_without_saved_complete(self):
        m=mission();m['records']['R10B']['state']='PARTIAL';m['completion_v183']['finished_zones']=[]
        m['connectivity']['local_reviewed'].remove('R10B')
        m['connectivity']['local_reviews'].pop('R10B')
        self.assertNotIn('R10B',completion_facts(m))

if __name__=='__main__':unittest.main(verbosity=2)
