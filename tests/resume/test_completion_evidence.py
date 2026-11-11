"""DN-012 v1.8.4: synthetic/contract tests; no ROS2/Gazebo movement."""
from __future__ import annotations
import copy
import types
import unittest
from collections import Counter
from zone_lifecycle import is_terminal_result, review_verdict, live_review_verdict, unresolved_zones
from resume_selection import completion_facts, apply_terminal_history
from resume_runtime import install_simple_completion_policy, EXPLORATION_COMPLETE

ZONES = ['R01','R02A','R02B','R03','R04','R05','R06A','R06B','R07A','R07B','R08A','R08B','R09',
         'R10A','R10B','R11','R12','R13','R14_T','R15_T','LOBBY','CORRIDOR_B','CONNECTOR_1',
         'CONNECTOR_2','CORRIDOR_C','CORRIDOR_D','CORRIDOR_E']
BAD = {'CORRIDOR_E':.16650239369191777,'R09':.0013054830287206266,'R14_T':.0033904051534158334}


def fixture():
    # 16 coverage-complete; 8 valid local-terminated; 3 budget-incomplete.
    full=set(ZONES[:16]);full-=set(BAD)
    while len(full)<16:full.add(next(z for z in ZONES if z not in full and z not in BAD))
    other=set(ZONES)-full-set(BAD)
    records={}
    reviews={}
    tasks=[]
    for z in ZONES:
        measured=(.98 if z in full else BAD.get(z,.85))
        records[z]=dict(measured=measured, state='EXPLORATION_COMPLETE',views=0,
                        deferred_reason=None,low_gain_count=0)
        status='UNVERIFIED' if z in BAD else 'OBSERVED'
        reason=('SECOND_PASS_PREFLIGHT_BUDGET_UNTESTED_CANDIDATES'
                if z in BAD else 'FRESH_COVERAGE_RESOLVED')
        tid=z+':task'
        tasks.append(dict(task_id=tid,zone_id=z,status=status,reason=reason,checks=24 if z in BAD else 1,
                          attempts=[dict(candidate_id='old_candidate',reason='NAVFN_ERROR',stage='PREFLIGHT')]))
        reviews[z]=dict(results=[dict(task_id=tid,status=status,reason=reason)],ended_ros_s=120.0)
    mission=dict(scope=list(ZONES),threshold=.96,records=records,
                 connectivity=dict(local_reviewed=list(ZONES),stage='PRIMARY',local_second_zone=None,
                                   local_reviews=reviews,second_zone_preflights={z:24 for z in BAD},
                                   second_zone_nav={},second_zone_work={}),
                 task_ledger=dict(tasks=tasks),
                 completion_v183=dict(finished_zones=list(ZONES),lineage_reconciled=True),
                 counters=dict(sent_goals=52,plan_attempts=127,succeeded_goals=45,
                               aborted_goals=0,coverage_cancels=0),
                 outcomes={z:dict(classification='EXPLORATION_COMPLETE',normal_termination=True,
                                  reason='ZONE_LOCAL_PROCEDURE_FINISHED') for z in ZONES})
    return mission


def fake_node(m):
    records={z:types.SimpleNamespace(**dict(copy.deepcopy(rec), state=('PARTIAL' if z in BAD else rec['state'])))
             for z,rec in m['records'].items()}
    book=types.SimpleNamespace(records=records,
        deferred_this_mission=set(BAD),audit_requested=dict.fromkeys(BAD,'RECHECK'),
        outcomes=copy.deepcopy(m['outcomes']))
    book.threshold=.96
    book.eligible=lambda:[dict(id=z) for z,r in records.items()
                          if r.state not in ('OBSERVATION_SUFFICIENT','EXPLORATION_COMPLETE','PARTIAL_DEFERRED')
                          and z not in book.deferred_this_mission]
    book.is_complete=lambda z:records[z].state=='OBSERVATION_SUFFICIENT'
    book.all_resolved=lambda:all(r.state=='OBSERVATION_SUFFICIENT' for r in records.values())
    book.update=lambda status:[]
    ledger=types.SimpleNamespace(tasks={t['task_id']:copy.deepcopy(t) for t in m['task_ledger']['tasks']})
    conn=m['connectivity']
    rt=types.SimpleNamespace(local_reviewed=set(conn['local_reviewed']),local_reviews=copy.deepcopy(conn['local_reviews']),
        exploration_stage='PRIMARY',local_second_zone=None,
        second_zone_preflights=dict(conn['second_zone_preflights']),
        second_zone_nav={},second_zone_work={},tasks=ledger,
        primary_zone=None,second_task=None,second_catalog=None,second_context=None)
    def end_local(inp):
        zid=rt.local_second_zone
        rt.local_reviewed.add(zid)
        rt.local_reviews[zid]=dict(results=[dict(task_id=zid+':task',status=ledger.tasks[zid+':task']['status'],
                                                    reason=ledger.tasks[zid+':task']['reason'])])
        rt.local_second_zone=None;rt.exploration_stage='PRIMARY'
    rt._end_local=end_local
    rt._maybe_begin_local=lambda inp:False
    rt.selection_hook=lambda inp:False
    logs=[]
    node=types.SimpleNamespace(book=book,connectivity=rt,zones_by_id={z:dict(id=z) for z in ZONES},
        _v183_finished_zones=set(),_v184_review_reopened=set(),event=lambda name,**kw:logs.append((name,kw)),
        nav_uuid=None,nav_handle=None,nav_send=None,plan_send=None,plan_handle=None,plan_result=None,
        current=None,worker=None)
    node.logs=logs
    return node


class PolicyTests(unittest.TestCase):
    def test_unverified_budget_is_not_complete(self):
        for z in BAD:
            m=fixture();ok,reason=review_verdict(z,m['connectivity'],m['task_ledger'])
            self.assertFalse(ok);self.assertIn('UNRESOLVED',reason)
    def test_27_real_zone_shape_reclassifies_bad_ones(self):
        m=fixture();facts=completion_facts(m)
        self.assertEqual(len(facts),24)
        self.assertEqual(set(ZONES)-set(facts),set(BAD))
        self.assertEqual(Counter(facts.values()),{'COVERAGE_96_REACHED':16,
                                                  'LOCAL_PROCEDURE_EVIDENCE_COMPLETE':8})
        self.assertEqual(unresolved_zones(m),sorted(BAD))
    def test_explicit_v183_terminal_not_trusted(self):
        m=fixture();self.assertEqual(len(m['completion_v183']['finished_zones']),27)
        self.assertEqual(set(completion_facts(m)) & set(BAD),set())
    def test_bad_inaccessible_outcome_not_terminal_without_review(self):
        m=fixture();m['outcomes']['CORRIDOR_E']=dict(classification='INACCESSIBLE',
            normal_termination=True,evidence=dict(current_map_unreachable=True))
        self.assertNotIn('CORRIDOR_E',completion_facts(m))
    def test_real_done_96_without_local_review(self):
        m=fixture();m['connectivity']['local_reviewed']=[]
        self.assertEqual(len(completion_facts(m)),16)
    def test_real_done_local_below_96(self):
        m=fixture();self.assertIn('R13',completion_facts(m))
    def test_unknown_review_not_done(self):
        m=fixture();del m['connectivity']['local_reviews']['R13']
        self.assertNotIn('R13',completion_facts(m))
    def test_no_tasks_and_reliable_empty_review_completed(self):
        m=fixture();m['connectivity']['local_reviews']['R13']['results']=[]
        m['task_ledger']['tasks']=[t for t in m['task_ledger']['tasks'] if t['zone_id']!='R13']
        self.assertIn('R13',completion_facts(m))
    def test_budget_and_incomplete_never_terminal(self):
        for reason in ('SECOND_PASS_PREFLIGHT_BUDGET_UNTESTED_CANDIDATES','ZONE_WORK_BUDGET',
                       'CONTEXT_TIMEOUT','UNVERIFIED','INCOMPLETE_CATALOG'):
            self.assertFalse(is_terminal_result(dict(status='ROBOT_EXPLORATION_UNAVAILABLE',reason=reason)))
        self.assertTrue(is_terminal_result(dict(status='ROBOT_EXPLORATION_UNAVAILABLE',
                        reason='ALL_BOUNDED_ROBOT_APPROACHES_FAILED')))
    def test_split_requires_all_children_terminal(self):
        m=fixture();z='R13';p=m['task_ledger']['tasks'];parent=next(x for x in p if x['zone_id']==z)
        parent.update(status='SPLIT',reason='OBSERVATION_SPLIT_RESIDUAL',children=[z+':child'])
        m['connectivity']['local_reviews'][z]['results']=[dict(task_id=parent['task_id'],status='SPLIT',reason=parent['reason'])]
        self.assertNotIn(z,completion_facts(m))
        p.append(dict(task_id=z+':child',zone_id=z,status='OBSERVED',reason='FRESH_COVERAGE_RESOLVED'))
        self.assertIn(z,completion_facts(m))


class RuntimeTests(unittest.TestCase):
    def test_reopen_preserves_history_and_resets_only_preflights(self):
        m=fixture();n=fake_node(m)
        n.sent_goals=52;n.plan_attempts=127;n.succeeded_goals=45
        apply_terminal_history(n,m)
        self.assertEqual(n._v184_review_reopened,set(BAD))
        self.assertEqual(n._v183_finished_zones,set(ZONES)-set(BAD))
        self.assertEqual((n.sent_goals,n.plan_attempts,n.succeeded_goals),(52,127,45))
        for zid in BAD:
            self.assertEqual(n.book.records[zid].state,'PARTIAL')
            self.assertIsNone(n.book.records[zid].deferred_reason)
            self.assertNotIn(zid,n.book.deferred_this_mission)
            self.assertNotIn(zid,n.connectivity.local_reviewed)
            task=n.connectivity.tasks.tasks[zid+':task']
            self.assertEqual(task['checks'],0)
            self.assertEqual(task['status'],'PENDING')
            self.assertEqual(task['attempts'],[])
            self.assertEqual(task['prior_preflight_episodes'][0]['previous_checks'],24)
            self.assertEqual(len(task['prior_preflight_episodes'][0]['attempts']),1)
            self.assertEqual(n.connectivity.second_zone_preflights[zid],0)
        self.assertEqual(len(n.logs),1)
        self.assertEqual(n.logs[0][0],'ZONE_FALSE_COMPLETION_REOPENED_V184')
    def test_normal_completion_policy_does_not_finalize_budget(self):
        m=fixture();n=fake_node(m)
        n.connectivity.local_reviewed=set(ZONES)
        install_simple_completion_policy(n)
        self.assertNotIn('CORRIDOR_E',n._v183_finished_zones)
        self.assertNotEqual(n.book.records['CORRIDOR_E'].state,EXPLORATION_COMPLETE)
        self.assertIn('R13',n._v183_finished_zones)
        # Make one past budgeted task genuinely terminal during a later review.
        t=n.connectivity.tasks.tasks['CORRIDOR_E:task']
        t['status']='ROBOT_EXPLORATION_UNAVAILABLE';t['reason']='ALL_BOUNDED_ROBOT_APPROACHES_FAILED'
        n.connectivity.local_reviews['CORRIDOR_E']['results']=[dict(task_id=t['task_id'],status=t['status'],reason=t['reason'])]
        n._dn012_finalize_reviewed()
        self.assertEqual(n.book.records['CORRIDOR_E'].state,EXPLORATION_COMPLETE)
    def test_budgeted_end_local_emits_incomplete_not_complete(self):
        m=fixture();n=fake_node(m)
        # Avoid old false-labels from fixture influencing the next simulated lifecycle.
        n.book.records['CORRIDOR_E'].state='PARTIAL'
        n.connectivity.local_reviewed.discard('CORRIDOR_E')
        install_simple_completion_policy(n)
        n.connectivity.exploration_stage='SECOND';n.connectivity.local_second_zone='CORRIDOR_E'
        n.connectivity._end_local({})
        self.assertIn('ZONE_REVIEW_INCOMPLETE_V184',[name for name,_ in n.logs])
        self.assertNotIn(('ZONE_EXPLORATION_COMPLETE','CORRIDOR_E'),[(name,data.get('zone')) for name,data in n.logs])
    def test_definite_end_local_emits_complete(self):
        m=fixture();n=fake_node(m)
        n.connectivity.local_reviewed.discard('CORRIDOR_E')
        t=n.connectivity.tasks.tasks['CORRIDOR_E:task']
        t.update(status='OBSERVED',reason='FRESH_COVERAGE_RESOLVED')
        install_simple_completion_policy(n)
        n.connectivity.exploration_stage='SECOND';n.connectivity.local_second_zone='CORRIDOR_E'
        n.connectivity._end_local({})
        self.assertIn(('ZONE_EXPLORATION_COMPLETE','CORRIDOR_E'),[(name,data.get('zone')) for name,data in n.logs])

if __name__=='__main__':unittest.main(verbosity=2)
