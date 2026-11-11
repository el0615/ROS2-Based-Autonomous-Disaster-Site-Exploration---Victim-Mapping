"""No ROS, Gazebo, action goals or navigation; deterministic Zone lifecycle tests."""
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import os
import unittest

from zone_lifecycle import (SCHEMA, ZoneLifecycle, legacy_upgrade,
    completion_facts_v187, is_exact_legacy_false_corridor, review_terminal,
    validate_table)
from resume_selection import completion_facts
from zone_lifecycle import unresolved_zones
from phase_resume import second_phase_candidates


class FakeNode:
    def __init__(self, *, status='IN_PROGRESS', measured=.2, entered=False,
                 views=0, results=None, reviewed=True, stage='PRIMARY'):
        rec = SimpleNamespace(measured=measured,views=views,state='PARTIAL')
        self.book = SimpleNamespace(records={'E':rec}, threshold=.96,
            visit_state={'E':SimpleNamespace(entered=entered)})
        self.connectivity = SimpleNamespace(
            local_reviewed={'E'} if reviewed else set(),
            local_reviews={'E':{'results':results}} if results is not None else {},
            exploration_stage=stage,
            local_second_zone='E' if stage=='SECOND' else None)
        self._v183_finished_zones=set()
        self._v187_lifecycle=ZoneLifecycle(['E'])
        self._v187_lifecycle.zones['E']['exploration_status']=status


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('V187_REAL_CP')
        if path and Path(path).is_file():
            cls.wrapped=json.loads(gzip.decompress(Path(path).read_bytes()))
            cls.original=cls.wrapped['payload']['mission']
            cls.source=cls.wrapped['payload']['coverage']['source_session']
            cls.digest=cls.wrapped['payload_sha256']
        else:
            cls.wrapped=None

    def real(self):
        if self.wrapped is None:self.skipTest('actual uploaded checkpoint not provided to test environment')
        return copy.deepcopy(self.original)

    def test_real_legacy_corridor_correction(self):
        m=self.real()
        self.assertTrue(is_exact_legacy_false_corridor(m,self.source))
        f=completion_facts(m)
        self.assertEqual(len(f),26)
        self.assertNotIn('CORRIDOR_E',f)
        self.assertIn('R04',f)
        self.assertEqual(set(m['scope'])-set(f),{'CORRIDOR_E'})

    def test_correct_legacy_goal_history_unchanged(self):
        m=self.real();old=copy.deepcopy(m)
        u=legacy_upgrade(m,source_coverage=self.source,source_digest=self.digest)
        self.assertEqual(m,old)
        self.assertEqual(u['counters'],m['counters'])
        self.assertEqual(u['visits'],m['visits'])
        self.assertEqual(u['task_ledger'],m['task_ledger'])
        self.assertEqual(u['counters']['sent_goals'],52)
        self.assertEqual(u['counters']['succeeded_goals'],45)
        self.assertEqual(u['zone_lifecycle_v187']['migration']['source_payload_sha256'],self.digest)

    def test_corridor_unvisited_but_coverage_retained(self):
        m=self.real();u=legacy_upgrade(m,source_coverage=self.source)
        e=u['zone_lifecycle_v187']['zones']['CORRIDOR_E']
        self.assertEqual(e['exploration_status'],'NOT_STARTED')
        self.assertEqual(e['access_status'],'UNKNOWN')
        self.assertIsNone(e['completion_reason'])
        self.assertFalse(e['entry_confirmed'])
        self.assertAlmostEqual(e['coverage_fraction'],m['records']['CORRIDOR_E']['measured'])

    def test_former_completed_zones_stay_complete(self):
        m=self.real();u=legacy_upgrade(m,source_coverage=self.source)
        status=u['zone_lifecycle_v187']['zones']
        self.assertEqual(sum(z['exploration_status']=='COMPLETE' for z in status.values()),26)
        for zid in ('R04','R09','R14_T','R01','CORRIDOR_D'):
            self.assertEqual(status[zid]['exploration_status'],'COMPLETE')
            self.assertTrue(status[zid]['completion_reason'])

    def test_real_previous_false_complete_is_reopenable(self):
        m=self.real();u=legacy_upgrade(m,source_coverage=self.source)
        self.assertEqual(unresolved_zones(u),['CORRIDOR_E'])
        self.assertEqual(second_phase_candidates(u,['CORRIDOR_E']),{})

    def test_original_checkpoint_does_not_gain_field(self):
        m=self.real()
        legacy_upgrade(m,source_coverage=self.source)
        self.assertNotIn('zone_lifecycle_v187',m)

    def test_new_table_is_authoritative_even_if_legacy_completed(self):
        m=self.real();u=legacy_upgrade(m,source_coverage=self.source)
        self.assertIn('CORRIDOR_E',(u['completion_v183']['finished_zones']))
        self.assertNotIn('CORRIDOR_E',completion_facts(u))
        self.assertEqual(len(completion_facts(u)),26)

    def test_legacy_false_complete_not_applied_to_different_mission(self):
        m=self.real();m['source_session']='session_20270101_000000_999999'
        self.assertFalse(is_exact_legacy_false_corridor(m,self.source))
        self.assertIn('CORRIDOR_E',completion_facts(m))

    def test_legacy_false_complete_not_applied_if_entered(self):
        m=self.real();m['visit_states']['CORRIDOR_E']['entered']=True
        self.assertFalse(is_exact_legacy_false_corridor(m,self.source))

    def test_legacy_false_complete_not_applied_if_candidates_exhausted(self):
        m=self.real()
        m['connectivity']['local_reviews']['CORRIDOR_E']['results'][0]['status']='ROBOT_EXPLORATION_UNAVAILABLE'
        self.assertFalse(is_exact_legacy_false_corridor(m,self.source))

    def test_legacy_false_complete_not_applied_if_goal_count_differs(self):
        m=self.real();m['counters']['sent_goals']=53
        self.assertFalse(is_exact_legacy_false_corridor(m,self.source))

    def test_fresh_table_roundtrip(self):
        m=self.real();u=legacy_upgrade(m,source_coverage=self.source)
        ledger=ZoneLifecycle(m['scope']);ledger.load(u)
        table=ledger.snapshot()
        self.assertEqual(table['schema'],SCHEMA)
        self.assertEqual(table['zones']['CORRIDOR_E']['exploration_status'],'NOT_STARTED')
        self.assertEqual(len(completion_facts_v187(dict(u,zone_lifecycle_v187=table))),26)

    def test_malformed_table_rejected(self):
        m=self.real();u=legacy_upgrade(m,source_coverage=self.source)
        del u['zone_lifecycle_v187']['zones']['R04']
        with self.assertRaisesRegex(ValueError,'ZONE_SCOPE_INVALID'):
            completion_facts(u)

    def test_incomplete_with_completion_reason_rejected(self):
        m=self.real();u=legacy_upgrade(m,source_coverage=self.source)
        u['zone_lifecycle_v187']['zones']['CORRIDOR_E']['completion_reason']='FAKE'
        with self.assertRaisesRegex(ValueError,'NONCOMPLETE_WITH_REASON'):
            completion_facts(u)

    def test_cant_finish_unentered_unchecked_budget(self):
        n=FakeNode(results=[dict(status='UNVERIFIED',reason='SECOND_PASS_PREFLIGHT_BUDGET_UNTESTED_CANDIDATES')])
        self.assertFalse(review_terminal(n,'E'))
        n._v183_finished_zones.add('E')
        n._v187_lifecycle.sync(n)
        self.assertNotEqual(n._v187_lifecycle.zones['E']['exploration_status'],'COMPLETE')

    def test_dont_complete_during_active_second(self):
        n=FakeNode(stage='SECOND',entered=True,views=2,results=[])
        self.assertFalse(review_terminal(n,'E'))

    def test_finish_entered_after_local_second_even_unverified(self):
        n=FakeNode(entered=True,views=3,results=[dict(status='UNVERIFIED',reason='BUDGET')])
        self.assertTrue(review_terminal(n,'E'))
        n._v183_finished_zones.add('E');n._v187_lifecycle.sync(n)
        self.assertEqual(n._v187_lifecycle.zones['E']['exploration_status'],'COMPLETE')
        self.assertEqual(n._v187_lifecycle.zones['E']['access_status'],'ACCESSIBLE')

    def test_finish_blocked_after_exhaustive_bounded_access_review(self):
        n=FakeNode(results=[dict(status='ROBOT_EXPLORATION_UNAVAILABLE',reason='ALL_BOUNDED_ROBOT_APPROACHES_FAILED')])
        self.assertTrue(review_terminal(n,'E'))
        n.connectivity.local_reviewed.add('E');n._v187_lifecycle.sync(n)
        self.assertEqual(n._v187_lifecycle.zones['E']['access_status'],'INACCESSIBLE')
        self.assertEqual(n._v187_lifecycle.zones['E']['completion_reason'],'BOUNDED_ACCESS_REVIEW_FINISHED')

    def test_96_percent_completes_without_entry(self):
        n=FakeNode(measured=.961,reviewed=False)
        self.assertTrue(review_terminal(n,'E'))
        n._v187_lifecycle.sync(n)
        self.assertEqual(n._v187_lifecycle.zones['E']['completion_reason'],'COVERAGE_96_REACHED')

    def test_unfinished_explicit_zone_reopens_across_multiple_resume_sessions(self):
        from resume_selection import apply_terminal_history
        m=self.real();u=legacy_upgrade(m,source_coverage=self.source)
        e='CORRIDOR_E'
        # This simulates a *new* v187 checkpoint after an unsuccessful local
        # review. It was never falsely COMPLETE in this new checkpoint.
        u['records'][e]['state']='PARTIAL_DEFERRED'
        u['records'][e]['deferred_reason']='LOCAL_APPROACH_REVIEW_FINISHED'
        u['completion_v183']['finished_zones']=[z for z in u['completion_v183']['finished_zones'] if z!=e]
        u['connectivity']['local_reviewed']=[z for z in u['connectivity']['local_reviewed'] if z!=e]
        self.assertEqual(unresolved_zones(u),[])
        class Rec:pass
        recs={z:Rec() for z in u['scope']}
        for z,r in recs.items():
            r.state='EXPLORATION_COMPLETE' if z!='CORRIDOR_E' else 'PARTIAL_DEFERRED'
            r.measured=u['records'][z]['measured'];r.deferred_reason=None
        book=SimpleNamespace(records=recs,deferred_this_mission={e},audit_requested={},outcomes={})
        node=SimpleNamespace(book=book,connectivity=None,event=lambda *a,**k:None)
        apply_terminal_history(node,u)
        self.assertNotIn(e,book.deferred_this_mission)
        self.assertEqual(recs[e].state,'PARTIAL')
        self.assertEqual(node._v184_review_reopened,{e})
        self.assertEqual(len(node._v183_finished_zones),26)

    def test_actual_legacy_terminal_masked_unchecked_budget_is_reset_for_retries(self):
        from resume_selection import apply_terminal_history
        m=self.real();u=legacy_upgrade(m,source_coverage=self.source)
        e='CORRIDOR_E'
        tasks=[copy.deepcopy(t) for t in m['task_ledger']['tasks'] if t['zone_id']==e]
        self.assertEqual(len(tasks),2)
        self.assertTrue(all(t['reason']=='ZONE_LIFECYCLE_ALREADY_FINISHED' for t in tasks))
        ids={t['task_id']:t for t in tasks}
        ledger=SimpleNamespace(tasks=ids)
        conn=SimpleNamespace(local_reviewed=set(m['connectivity']['local_reviewed']),
            local_reviews=copy.deepcopy(m['connectivity']['local_reviews']),
            second_zone_preflights={e:24},tasks=ledger,local_second_zone=None)
        class Rec:pass
        records={z:Rec() for z in u['scope']}
        for z,r in records.items():
            r.state='EXPLORATION_COMPLETE' if z!=e else 'PARTIAL_DEFERRED'
            r.measured=u['records'][z]['measured'];r.deferred_reason=None
        book=SimpleNamespace(records=records,deferred_this_mission={e},audit_requested={},outcomes={})
        events=[]
        n=SimpleNamespace(book=book,connectivity=conn,event=lambda *a,**kw:events.append((a,kw)))
        apply_terminal_history(n,u)
        self.assertEqual(conn.second_zone_preflights[e],0)
        self.assertEqual(records[e].state,'PARTIAL')
        for t in ids.values():
            self.assertEqual(t['checks'],0)
            self.assertEqual(t['status'],'PENDING')
            self.assertEqual(t['prior_preflight_episodes'][-1]['previous_checks'],12)
            self.assertEqual(t['prior_preflight_episodes'][-1]['reason'],'ZONE_LIFECYCLE_ALREADY_FINISHED')
        self.assertNotIn(e,book.deferred_this_mission)
        self.assertEqual(len(n._v183_finished_zones),26)

    def test_state_not_changed_by_report_only_event(self):
        ledger=ZoneLifecycle(['E'])
        ledger.event('ZONE_EXPLORATION_COMPLETE', {'zone':'E'})
        self.assertNotEqual(ledger.zones['E']['exploration_status'],'COMPLETE')

    def test_completed_zone_never_reopened_by_unverified(self):
        n=FakeNode(status='COMPLETE',results=[dict(status='UNVERIFIED',reason='BUDGET')])
        n._v187_lifecycle.zones['E'].update(completion_reason='ZONE_LOCAL_PROCEDURE_FINISHED',phase='TERMINAL')
        self.assertTrue(review_terminal(n,'E'))
        n._v187_lifecycle.sync(n)
        self.assertEqual(n._v187_lifecycle.zones['E']['exploration_status'],'COMPLETE')

    def test_live_entry_confirmed_updates_access_and_phase(self):
        ledger=ZoneLifecycle(['E']);ledger.event('ENTRY_CONFIRMED',{'zone':'E'})
        self.assertEqual(ledger.zones['E']['access_status'],'ACCESSIBLE')
        self.assertEqual(ledger.zones['E']['exploration_status'],'IN_PROGRESS')
        self.assertEqual(ledger.zones['E']['phase'],'PRIMARY_OBSERVE')

    def test_new_mission_without_legacy_overrides(self):
        ledger=ZoneLifecycle(['R01','CORRIDOR_E'])
        self.assertTrue(all(x['exploration_status']=='NOT_STARTED' for x in ledger.zones.values()))
        self.assertEqual(len(ledger.snapshot()['zones']),2)

    def test_runtime_wrapper_imports_no_ros_required(self):
        from resume_runtime import capture_mission, install_simple_completion_policy
        self.assertTrue(callable(capture_mission))
        self.assertTrue(callable(install_simple_completion_policy))


if __name__=='__main__':unittest.main(verbosity=2)
