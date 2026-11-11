"""DN012 1.8.2: test Zone terminal semantics without ROS."""
from __future__ import annotations
import copy
import types
import unittest
from resume_runtime import install_simple_completion_policy, EXPLORATION_COMPLETE


def fake():
    ids=['R01','R10B','CORRIDOR_E']
    status={'R01':'OBSERVATION_SUFFICIENT','R10B':'PARTIAL_DEFERRED','CORRIDOR_E':'PARTIAL'}
    measured={'R01':.9947,'R10B':.0192,'CORRIDOR_E':.70}
    records={z:types.SimpleNamespace(state=status[z], measured=measured[z],
              deferred_reason='PATH_BLOCKED_NO_CONFIRMED_DETOUR' if z=='R10B' else None)
              for z in ids}
    book=types.SimpleNamespace(records=records,
        outcomes={'R10B':dict(reason='PATH_BLOCKED_NO_CONFIRMED_DETOUR')},
        audit_requested={},
        deferred_this_mission={'R10B'})
    def baseline_update(msg):
        for rec in records.values():
            if rec.state=='EXPLORATION_COMPLETE':
                rec.state='PARTIAL' # parent's ThresholdTracker recalculates state
        return []
    book.update=baseline_update
    book.eligible=lambda:[{'id':z} for z,r in records.items() if r.state not in ('OBSERVATION_SUFFICIENT','THRESHOLD_PENDING','PARTIAL_DEFERRED')]
    book.all_resolved=lambda:all(r.state=='OBSERVATION_SUFFICIENT' for r in records.values())
    book.is_complete=lambda z:records[z].state=='OBSERVATION_SUFFICIENT'
    runtime=types.SimpleNamespace(local_reviewed={'R10B'},exploration_stage='PRIMARY',local_second_zone=None)
    def end_local(inp):
        runtime.local_reviewed.add('CORRIDOR_E')
        return True
    runtime._end_local=end_local
    events=[]
    n=types.SimpleNamespace(book=book,connectivity=runtime,event=lambda *a,**kw:events.append((a,kw)))
    install_simple_completion_policy(n)
    return n,events


class SimpleCompletionTests(unittest.TestCase):
    def test_historical_deferred_is_terminal_complete(self):
        n,_=fake()
        self.assertEqual(n.book.records['R10B'].state,EXPLORATION_COMPLETE)
        self.assertIsNone(n.book.records['R10B'].deferred_reason)
        self.assertTrue(n.book.is_complete('R10B'))
        self.assertEqual(n.book.outcomes['R10B']['display_label'],'탐사 완료')
        self.assertFalse(n.book.outcomes['R10B']['coverage_target_met'])
        self.assertEqual(n.book.outcomes['R10B']['historical_diagnosis']['reason'],'PATH_BLOCKED_NO_CONFIRMED_DETOUR')
    def test_no_second_entry_into_reviewed_zone(self):
        n,_=fake()
        self.assertEqual([z['id'] for z in n.book.eligible()],['CORRIDOR_E'])
        n.book.update({})
        self.assertEqual([z['id'] for z in n.book.eligible()],['CORRIDOR_E'])
    def test_end_local_marks_coverage_below_96_complete(self):
        n,events=fake()
        n.connectivity._end_local({})
        self.assertEqual(n.book.records['CORRIDOR_E'].state,EXPLORATION_COMPLETE)
        self.assertTrue(n.book.all_resolved())
        self.assertEqual(events[-1][0][0],'ZONE_EXPLORATION_COMPLETE')
    def test_do_not_finish_zone_before_local_pass_ends(self):
        n,_=fake()
        r=n.connectivity
        r.local_second_zone='CORRIDOR_E'
        r.exploration_stage='SECOND'
        r.local_reviewed.add('CORRIDOR_E')
        n._dn012_finalize_reviewed()
        self.assertEqual(n.book.records['CORRIDOR_E'].state,'PARTIAL')
        r.local_second_zone=None;r.exploration_stage='PRIMARY'
        n._dn012_finalize_reviewed()
        self.assertEqual(n.book.records['CORRIDOR_E'].state,EXPLORATION_COMPLETE)
    def test_96_percent_already_sufficient_stays_sufficient(self):
        n,_=fake()
        self.assertEqual(n.book.records['R01'].state,'OBSERVATION_SUFFICIENT')
    def test_cleanup_idempotent(self):
        n,_=fake()
        original=n.book.outcomes['R10B']['historical_diagnosis']
        n._dn012_finalize_reviewed();n._dn012_finalize_reviewed()
        self.assertEqual(n.book.outcomes['R10B']['historical_diagnosis'],original)
        self.assertNotIn('historical_diagnosis',original)
    def test_no_finish_when_no_local_review(self):
        n,_=fake()
        self.assertEqual(n.book.records['CORRIDOR_E'].state,'PARTIAL')
        self.assertFalse(n.book.all_resolved())
    def test_no_unreachable_cert_required(self):
        n,_=fake()
        self.assertNotIn('physical_passage_cut_proven',n.book.outcomes['R10B'])
    def test_real_reported_15_sufficient_11_local_finished_1_remaining(self):
        n,_=fake()
        all_zones=('R01','R02A','R02B','R03','R04','R05','R06A','R06B','R07A','R07B',
                   'R08A','R08B','R09','R10A','R10B','R11','R12','R13','R14_T','R15_T',
                   'LOBBY','CORRIDOR_B','CONNECTOR_1','CONNECTOR_2','CORRIDOR_C','CORRIDOR_D','CORRIDOR_E')
        incomplete={'R02B','R04','R06A','R07B','R08A','R09','R10A','R10B','R11','R12','R14_T'}
        reviewed=set(all_zones[:20])
        for zid in all_zones:
            if zid in n.book.records:continue
            state=('PARTIAL_DEFERRED' if zid in incomplete else 'OBSERVATION_SUFFICIENT')
            n.book.records[zid]=types.SimpleNamespace(state=state,
                measured=.50 if zid in incomplete else .98,
                deferred_reason='PATH_BLOCKED' if zid in incomplete else None)
        n.connectivity.local_reviewed.update(reviewed)
        n._dn012_finalize_reviewed()
        from collections import Counter
        observed=Counter(rec.state for rec in n.book.records.values())
        self.assertEqual(observed,{'OBSERVATION_SUFFICIENT':15,
                                  'EXPLORATION_COMPLETE':11,'PARTIAL':1})
        self.assertEqual([z['id'] for z in n.book.eligible()],['CORRIDOR_E'])
    def test_unknown_reviewed_rejected(self):
        n,_=fake();n.connectivity.local_reviewed.add('UNLISTED')
        with self.assertRaisesRegex(ValueError,'UNKNOWN_ZONE'):
            n._dn012_finalize_reviewed()

if __name__=='__main__':unittest.main(verbosity=2)
