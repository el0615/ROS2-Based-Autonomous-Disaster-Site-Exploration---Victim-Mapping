import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/exploration/navigation/return_home'))
from return_home_policy import check_mission_complete, pose_distance


def success():
    manifest={'options':{'threshold':0.96,'execute':True,'zone_ids':None},'zone_bundle_signature':'same'}
    summary={'threshold':0.96,'execute':True,'state':'ALL_REQUESTED_ZONES_OBSERVED',
             'sent':35,'nav_succeeded':35,'stop_reason':'ALL_REQUESTED_ZONES_OBSERVED',
             'zone_bundle_signature':'same','zones':{f'Z{i:02d}':{
                 'state':'OBSERVATION_SUFFICIENT','measured_percent':97.0}
                 for i in range(27)}}
    return manifest,summary


class PolicyTests(unittest.TestCase):
    def test_success(self):
        a,b=success()
        self.assertTrue(check_mission_complete(a,b)[0])

    def test_96_target(self):
        a,b=success();b['zones']['Z00']['measured_percent']=96.0
        self.assertTrue(check_mission_complete(a,b)[0])

    def test_959_not_complete(self):
        a,b=success();b['zones']['Z00']['measured_percent']=95.9
        self.assertFalse(check_mission_complete(a,b)[0])

    def test_deferred_never_passes(self):
        a,b=success();b['zones']['Z00']['state']='PARTIAL_DEFERRED'
        self.assertFalse(check_mission_complete(a,b)[0])

    def test_partial_map_never_passes(self):
        a,b=success();b['zones'].pop('Z00')
        self.assertFalse(check_mission_complete(a,b)[0])

    def test_goal_limit_never_passes(self):
        a,b=success();b['state']='GOAL_LIMIT_REACHED'
        self.assertFalse(check_mission_complete(a,b)[0])

    def test_user_interrupt_never_passes(self):
        a,b=success();b['state']='USER_INTERRUPT'
        self.assertFalse(check_mission_complete(a,b)[0])

    def test_wrong_threshold_never_passes(self):
        a,b=success();a['options']['threshold']=0.90
        self.assertFalse(check_mission_complete(a,b)[0])

    def test_wrong_signature_never_passes(self):
        a,b=success();b['zone_bundle_signature']='different'
        self.assertFalse(check_mission_complete(a,b)[0])

    def test_dryrun_never_passes(self):
        a,b=success();a['options']['execute']=False
        self.assertFalse(check_mission_complete(a,b)[0])

    def test_no_completed_goals_never_passes(self):
        a,b=success();b['nav_succeeded']=34
        self.assertFalse(check_mission_complete(a,b)[0])

    def test_pose_distance(self):
        d,y=pose_distance({'x':0,'y':0,'yaw':3.13},{'x':3,'y':4,'yaw':-3.13})
        self.assertAlmostEqual(d,5.0)
        self.assertLess(y,0.03)

if __name__=='__main__':unittest.main()
