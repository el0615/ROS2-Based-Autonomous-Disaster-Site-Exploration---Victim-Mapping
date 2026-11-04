import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/exploration/navigation/return_home'))
from return_home_policy import parse_saved_session_path

class SavedSessionTests(unittest.TestCase):
    def test_three_part_session_id_is_not_truncated(self):
        path = '/home/robotics/Projects/ROS2-Based Autonomous Disaster-Site Exploration & Victim Mapping/logs/hospital_zone_mission/session_20261104_100650_529617'
        self.assertEqual(parse_saved_session_path('[SAVED] ' + path + '\n'), path)

    def test_last_line_without_newline(self):
        self.assertEqual(parse_saved_session_path('[SAVED] /a/session_20261103_222302_101671'),
                         '/a/session_20261103_222302_101671')

    def test_unrelated_lines_ignored(self):
        self.assertIsNone(parse_saved_session_path('[SUMMARY] state=DRY_RUN_PASS; goals_sent=0\n'))

    def test_no_trailing_text_accepted(self):
        self.assertIsNone(parse_saved_session_path('pre: [SAVED] /tmp/session_20261104_100650_529617'))

if __name__ == '__main__':
    unittest.main()
