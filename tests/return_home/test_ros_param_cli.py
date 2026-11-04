import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/exploration/navigation/return_home'))
from return_home_policy import parse_cli_param_output


class CLIParameterParsing(unittest.TestCase):
    def test_plugin(self):
        self.assertEqual(parse_cli_param_output('GridBased.plugin',
            'String value is: nav2_smac_planner/SmacPlanner2D\n'),
            'nav2_smac_planner/SmacPlanner2D')
    def test_double(self):
        self.assertEqual(parse_cli_param_output('GridBased.cost_travel_multiplier',
            'Double value is: 3.0\n'), 3.0)
    def test_radius(self):
        self.assertEqual(parse_cli_param_output('inflation_layer.inflation_radius',
            'Double value is: 0.7\n'), 0.7)
    def test_arrays(self):
        self.assertEqual(parse_cli_param_output('max_velocity',
            "Double values are: array('d', [1.0, 0.0, 1.5])\n"), [1.,0.,1.5])
        self.assertEqual(parse_cli_param_output('min_velocity',
            "Double values are: array('d', [-0.26, 0.0, -1.5])\n"), [-.26,0.,-1.5])
    def test_unavailable(self):
        with self.assertRaises(ValueError):
            parse_cli_param_output('GridBased.plugin', 'Node not found\n')
    def test_wrong_array_type_fails(self):
        with self.assertRaises(ValueError):
            parse_cli_param_output('max_velocity', 'String value is: [1, 0, 1.5]\n')
    def test_nonfinite_fails(self):
        with self.assertRaises(ValueError):
            parse_cli_param_output('GridBased.cost_travel_multiplier', 'Double value is: nan\n')

if __name__ == '__main__': unittest.main()
