import unittest

import dual_numbers as dn
from autodiff_engine import Value
import jacobian_comparison as jc


class TestJacobianNumerical(unittest.TestCase):
    def test_known_values(self):
        f = lambda v: [v[0] + v[1], v[0] * v[1]]
        jac = jc.jacobian_numerical(f, [2.0, 3.0])
        self.assertAlmostEqual(jac[0][0], 1.0, places=4)
        self.assertAlmostEqual(jac[0][1], 1.0, places=4)
        self.assertAlmostEqual(jac[1][0], 3.0, places=4)
        self.assertAlmostEqual(jac[1][1], 2.0, places=4)


class TestThreeWayAgreement(unittest.TestCase):
    def test_forward_reverse_numerical_all_agree(self):
        # Built only from operations both the dual-number engine and
        # the Value engine implement, so the same mathematical
        # function can be handed to all three methods.
        def f_dual_shared(v):
            x, y = v
            return [x * x + y, dn.exp(x) * y]

        def f_value_shared(v):
            x, y = v
            return [x * x + y, x.exp() * y]

        def f_plain(v):
            import math
            x, y = v
            return [x * x + y, math.exp(x) * y]

        x = [0.6, 1.4]
        jac_forward = jc.jacobian_forward(f_dual_shared, x)
        jac_reverse = jc.jacobian_reverse(f_value_shared, x)
        jac_numerical = jc.jacobian_numerical(f_plain, x)

        for row in range(2):
            for col in range(2):
                self.assertAlmostEqual(jac_forward[row][col], jac_reverse[row][col], places=6)
                self.assertAlmostEqual(jac_forward[row][col], jac_numerical[row][col], places=4)


class TestCostComparison(unittest.TestCase):
    def test_forward_cheaper_for_many_outputs_few_inputs(self):
        result = jc.compare_costs(n_inputs=1, n_outputs=1000)
        self.assertEqual(result["cheaper_method"], "forward")
        self.assertEqual(result["forward_mode_passes"], 1)
        self.assertEqual(result["reverse_mode_passes"], 1000)

    def test_reverse_cheaper_for_many_inputs_one_output(self):
        result = jc.compare_costs(n_inputs=1000000, n_outputs=1)
        self.assertEqual(result["cheaper_method"], "reverse")
        self.assertEqual(result["reverse_mode_passes"], 1)

    def test_tie_for_equal_dimensions(self):
        result = jc.compare_costs(n_inputs=5, n_outputs=5)
        self.assertEqual(result["cheaper_method"], "tie")

    def test_neural_network_scale_favors_reverse(self):
        # A tiny network with, say, 50 parameters and a single scalar
        # loss: reverse mode needs one backward pass; forward mode
        # would need fifty full forward passes.
        result = jc.compare_costs(n_inputs=50, n_outputs=1)
        self.assertEqual(result["cheaper_method"], "reverse")
        self.assertEqual(result["forward_mode_passes"], 50)
        self.assertEqual(result["reverse_mode_passes"], 1)


if __name__ == "__main__":
    unittest.main()
