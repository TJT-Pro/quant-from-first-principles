import math
import unittest

import gradient_methods as gm
import linear_algebra as la


class TestBacktrackingLineSearch(unittest.TestCase):
    def test_satisfies_armijo_condition(self):
        f = lambda x: x[0] ** 2 + x[1] ** 2
        x = [3.0, 4.0]
        grad = [6.0, 8.0]
        direction = [-6.0, -8.0]
        alpha = gm.backtracking_line_search(f, x, grad, direction)
        x_new = la.vec_add(x, la.vec_scale(direction, alpha))
        directional_derivative = la.dot(grad, direction)
        self.assertLessEqual(f(x_new), f(x) + 1e-4 * alpha * directional_derivative + 1e-12)

    def test_step_size_is_positive_and_bounded(self):
        f = lambda x: x[0] ** 2
        alpha = gm.backtracking_line_search(f, [5.0], [10.0], [-10.0])
        self.assertGreater(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)


class TestGradientDescent(unittest.TestCase):
    def test_converges_on_simple_quadratic(self):
        f = lambda x: (x[0] - 3) ** 2 + (x[1] + 2) ** 2
        result = gm.gradient_descent(f, [0.0, 0.0])
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["x"][0], 3.0, places=4)
        self.assertAlmostEqual(result["x"][1], -2.0, places=4)

    def test_isotropic_quadratic_converges_in_one_step_with_line_search(self):
        f = lambda x: (x[0] - 5) ** 2 + (x[1] - 5) ** 2
        result = gm.gradient_descent(f, [0.0, 0.0])
        self.assertEqual(result["iterations"], 1)

    def test_ill_conditioned_quadratic_still_converges_but_slowly(self):
        f = lambda x: 100 * x[0] ** 2 + x[1] ** 2
        result = gm.gradient_descent(f, [1.0, 1.0])
        self.assertTrue(result["converged"])
        self.assertGreater(result["iterations"], 100)
        self.assertAlmostEqual(result["x"][0], 0.0, places=3)
        self.assertAlmostEqual(result["x"][1], 0.0, places=3)

    def test_history_is_monotonically_nonincreasing(self):
        f = lambda x: (x[0] - 1) ** 2 + (x[1] - 1) ** 2 + 0.5 * x[0] * x[1]
        result = gm.gradient_descent(f, [10.0, -10.0])
        history = result["history"]
        for earlier, later in zip(history, history[1:]):
            self.assertLessEqual(later, earlier + 1e-10)

    def test_fixed_step_mode_also_converges_on_well_scaled_problem(self):
        f = lambda x: x[0] ** 2 + x[1] ** 2
        result = gm.gradient_descent(f, [5.0, 5.0], use_line_search=False, fixed_step=0.1)
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["x"][0], 0.0, places=3)

    def test_custom_gradient_function_matches_numerical_default(self):
        f = lambda x: x[0] ** 2 + 2 * x[1] ** 2
        analytic_grad = lambda x: [2 * x[0], 4 * x[1]]
        result_numeric = gm.gradient_descent(f, [3.0, -2.0])
        result_analytic = gm.gradient_descent(f, [3.0, -2.0], grad_fn=analytic_grad)
        self.assertAlmostEqual(result_numeric["x"][0], result_analytic["x"][0], places=4)
        self.assertAlmostEqual(result_numeric["x"][1], result_analytic["x"][1], places=4)


if __name__ == "__main__":
    unittest.main()
