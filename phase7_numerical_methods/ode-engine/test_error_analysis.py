import math
import unittest

import error_analysis as ea


class TestGlobalError(unittest.TestCase):
    def test_error_shrinks_with_smaller_h(self):
        f = lambda t, y: -y
        true_solution = lambda t: math.exp(-t)
        err_coarse = ea.global_error(f, 0.0, 1.0, 1.0, 0.1, "euler", true_solution)
        err_fine = ea.global_error(f, 0.0, 1.0, 1.0, 0.01, "euler", true_solution)
        self.assertGreater(err_coarse, err_fine)

    def test_zero_at_exact_solution(self):
        f = lambda t, y: 2 * t
        true_solution = lambda t: t ** 2
        err = ea.global_error(f, 0.0, 0.0, 1.0, 0.25, "rk4", true_solution)
        self.assertAlmostEqual(err, 0.0, places=8)


class TestEstimateConvergenceOrder(unittest.TestCase):
    def _check_order(self, method, expected_order, tolerance=0.15):
        f = lambda t, y: -y
        true_solution = lambda t: math.exp(-t)
        h_values = [0.2, 0.1, 0.05, 0.025, 0.0125]
        result = ea.estimate_convergence_order(f, 0.0, 1.0, 2.0, h_values, method, true_solution)
        self.assertAlmostEqual(result["estimated_order"], expected_order, delta=tolerance)

    def test_euler_is_first_order(self):
        self._check_order("euler", 1.0)

    def test_improved_euler_is_second_order(self):
        self._check_order("improved_euler", 2.0)

    def test_midpoint_is_second_order(self):
        self._check_order("midpoint", 2.0)

    def test_rk4_is_fourth_order(self):
        self._check_order("rk4", 4.0)

    def test_errors_are_returned_and_decreasing(self):
        f = lambda t, y: -y
        true_solution = lambda t: math.exp(-t)
        h_values = [0.2, 0.1, 0.05]
        result = ea.estimate_convergence_order(f, 0.0, 1.0, 1.0, h_values, "rk4", true_solution)
        errors = result["errors"]
        for earlier, later in zip(errors, errors[1:]):
            self.assertGreaterEqual(earlier, later)

    def test_different_ode_confirms_order_is_not_problem_specific(self):
        # A different linear ODE, y' = 2y, to confirm the estimated
        # order reflects the method, not something specific to the
        # decay example used elsewhere.
        f = lambda t, y: 2 * y
        true_solution = lambda t: math.exp(2 * t)
        h_values = [0.05, 0.025, 0.0125, 0.00625]
        result = ea.estimate_convergence_order(f, 0.0, 1.0, 1.0, h_values, "rk4", true_solution)
        self.assertAlmostEqual(result["estimated_order"], 4.0, delta=0.2)


class TestRichardsonExtrapolation(unittest.TestCase):
    def test_improves_on_either_individual_estimate(self):
        f = lambda t, y: -y
        true_solution = math.exp(-1.0)
        h = 0.1
        plain = ea.global_error(f, 0.0, 1.0, 1.0, h, "euler", lambda t: math.exp(-t))
        extrapolated_value = ea.richardson_extrapolate(f, 0.0, 1.0, 1.0, h, "euler", order=1)
        extrapolated_error = abs(extrapolated_value - true_solution)
        self.assertLess(extrapolated_error, plain)

    def test_extrapolated_rk4_is_extremely_accurate(self):
        f = lambda t, y: -y
        true_solution = math.exp(-1.0)
        value = ea.richardson_extrapolate(f, 0.0, 1.0, 1.0, 0.1, "rk4", order=4)
        self.assertAlmostEqual(value, true_solution, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
