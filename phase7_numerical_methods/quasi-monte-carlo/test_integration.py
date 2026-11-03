import math
import unittest

import integration as intg


class TestMonteCarloIntegrate(unittest.TestCase):
    def test_known_integral_x_squared(self):
        f = lambda x: x[0] ** 2
        result = intg.monte_carlo_integrate(f, dim=1, n_points=200000, seed=1)
        self.assertAlmostEqual(result, 1 / 3, delta=0.01)

    def test_known_integral_product_of_coordinates(self):
        f = lambda x: x[0] * x[1]
        result = intg.monte_carlo_integrate(f, dim=2, n_points=200000, seed=1)
        self.assertAlmostEqual(result, 0.25, delta=0.01)

    def test_constant_function_gives_exact_result(self):
        f = lambda x: 5.0
        result = intg.monte_carlo_integrate(f, dim=3, n_points=100, seed=1)
        self.assertAlmostEqual(result, 5.0)

    def test_deterministic_given_seed(self):
        f = lambda x: x[0]
        r1 = intg.monte_carlo_integrate(f, dim=1, n_points=1000, seed=42)
        r2 = intg.monte_carlo_integrate(f, dim=1, n_points=1000, seed=42)
        self.assertEqual(r1, r2)


class TestQMCIntegrate(unittest.TestCase):
    def test_known_integral_x_squared(self):
        f = lambda x: x[0] ** 2
        result = intg.qmc_integrate(f, dim=1, n_points=5000)
        self.assertAlmostEqual(result, 1 / 3, delta=0.001)

    def test_known_integral_exp_sum(self):
        f = lambda x: math.exp(sum(x))
        exact = (math.e - 1) ** 2
        result = intg.qmc_integrate(f, dim=2, n_points=5000)
        self.assertAlmostEqual(result, exact, delta=0.005)

    def test_deterministic_no_seed_needed(self):
        f = lambda x: x[0] * x[1]
        r1 = intg.qmc_integrate(f, dim=2, n_points=1000)
        r2 = intg.qmc_integrate(f, dim=2, n_points=1000)
        self.assertEqual(r1, r2)

    def test_constant_function_gives_exact_result(self):
        f = lambda x: 3.0
        result = intg.qmc_integrate(f, dim=2, n_points=500)
        self.assertAlmostEqual(result, 3.0)


class TestEstimateConvergenceOrder(unittest.TestCase):
    def test_exact_power_law_recovers_exact_order(self):
        n_values = [100, 400, 1600, 6400]
        c, p_true = 2.0, 0.5
        errors = [c * n ** (-p_true) for n in n_values]
        order = intg.estimate_convergence_order(errors, n_values)
        self.assertAlmostEqual(order, p_true, places=8)

    def test_exact_power_law_order_one(self):
        n_values = [50, 200, 800]
        errors = [3.0 / n for n in n_values]
        order = intg.estimate_convergence_order(errors, n_values)
        self.assertAlmostEqual(order, 1.0, places=8)


class TestMonteCarloVsQMCConvergence(unittest.TestCase):
    def test_monte_carlo_order_near_half(self):
        f = lambda x: math.exp(x[0])
        exact = math.e - 1
        n_values = [100, 400, 1600, 6400]
        errors = [intg.monte_carlo_repeated_error(f, 1, n, exact, n_repeats=25, seed=1) for n in n_values]
        order = intg.estimate_convergence_order(errors, n_values)
        self.assertAlmostEqual(order, 0.5, delta=0.15)

    def test_qmc_order_substantially_better_than_monte_carlo(self):
        f = lambda x: math.exp(x[0])
        exact = math.e - 1
        n_values = [100, 400, 1600, 6400]
        mc_errors = [intg.monte_carlo_repeated_error(f, 1, n, exact, n_repeats=25, seed=1) for n in n_values]
        qmc_errors = [abs(intg.qmc_integrate(f, 1, n) - exact) for n in n_values]
        mc_order = intg.estimate_convergence_order(mc_errors, n_values)
        qmc_order = intg.estimate_convergence_order(qmc_errors, n_values)
        self.assertGreater(qmc_order, mc_order)
        self.assertGreater(qmc_order, 0.8)

    def test_qmc_achieves_lower_error_at_matched_sample_size(self):
        f = lambda x: math.exp(sum(x))
        exact = (math.e - 1) ** 2
        n = 2000
        mc_error = intg.monte_carlo_repeated_error(f, 2, n, exact, n_repeats=25, seed=1)
        qmc_error = abs(intg.qmc_integrate(f, 2, n) - exact)
        self.assertLess(qmc_error, mc_error)


if __name__ == "__main__":
    unittest.main()
