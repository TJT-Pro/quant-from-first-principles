import math
import unittest

import convergence_analysis as ca
import sde_solvers as sde


class TestEstimateConvergenceOrder(unittest.TestCase):
    def test_exact_power_law_recovers_exact_order(self):
        step_sizes = [0.1, 0.05, 0.025, 0.0125]
        c = 2.0
        p_true = 0.5
        errors = [c * h ** p_true for h in step_sizes]
        order = ca.estimate_convergence_order(errors, step_sizes)
        self.assertAlmostEqual(order, p_true, places=8)

    def test_exact_power_law_order_one(self):
        step_sizes = [0.2, 0.1, 0.05, 0.025]
        errors = [3.0 * h for h in step_sizes]
        order = ca.estimate_convergence_order(errors, step_sizes)
        self.assertAlmostEqual(order, 1.0, places=8)


class TestLinearDriftExactMeanError(unittest.TestCase):
    def test_matches_manual_recursion(self):
        mu_param, x0, t_end, n = 0.05, 1.0, 1.0, 8
        h = t_end / n
        expected = x0
        for _ in range(n):
            expected *= (1 + mu_param * h)
        self.assertAlmostEqual(ca.linear_drift_exact_mean_error(mu_param, x0, t_end, n), expected)

    def test_converges_to_exact_exponential_as_steps_increase(self):
        mu_param, x0, t_end = 0.05, 1.0, 1.0
        exact = x0 * math.exp(mu_param * t_end)
        coarse = ca.linear_drift_exact_mean_error(mu_param, x0, t_end, 4)
        fine = ca.linear_drift_exact_mean_error(mu_param, x0, t_end, 256)
        self.assertLess(abs(fine - exact), abs(coarse - exact))

    def test_weak_order_on_linear_drift_matches_theory(self):
        mu_param, x0, t_end = 0.05, 1.0, 1.0
        exact = x0 * math.exp(mu_param * t_end)
        step_counts = [4, 8, 16, 32, 64, 128]
        step_sizes = [t_end / n for n in step_counts]
        errors = [abs(ca.linear_drift_exact_mean_error(mu_param, x0, t_end, n) - exact) for n in step_counts]
        order = ca.estimate_convergence_order(errors, step_sizes)
        self.assertAlmostEqual(order, 1.0, delta=0.05)


class TestStrongConvergenceOrders(unittest.TestCase):
    def setUp(self):
        self.mu_param = 0.05
        self.sigma_param = 0.3
        self.x0 = 1.0
        self.t_end = 1.0
        self.mu_fn = lambda x, t: self.mu_param * x
        self.sigma_fn = lambda x, t: self.sigma_param * x
        self.sigma_x_fn = lambda x, t: self.sigma_param
        self.exact_fn = lambda x0, w_t: x0 * math.exp(
            (self.mu_param - 0.5 * self.sigma_param ** 2) * self.t_end + self.sigma_param * w_t)

    def test_euler_maruyama_strong_order_near_half(self):
        n_fine = 2048
        step_counts = [16, 32, 64, 128]
        step_sizes = [self.t_end / n for n in step_counts]
        errors = [ca.strong_error(sde.euler_maruyama, self.mu_fn, self.sigma_fn, None, self.x0, self.t_end,
                                   n, n_fine, 3000, self.exact_fn, seed=1) for n in step_counts]
        order = ca.estimate_convergence_order(errors, step_sizes)
        self.assertAlmostEqual(order, 0.5, delta=0.2)

    def test_milstein_strong_order_near_one(self):
        n_fine = 2048
        step_counts = [16, 32, 64, 128]
        step_sizes = [self.t_end / n for n in step_counts]
        errors = [ca.strong_error(sde.milstein, self.mu_fn, self.sigma_fn, self.sigma_x_fn, self.x0, self.t_end,
                                   n, n_fine, 3000, self.exact_fn, seed=1) for n in step_counts]
        order = ca.estimate_convergence_order(errors, step_sizes)
        self.assertAlmostEqual(order, 1.0, delta=0.25)

    def test_milstein_has_lower_error_than_euler_maruyama_at_matched_step_count(self):
        n_fine = 2048
        n_coarse = 32
        em_error = ca.strong_error(sde.euler_maruyama, self.mu_fn, self.sigma_fn, None, self.x0, self.t_end,
                                    n_coarse, n_fine, 3000, self.exact_fn, seed=2)
        mil_error = ca.strong_error(sde.milstein, self.mu_fn, self.sigma_fn, self.sigma_x_fn, self.x0, self.t_end,
                                     n_coarse, n_fine, 3000, self.exact_fn, seed=2)
        self.assertLess(mil_error, em_error)


if __name__ == "__main__":
    unittest.main()
