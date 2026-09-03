import math
import unittest

import sde_solvers as sde


class TestBrownianIncrementHelpers(unittest.TestCase):
    def test_aggregate_matches_direct_sum(self):
        fine = [0.1, 0.2, -0.1, 0.3, 0.05, -0.05]
        coarse = sde.aggregate_increments(fine, 3)
        self.assertAlmostEqual(coarse[0], 0.1 + 0.2)
        self.assertAlmostEqual(coarse[1], -0.1 + 0.3)
        self.assertAlmostEqual(coarse[2], 0.05 - 0.05)

    def test_aggregate_preserves_total_sum(self):
        fine = sde.generate_fine_brownian_increments(1.0, 1000, seed=1)
        coarse = sde.aggregate_increments(fine, 10)
        self.assertAlmostEqual(sum(fine), sum(coarse), places=10)

    def test_incompatible_lengths_raise(self):
        with self.assertRaises(ValueError):
            sde.aggregate_increments([1.0, 2.0, 3.0], 2)

    def test_generated_increments_have_correct_variance_scale(self):
        import statistics
        fine = sde.generate_fine_brownian_increments(2.0, 5000, seed=1)
        dt = 2.0 / 5000
        self.assertAlmostEqual(statistics.pvariance(fine), dt, delta=dt * 0.2)


class TestEulerMaruyamaOnGBM(unittest.TestCase):
    def setUp(self):
        self.mu_param = 0.05
        self.sigma_param = 0.2
        self.x0 = 100.0
        self.t_end = 1.0
        self.mu_fn = lambda x, t: self.mu_param * x
        self.sigma_fn = lambda x, t: self.sigma_param * x
        self.sigma_x_fn = lambda x, t: self.sigma_param

    def _exact_solution(self, w_t):
        return self.x0 * math.exp((self.mu_param - 0.5 * self.sigma_param ** 2) * self.t_end
                                   + self.sigma_param * w_t)

    def test_converges_to_exact_solution_as_steps_increase(self):
        dw_fine = sde.generate_fine_brownian_increments(self.t_end, 20000, seed=1)
        w_t = sum(dw_fine)
        exact = self._exact_solution(w_t)

        dw_coarse = sde.aggregate_increments(dw_fine, 100)
        dw_fine_matching = sde.aggregate_increments(dw_fine, 20000)

        result_coarse = sde.euler_maruyama(self.mu_fn, self.sigma_fn, self.x0, 0.0, self.t_end, dw_coarse)
        result_fine = sde.euler_maruyama(self.mu_fn, self.sigma_fn, self.x0, 0.0, self.t_end, dw_fine_matching)

        error_coarse = abs(result_coarse["x"][-1] - exact)
        error_fine = abs(result_fine["x"][-1] - exact)
        self.assertLess(error_fine, error_coarse)

    def test_path_starts_at_x0(self):
        dw = sde.generate_fine_brownian_increments(self.t_end, 100, seed=1)
        result = sde.euler_maruyama(self.mu_fn, self.sigma_fn, self.x0, 0.0, self.t_end, dw)
        self.assertEqual(result["x"][0], self.x0)

    def test_zero_diffusion_reduces_to_deterministic_growth(self):
        zero_sigma = lambda x, t: 0.0
        dw = [0.0] * 1000
        result = sde.euler_maruyama(self.mu_fn, zero_sigma, self.x0, 0.0, self.t_end, dw)
        expected = self.x0 * math.exp(self.mu_param * self.t_end)
        self.assertAlmostEqual(result["x"][-1], expected, places=2)


class TestMilsteinOnGBM(unittest.TestCase):
    def setUp(self):
        self.mu_param = 0.05
        self.sigma_param = 0.2
        self.x0 = 100.0
        self.t_end = 1.0
        self.mu_fn = lambda x, t: self.mu_param * x
        self.sigma_fn = lambda x, t: self.sigma_param * x
        self.sigma_x_fn = lambda x, t: self.sigma_param

    def _exact_solution(self, w_t):
        return self.x0 * math.exp((self.mu_param - 0.5 * self.sigma_param ** 2) * self.t_end
                                   + self.sigma_param * w_t)

    def test_milstein_more_accurate_than_euler_maruyama_at_same_step_count(self):
        dw = sde.generate_fine_brownian_increments(self.t_end, 5000, seed=1)
        w_t = sum(dw)
        exact = self._exact_solution(w_t)

        em_result = sde.euler_maruyama(self.mu_fn, self.sigma_fn, self.x0, 0.0, self.t_end, dw)
        mil_result = sde.milstein(self.mu_fn, self.sigma_fn, self.x0, 0.0, self.t_end, dw,
                                   sigma_x=self.sigma_x_fn)

        em_error = abs(em_result["x"][-1] - exact)
        mil_error = abs(mil_result["x"][-1] - exact)
        self.assertLess(mil_error, em_error)

    def test_numerical_sigma_x_matches_analytic_closely(self):
        dw = sde.generate_fine_brownian_increments(self.t_end, 2000, seed=2)
        analytic_result = sde.milstein(self.mu_fn, self.sigma_fn, self.x0, 0.0, self.t_end, dw,
                                        sigma_x=self.sigma_x_fn)
        numerical_result = sde.milstein(self.mu_fn, self.sigma_fn, self.x0, 0.0, self.t_end, dw)
        self.assertAlmostEqual(analytic_result["x"][-1], numerical_result["x"][-1], places=4)

    def test_path_starts_at_x0(self):
        dw = sde.generate_fine_brownian_increments(self.t_end, 100, seed=1)
        result = sde.milstein(self.mu_fn, self.sigma_fn, self.x0, 0.0, self.t_end, dw, sigma_x=self.sigma_x_fn)
        self.assertEqual(result["x"][0], self.x0)


if __name__ == "__main__":
    unittest.main()
