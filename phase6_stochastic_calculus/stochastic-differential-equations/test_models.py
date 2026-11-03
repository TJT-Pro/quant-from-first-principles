import math
import statistics
import unittest

import models


class TestGBMExact(unittest.TestCase):
    def test_starts_at_x0(self):
        ts, xs = models.simulate_gbm_exact(100.0, 0.05, 0.2, 1.0, 50, seed=1)
        self.assertEqual(xs[0], 100.0)

    def test_always_positive(self):
        ts, xs = models.simulate_gbm_exact(100.0, 0.05, 0.2, 1.0, 100, seed=1)
        for x in xs:
            self.assertGreater(x, 0.0)

    def test_log_terminal_mean_matches_theory(self):
        x0, mu, sigma, t_end = 100.0, 0.05, 0.2, 1.0
        log_terminals = []
        for i in range(20000):
            _, xs = models.simulate_gbm_exact(x0, mu, sigma, t_end, 10, seed=i)
            log_terminals.append(math.log(xs[-1]))
        expected_mean = math.log(x0) + (mu - 0.5 * sigma ** 2) * t_end
        self.assertAlmostEqual(statistics.mean(log_terminals), expected_mean, delta=0.02)

    def test_log_terminal_variance_matches_theory(self):
        x0, mu, sigma, t_end = 100.0, 0.05, 0.2, 1.0
        log_terminals = []
        for i in range(20000):
            _, xs = models.simulate_gbm_exact(x0, mu, sigma, t_end, 10, seed=i)
            log_terminals.append(math.log(xs[-1]))
        self.assertAlmostEqual(statistics.variance(log_terminals), sigma ** 2 * t_end, delta=0.005)

    def test_zero_volatility_gives_deterministic_exponential_growth(self):
        ts, xs = models.simulate_gbm_exact(100.0, 0.05, 0.0, 2.0, 20, seed=1)
        expected = 100.0 * math.exp(0.05 * 2.0)
        self.assertAlmostEqual(xs[-1], expected, places=8)


class TestOUExact(unittest.TestCase):
    def test_starts_at_x0(self):
        ts, xs = models.simulate_ou_exact(0.0, 2.0, 0.05, 0.3, 1.0, 50, seed=1)
        self.assertEqual(xs[0], 0.0)

    def test_stationary_mean_matches_theory_over_long_horizon(self):
        theta, mu, sigma = 2.0, 0.05, 0.3
        terminals = []
        for i in range(20000):
            _, xs = models.simulate_ou_exact(0.0, theta, mu, sigma, 20.0, 50, seed=i)
            terminals.append(xs[-1])
        self.assertAlmostEqual(statistics.mean(terminals), mu, delta=0.01)

    def test_stationary_variance_matches_theory_over_long_horizon(self):
        theta, mu, sigma = 2.0, 0.05, 0.3
        terminals = []
        for i in range(20000):
            _, xs = models.simulate_ou_exact(0.0, theta, mu, sigma, 20.0, 50, seed=i)
            terminals.append(xs[-1])
        _, expected_var = models.ou_stationary_distribution(theta, mu, sigma)
        self.assertAlmostEqual(statistics.variance(terminals), expected_var, delta=0.002)

    def test_exact_simulation_has_no_bias_even_with_large_steps(self):
        theta, mu, sigma = 2.0, 0.05, 0.3
        terminals_one_step = []
        terminals_many_steps = []
        for i in range(10000):
            _, xs1 = models.simulate_ou_exact(0.0, theta, mu, sigma, 5.0, 1, seed=i)
            _, xs2 = models.simulate_ou_exact(0.0, theta, mu, sigma, 5.0, 500, seed=i + 10**7)
            terminals_one_step.append(xs1[-1])
            terminals_many_steps.append(xs2[-1])
        self.assertAlmostEqual(statistics.mean(terminals_one_step), statistics.mean(terminals_many_steps),
                                delta=0.03)
        self.assertAlmostEqual(statistics.variance(terminals_one_step), statistics.variance(terminals_many_steps),
                                delta=0.01)


class TestOUStationaryDistribution(unittest.TestCase):
    def test_matches_known_formula(self):
        theta, mu, sigma = 3.0, 0.1, 0.5
        mean, var = models.ou_stationary_distribution(theta, mu, sigma)
        self.assertEqual(mean, mu)
        self.assertAlmostEqual(var, sigma ** 2 / (2 * theta))

    def test_nonpositive_theta_raises(self):
        with self.assertRaises(ValueError):
            models.ou_stationary_distribution(0.0, 0.1, 0.5)
        with self.assertRaises(ValueError):
            models.ou_stationary_distribution(-1.0, 0.1, 0.5)


if __name__ == "__main__":
    unittest.main()
