import math
import unittest

import grid_approximation as ga
import conjugate_models as cm


class TestLogSumExp(unittest.TestCase):
    def test_matches_direct_computation_for_small_values(self):
        values = [1.0, 2.0, 0.5]
        direct = math.log(sum(math.exp(v) for v in values))
        self.assertAlmostEqual(ga.log_sum_exp(values), direct, places=10)

    def test_avoids_overflow_for_large_values(self):
        values = [1000.0, 999.0, 998.0]
        result = ga.log_sum_exp(values)
        self.assertTrue(math.isfinite(result))

    def test_all_negative_infinity(self):
        self.assertEqual(ga.log_sum_exp([float("-inf"), float("-inf")]), float("-inf"))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ga.log_sum_exp([])


class TestGridPosteriorAgainstConjugateGroundTruth(unittest.TestCase):
    def test_matches_beta_binomial_posterior_mean_and_variance(self):
        a, b, k, n = 2.0, 3.0, 15, 20
        grid = [i / 10000 for i in range(1, 10000)]

        def log_prior(theta):
            return (a - 1) * math.log(theta) + (b - 1) * math.log(1 - theta)

        def log_likelihood(theta):
            return k * math.log(theta) + (n - k) * math.log(1 - theta)

        probs = ga.grid_posterior(log_prior, log_likelihood, grid)
        grid_mean = ga.posterior_mean(grid, probs)
        grid_var = ga.posterior_variance(grid, probs)

        analytical = cm.BetaBinomial(a, b).update(k, n)
        self.assertAlmostEqual(grid_mean, analytical.mean(), places=3)
        self.assertAlmostEqual(grid_var, analytical.variance(), places=5)

    def test_probabilities_sum_to_one(self):
        grid = [i / 1000 for i in range(1, 1000)]
        log_prior = lambda theta: 0.0
        log_likelihood = lambda theta: 5 * math.log(theta) + 3 * math.log(1 - theta)
        probs = ga.grid_posterior(log_prior, log_likelihood, grid)
        self.assertAlmostEqual(sum(probs), 1.0, places=8)

    def test_credible_interval_matches_conjugate_ground_truth(self):
        a, b, k, n = 1.0, 1.0, 700, 1000
        grid = [i / 20000 for i in range(1, 20000)]

        def log_prior(theta):
            return 0.0

        def log_likelihood(theta):
            return k * math.log(theta) + (n - k) * math.log(1 - theta)

        probs = ga.grid_posterior(log_prior, log_likelihood, grid)
        lo, hi = ga.credible_interval(grid, probs, 0.95)

        analytical = cm.BetaBinomial(a, b).update(k, n)
        analytical_lo, analytical_hi = analytical.credible_interval(0.95)
        self.assertAlmostEqual(lo, analytical_lo, delta=0.01)
        self.assertAlmostEqual(hi, analytical_hi, delta=0.01)


if __name__ == "__main__":
    unittest.main()
