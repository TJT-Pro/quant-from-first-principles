import math
import unittest

import conjugate_models as cm


class TestBetaPdf(unittest.TestCase):
    def test_uniform_case(self):
        for theta in (0.1, 0.5, 0.9):
            self.assertAlmostEqual(cm.beta_pdf(theta, 1.0, 1.0), 1.0)

    def test_outside_support_is_zero(self):
        self.assertEqual(cm.beta_pdf(-0.1, 2.0, 2.0), 0.0)
        self.assertEqual(cm.beta_pdf(1.1, 2.0, 2.0), 0.0)

    def test_symmetric_case_peaks_at_half(self):
        self.assertGreater(cm.beta_pdf(0.5, 5.0, 5.0), cm.beta_pdf(0.3, 5.0, 5.0))


class TestBetaBinomial(unittest.TestCase):
    def test_update_adds_counts(self):
        model = cm.BetaBinomial(1.0, 1.0)
        posterior = model.update(successes=7, trials=10)
        self.assertAlmostEqual(posterior.a, 8.0)
        self.assertAlmostEqual(posterior.b, 4.0)

    def test_mean_matches_formula(self):
        model = cm.BetaBinomial(2.0, 3.0)
        self.assertAlmostEqual(model.mean(), 2.0 / 5.0)

    def test_variance_matches_formula(self):
        a, b = 2.0, 3.0
        model = cm.BetaBinomial(a, b)
        expected = (a * b) / ((a + b) ** 2 * (a + b + 1))
        self.assertAlmostEqual(model.variance(), expected)

    def test_uninformative_prior_posterior_mean_near_sample_proportion(self):
        model = cm.BetaBinomial(1.0, 1.0)
        posterior = model.update(successes=700, trials=1000)
        self.assertAlmostEqual(posterior.mean(), 0.7, places=2)

    def test_credible_interval_brackets_true_value_with_strong_data(self):
        model = cm.BetaBinomial(1.0, 1.0).update(successes=6000, trials=10000)
        lo, hi = model.credible_interval(0.95)
        self.assertLess(lo, 0.6)
        self.assertGreater(hi, 0.6)
        self.assertLess(hi - lo, 0.03)

    def test_posterior_predictive_sums_to_one(self):
        model = cm.BetaBinomial(2.0, 3.0)
        m = 10
        total = sum(model.posterior_predictive_pmf(y, m) for y in range(m + 1))
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_posterior_predictive_matches_manual_beta_binomial_formula(self):
        model = cm.BetaBinomial(2.0, 3.0)
        y, m = 3, 8
        expected = (math.comb(m, y) * math.exp(cm.log_beta(y + 2.0, m - y + 3.0) - cm.log_beta(2.0, 3.0)))
        self.assertAlmostEqual(model.posterior_predictive_pmf(y, m), expected, places=10)

    def test_invalid_prior_raises(self):
        with self.assertRaises(ValueError):
            cm.BetaBinomial(0.0, 1.0)

    def test_invalid_update_raises(self):
        model = cm.BetaBinomial(1.0, 1.0)
        with self.assertRaises(ValueError):
            model.update(successes=11, trials=10)

    def test_log_evidence_matches_brute_force_numerical_integration(self):
        model = cm.BetaBinomial(2.0, 2.0)
        k, n = 6, 10
        n_grid = 200000
        step = 1.0 / n_grid
        total = 0.0
        for i in range(n_grid + 1):
            theta = i / n_grid
            if theta in (0.0, 1.0):
                continue
            prior_density = cm.beta_pdf(theta, 2.0, 2.0)
            log_binom = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            likelihood = math.exp(log_binom + k * math.log(theta) + (n - k) * math.log(1 - theta))
            total += prior_density * likelihood * step
        self.assertAlmostEqual(math.exp(model.log_evidence(k, n)), total, places=4)


class TestGammaPoisson(unittest.TestCase):
    def test_update_adds_counts_and_exposure(self):
        model = cm.GammaPoisson(2.0, 1.0)
        posterior = model.update(total_count=15, n_intervals=5)
        self.assertAlmostEqual(posterior.a, 17.0)
        self.assertAlmostEqual(posterior.b, 6.0)

    def test_mean_matches_formula(self):
        model = cm.GammaPoisson(6.0, 2.0)
        self.assertAlmostEqual(model.mean(), 3.0)

    def test_variance_matches_formula(self):
        a, b = 6.0, 2.0
        model = cm.GammaPoisson(a, b)
        self.assertAlmostEqual(model.variance(), a / b ** 2)

    def test_posterior_mean_converges_to_true_rate(self):
        import random
        rng = random.Random(1)
        true_lambda = 3.0
        model = cm.GammaPoisson(1.0, 1.0)
        counts = [max(0, round(rng.gauss(true_lambda, math.sqrt(true_lambda)))) for _ in range(2000)]
        model = model.update(sum(counts), len(counts))
        self.assertAlmostEqual(model.mean(), true_lambda, delta=0.15)

    def test_posterior_predictive_sums_close_to_one(self):
        model = cm.GammaPoisson(5.0, 2.0)
        total = sum(model.posterior_predictive_pmf(y) for y in range(200))
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_negative_y_gives_zero_pmf(self):
        model = cm.GammaPoisson(2.0, 1.0)
        self.assertEqual(model.posterior_predictive_pmf(-1), 0.0)

    def test_invalid_prior_raises(self):
        with self.assertRaises(ValueError):
            cm.GammaPoisson(-1.0, 1.0)

    def test_invalid_update_raises(self):
        model = cm.GammaPoisson(1.0, 1.0)
        with self.assertRaises(ValueError):
            model.update(total_count=-5, n_intervals=1)


class TestNormalNormal(unittest.TestCase):
    def test_update_matches_precision_weighted_formula(self):
        mu0, tau0_sq = 0.0, 1.0
        sigma_sq, n, xbar = 4.0, 10, 2.0
        model = cm.NormalNormal(mu0, tau0_sq)
        posterior = model.update(xbar, sigma_sq, n)

        expected_precision = 1 / tau0_sq + n / sigma_sq
        expected_var = 1 / expected_precision
        expected_mean = expected_var * (mu0 / tau0_sq + n * xbar / sigma_sq)
        self.assertAlmostEqual(posterior.mean(), expected_mean)
        self.assertAlmostEqual(posterior.variance(), expected_var)

    def test_more_data_shrinks_posterior_variance(self):
        model = cm.NormalNormal(0.0, 1.0)
        post_small_n = model.update(1.0, 1.0, n=5)
        post_large_n = model.update(1.0, 1.0, n=500)
        self.assertGreater(post_small_n.variance(), post_large_n.variance())

    def test_uninformative_prior_posterior_mean_near_sample_mean(self):
        model = cm.NormalNormal(0.0, 1e6)
        posterior = model.update(sample_mean=3.7, sigma_sq=1.0, n=1000)
        self.assertAlmostEqual(posterior.mean(), 3.7, places=2)

    def test_posterior_predictive_variance_exceeds_posterior_variance(self):
        model = cm.NormalNormal(0.0, 1.0).update(1.0, 2.0, n=10)
        mean, var = model.posterior_predictive(sigma_sq=2.0)
        self.assertGreater(var, model.variance())

    def test_invalid_prior_variance_raises(self):
        with self.assertRaises(ValueError):
            cm.NormalNormal(0.0, 0.0)

    def test_invalid_update_raises(self):
        model = cm.NormalNormal(0.0, 1.0)
        with self.assertRaises(ValueError):
            model.update(1.0, sigma_sq=-1.0, n=10)


if __name__ == "__main__":
    unittest.main()
