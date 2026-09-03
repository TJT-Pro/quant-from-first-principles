import math
import unittest

import arma_models as arma


class TestSimulateARMA(unittest.TestCase):
    def test_length(self):
        x = arma.simulate_arma(500, 0.0, [0.5], [0.3], seed=1)
        self.assertEqual(len(x), 500)

    def test_deterministic_with_seed(self):
        a = arma.simulate_arma(200, 0.0, [0.4], [0.2], seed=7)
        b = arma.simulate_arma(200, 0.0, [0.4], [0.2], seed=7)
        self.assertEqual(a, b)

    def test_reduces_to_pure_ar_with_no_ma_terms(self):
        import ar_models as ar
        x1 = arma.simulate_arma(500, 1.0, [0.5], [], seed=3)
        x2 = ar.simulate_ar(500, 1.0, [0.5], seed=3, burn_in=200)
        self.assertEqual(x1, x2)

    def test_reduces_to_pure_ma_with_no_ar_terms(self):
        # simulate_arma and simulate_ma consume randomness in
        # different orders (one streams a single history buffer, the
        # other pre-generates a flat array of innovations), so they
        # will not reproduce bit-identical output from the same seed
        # even though they implement the same model; checking that
        # their long-run statistics agree is the correct equivalence
        # test here, not exact reproducibility.
        import ma_models as ma
        import statistics
        x1 = arma.simulate_arma(20000, 0.0, [], [0.4], seed=5, burn_in=0)
        x2 = ma.simulate_ma(20000, 0.0, [0.4], seed=6)
        self.assertAlmostEqual(statistics.mean(x1), statistics.mean(x2), delta=0.05)
        self.assertAlmostEqual(statistics.variance(x1), statistics.variance(x2), delta=0.1)


class TestCSSResiduals(unittest.TestCase):
    def test_pure_ar_matches_manual_calculation(self):
        x = [1.0, 2.0, 4.0, 3.0]
        residuals = arma.css_residuals(x, c=0.0, phis=[0.5], thetas=[])
        # eps_t = x_t - 0.5*x_{t-1}, for t=1..3 (0-indexed from p=1)
        self.assertAlmostEqual(residuals[0], 2.0 - 0.5 * 1.0)
        self.assertAlmostEqual(residuals[1], 4.0 - 0.5 * 2.0)
        self.assertAlmostEqual(residuals[2], 3.0 - 0.5 * 4.0)

    def test_too_short_series_raises(self):
        with self.assertRaises(ValueError):
            arma.css_residuals([1.0, 2.0], c=0.0, phis=[0.5, 0.3], thetas=[])

    def test_residuals_length(self):
        x = list(range(20))
        residuals = arma.css_residuals([float(v) for v in x], c=0.0, phis=[0.3], thetas=[0.2])
        self.assertEqual(len(residuals), 19)


class TestFitCSS(unittest.TestCase):
    def test_recovers_arma11_parameters_approximately(self):
        true_c, true_phi, true_theta = 0.5, 0.5, 0.3
        x = arma.simulate_arma(4000, true_c, [true_phi], [true_theta], seed=1)
        result = arma.fit_css(x, p=1, q=1)
        self.assertAlmostEqual(result["phis"][0], true_phi, delta=0.15)
        self.assertAlmostEqual(result["thetas"][0], true_theta, delta=0.15)

    def test_overflow_during_search_is_penalised_not_raised(self):
        # Regression test for a real crash found during development:
        # certain starting points sent Nelder-Mead through explosive
        # (non-stationary) AR dynamics mid-search, overflowing the
        # recursive residual computation and raising an uncaught
        # OverflowError. The fix catches it and scores that candidate
        # as infinitely bad instead of crashing the whole fit.
        x = arma.simulate_arma(2000, 0.0, [0.3], [0.2], seed=9)
        result = arma.fit_css(x, p=1, q=1, initial_phis=[-0.5], initial_thetas=[0.5])
        self.assertTrue(result["sse"] == result["sse"])  # not NaN
        self.assertNotEqual(result["sse"], float("inf"))

    def test_fitted_model_flagged_stationary_and_invertible(self):
        x = arma.simulate_arma(3000, 0.0, [0.4], [0.2], seed=2)
        result = arma.fit_css(x, p=1, q=1)
        self.assertTrue(result["stationary"])
        self.assertTrue(result["invertible"])

    def test_pure_ar_case_q_zero(self):
        x = arma.simulate_arma(3000, 1.0, [0.6], [], seed=3)
        result = arma.fit_css(x, p=1, q=0)
        self.assertAlmostEqual(result["phis"][0], 0.6, delta=0.1)
        self.assertEqual(result["thetas"], [])


class TestLogLikelihoodGaussian(unittest.TestCase):
    def test_matches_manual_formula(self):
        residuals = [1.0, -1.0, 0.5, -0.5]
        sigma2 = 1.0
        n = len(residuals)
        sse = sum(e ** 2 for e in residuals)
        expected = -0.5 * n * math.log(2 * math.pi * sigma2) - sse / (2 * sigma2)
        self.assertAlmostEqual(arma.log_likelihood_gaussian(residuals, sigma2), expected)

    def test_nonpositive_sigma2_raises(self):
        with self.assertRaises(ValueError):
            arma.log_likelihood_gaussian([1.0, 2.0], 0.0)

    def test_smaller_sse_gives_higher_likelihood(self):
        ll_small = arma.log_likelihood_gaussian([0.1, -0.1], 1.0)
        ll_large = arma.log_likelihood_gaussian([2.0, -2.0], 1.0)
        self.assertGreater(ll_small, ll_large)


class TestParameterRedundancy(unittest.TestCase):
    def test_arma11_on_white_noise_finds_near_cancelling_ridge(self):
        # A real, documented ARMA pathology: fitting ARMA(1,1) to data
        # with no genuine ARMA structure (here, synthetic white noise)
        # tends to converge to a phi ~ -theta pair (the AR and MA
        # polynomials nearly cancel, (1-phi*L)/(1+theta*L) ~ 1), from
        # multiple different starting points, rather than to phi=theta=0.
        import random
        rng = random.Random(42)
        x = [rng.gauss(0, 1) for _ in range(3000)]
        fit_a = arma.fit_css(x, p=1, q=1, initial_phis=[0.3], initial_thetas=[-0.3])
        fit_b = arma.fit_css(x, p=1, q=1, initial_phis=[-0.4], initial_thetas=[0.4])
        # Both fits should land on the same ridge: phi close to -theta.
        self.assertAlmostEqual(fit_a["phis"][0], -fit_a["thetas"][0], delta=0.05)
        self.assertAlmostEqual(fit_b["phis"][0], -fit_b["thetas"][0], delta=0.05)

    def test_redundant_fit_barely_improves_on_white_noise_sse(self):
        import random
        rng = random.Random(43)
        x = [rng.gauss(0, 1) for _ in range(3000)]
        fit_11 = arma.fit_css(x, p=1, q=1)
        fit_00 = arma.fit_css(x, p=0, q=0)
        # The extra two parameters should buy almost nothing in SSE.
        relative_improvement = (fit_00["sse"] - fit_11["sse"]) / fit_00["sse"]
        self.assertLess(relative_improvement, 0.01)


if __name__ == "__main__":
    unittest.main()
