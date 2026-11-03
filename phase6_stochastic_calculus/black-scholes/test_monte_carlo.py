import math
import statistics
import unittest

import black_scholes as bs
import monte_carlo as mc


class TestSimulateTerminalPrices(unittest.TestCase):
    def test_correct_count(self):
        prices = mc.simulate_terminal_prices(100.0, 1.0, 0.03, 0.2, 1000, seed=1)
        self.assertEqual(len(prices), 1000)

    def test_mean_matches_risk_neutral_growth(self):
        s, t, r, sigma, n = 100.0, 1.0, 0.05, 0.2, 200000
        prices = mc.simulate_terminal_prices(s, t, r, sigma, n, seed=1)
        expected_mean = s * math.exp(r * t)
        self.assertAlmostEqual(statistics.mean(prices), expected_mean, delta=expected_mean * 0.02)

    def test_deterministic_with_seed(self):
        a = mc.simulate_terminal_prices(100.0, 1.0, 0.03, 0.2, 500, seed=7)
        b = mc.simulate_terminal_prices(100.0, 1.0, 0.03, 0.2, 500, seed=7)
        self.assertEqual(a, b)

    def test_antithetic_produces_correct_count(self):
        prices = mc.simulate_terminal_prices(100.0, 1.0, 0.03, 0.2, 999, seed=1, antithetic=True)
        self.assertEqual(len(prices), 999)

    def test_antithetic_pairs_are_mirrored_in_log_space(self):
        prices = mc.simulate_terminal_prices(100.0, 1.0, 0.0, 0.2, 4, seed=1, antithetic=True)
        log_p = [math.log(p / 100.0) for p in prices]
        # drift is (r - q - sigma^2/2)*t = -0.02 here; pairs should be
        # symmetric about that drift.
        drift = (0.0 - 0.0 - 0.5 * 0.2 ** 2) * 1.0
        self.assertAlmostEqual((log_p[0] - drift) + (log_p[1] - drift), 0.0, places=8)

    def test_zero_paths_raises(self):
        with self.assertRaises(ValueError):
            mc.simulate_terminal_prices(100.0, 1.0, 0.03, 0.2, 0)


class TestMonteCarloPrice(unittest.TestCase):
    def test_converges_to_black_scholes_call(self):
        s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.2
        bs_price = bs.call_price(s, k, t, r, sigma)
        result = mc.monte_carlo_price(s, k, t, r, sigma, 200000, "call", seed=1)
        self.assertAlmostEqual(result["price"], bs_price, delta=4 * result["std_error"])

    def test_converges_to_black_scholes_put(self):
        s, k, t, r, sigma, q = 100.0, 105.0, 0.5, 0.02, 0.25, 0.01
        bs_price = bs.put_price(s, k, t, r, sigma, q)
        result = mc.monte_carlo_price(s, k, t, r, sigma, 200000, "put", q, seed=1)
        self.assertAlmostEqual(result["price"], bs_price, delta=4 * result["std_error"])

    def test_confidence_interval_contains_bs_price_typically(self):
        s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.2
        bs_price = bs.call_price(s, k, t, r, sigma)
        contained = 0
        trials = 30
        for i in range(trials):
            result = mc.monte_carlo_price(s, k, t, r, sigma, 5000, "call", seed=i)
            if result["ci_low"] <= bs_price <= result["ci_high"]:
                contained += 1
        # A properly calibrated 95% CI should contain the true value
        # in roughly 95% of independent trials; with only 30 trials,
        # allow generous slack around that rate.
        self.assertGreaterEqual(contained / trials, 0.8)

    def test_invalid_option_type_raises(self):
        with self.assertRaises(ValueError):
            mc.monte_carlo_price(100.0, 100.0, 1.0, 0.02, 0.2, 100, option_type="straddle")

    def test_antithetic_reduces_standard_error(self):
        s, k, t, r, sigma, n = 100.0, 100.0, 1.0, 0.03, 0.2, 20000
        plain = mc.monte_carlo_price(s, k, t, r, sigma, n, "call", seed=1, antithetic=False)
        anti = mc.monte_carlo_price(s, k, t, r, sigma, n, "call", seed=1, antithetic=True)
        self.assertLess(anti["std_error"], plain["std_error"])


class TestConvergenceSeries(unittest.TestCase):
    def test_standard_error_shrinks_with_more_paths(self):
        s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.2
        counts = [100, 1000, 10000, 100000]
        series = mc.monte_carlo_convergence_series(s, k, t, r, sigma, "call", counts, seed=1)
        errors = [row["std_error"] for row in series]
        for earlier, later in zip(errors, errors[1:]):
            self.assertGreater(earlier, later)

    def test_standard_error_scales_like_inverse_sqrt_n(self):
        s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.2
        counts = [1000, 4000]
        series = mc.monte_carlo_convergence_series(s, k, t, r, sigma, "call", counts, seed=1)
        # Quadrupling n should roughly halve the standard error.
        ratio = series[0]["std_error"] / series[1]["std_error"]
        self.assertAlmostEqual(ratio, 2.0, delta=0.3)

    def test_series_prices_converge_to_bs_price(self):
        s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.2
        bs_price = bs.call_price(s, k, t, r, sigma)
        counts = [500, 5000, 50000]
        series = mc.monte_carlo_convergence_series(s, k, t, r, sigma, "call", counts, seed=1)
        self.assertLess(abs(series[-1]["price"] - bs_price), abs(series[0]["price"] - bs_price) + 1e-9)


if __name__ == "__main__":
    unittest.main()
