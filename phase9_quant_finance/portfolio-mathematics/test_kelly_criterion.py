import unittest

import kelly_criterion as kc
import tangency_portfolio as tp


class TestKellyFractionSingleAsset(unittest.TestCase):
    def test_matches_manual_formula(self):
        mu, sigma = 0.001, 0.02
        self.assertAlmostEqual(kc.kelly_fraction_single_asset(mu, sigma), mu / sigma ** 2)

    def test_zero_mean_gives_zero_kelly_fraction(self):
        self.assertAlmostEqual(kc.kelly_fraction_single_asset(0.0, 0.02), 0.0)

    def test_negative_mean_gives_negative_kelly_fraction(self):
        self.assertLess(kc.kelly_fraction_single_asset(-0.001, 0.02), 0.0)


class TestExpectedLogGrowthRate(unittest.TestCase):
    def test_maximized_at_kelly_fraction(self):
        mu, sigma = 0.0005, 0.01
        f_star = kc.kelly_fraction_single_asset(mu, sigma)
        growth_at_star = kc.expected_log_growth_rate(f_star, mu, sigma)
        for delta in (-0.5, -0.2, 0.2, 0.5, 1.0):
            growth_nearby = kc.expected_log_growth_rate(f_star + delta, mu, sigma)
            self.assertLessEqual(growth_nearby, growth_at_star + 1e-12)

    def test_zero_leverage_gives_zero_growth(self):
        self.assertAlmostEqual(kc.expected_log_growth_rate(0.0, 0.001, 0.02), 0.0)


class TestKellyMaximizesRealizedCompoundGrowth(unittest.TestCase):
    def test_full_kelly_beats_half_and_one_point_five_x_kelly_in_simulation(self):
        mu, sigma = 0.0005, 0.01
        f_star = kc.kelly_fraction_single_asset(mu, sigma)
        growth_full = kc.simulate_compound_growth(f_star, mu, sigma, n_periods=252, n_paths=4000, seed=1)
        growth_half = kc.simulate_compound_growth(0.5 * f_star, mu, sigma, n_periods=252, n_paths=4000, seed=1)
        growth_over = kc.simulate_compound_growth(1.5 * f_star, mu, sigma, n_periods=252, n_paths=4000, seed=1)
        self.assertGreater(growth_full, growth_half)
        self.assertGreater(growth_full, growth_over)


class TestKellyWeightsMultiAsset(unittest.TestCase):
    def test_matches_sigma_inverse_mu(self):
        import linear_algebra as la
        mu = [0.05, 0.10, 0.15]
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        kelly_w = kc.kelly_weights_multi_asset(mu, sigma)
        expected = la.mat_vec(la.invert(sigma), mu)
        for a, b in zip(kelly_w, expected):
            self.assertAlmostEqual(a, b, places=10)

    def test_normalized_kelly_weights_match_tangency_portfolio_at_zero_risk_free_rate(self):
        mu = [0.05, 0.10, 0.15]
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        kelly_w = kc.kelly_weights_multi_asset(mu, sigma)
        kelly_normalized = [w / sum(kelly_w) for w in kelly_w]
        tan = tp.tangency_portfolio(mu, sigma, rf=0.0)
        for a, b in zip(kelly_normalized, tan["weights"]):
            self.assertAlmostEqual(a, b, places=8)

    def test_nonzero_risk_free_rate_breaks_the_exact_equivalence(self):
        mu = [0.05, 0.10, 0.15]
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        kelly_w = kc.kelly_weights_multi_asset(mu, sigma)
        kelly_normalized = [w / sum(kelly_w) for w in kelly_w]
        tan = tp.tangency_portfolio(mu, sigma, rf=0.03)
        differs = any(abs(a - b) > 1e-4 for a, b in zip(kelly_normalized, tan["weights"]))
        self.assertTrue(differs)


class TestFractionalKellyComparison(unittest.TestCase):
    def test_full_kelly_has_highest_mean_growth_among_tested_fractions(self):
        mu, sigma = 0.0005, 0.01
        f_star = kc.kelly_fraction_single_asset(mu, sigma)
        result = kc.fractional_kelly_comparison(f_star, mu, sigma, n_periods=252, n_paths=3000,
                                                  fractions=[0.25, 0.5, 1.0, 1.5], seed=1)
        growth_rates = {r["fraction_of_kelly"]: r["mean_log_growth_per_period"] for r in result["results"]}
        self.assertGreater(growth_rates[1.0], growth_rates[0.25])
        self.assertGreater(growth_rates[1.0], growth_rates[1.5])

    def test_terminal_wealth_volatility_increases_with_leverage(self):
        mu, sigma = 0.0005, 0.01
        f_star = kc.kelly_fraction_single_asset(mu, sigma)
        result = kc.fractional_kelly_comparison(f_star, mu, sigma, n_periods=252, n_paths=3000,
                                                  fractions=[0.25, 0.5, 1.0, 1.5], seed=1)
        stds = [r["std_terminal_log_wealth"] for r in result["results"]]
        for earlier, later in zip(stds, stds[1:]):
            self.assertLess(earlier, later)


if __name__ == "__main__":
    unittest.main()
