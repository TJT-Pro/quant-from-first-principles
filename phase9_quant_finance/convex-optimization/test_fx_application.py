import os
import unittest

import fx_application as fx
import linear_algebra as la

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestDataLoading(unittest.TestCase):
    def test_loads_expected_count(self):
        returns = fx.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)


class TestBuildStrategyReturns(unittest.TestCase):
    def test_returns_four_series_of_equal_length(self):
        returns = fx.load_log_returns(DATA_PATH)
        strategies = fx.build_strategy_returns(returns)
        self.assertEqual(set(strategies.keys()), {"buy_hold", "momentum", "mean_reversion", "vol_scaled"})
        lengths = {len(v) for v in strategies.values()}
        self.assertEqual(len(lengths), 1)

    def test_momentum_and_mean_reversion_are_not_exact_opposites(self):
        returns = fx.load_log_returns(DATA_PATH)
        strategies = fx.build_strategy_returns(returns)
        momentum, mean_rev = strategies["momentum"], strategies["mean_reversion"]
        sigma = fx.sample_covariance([momentum, mean_rev])
        correlation = sigma[0][1] / (sigma[0][0] ** 0.5 * sigma[1][1] ** 0.5)
        self.assertGreater(correlation, -0.99)

    def test_buy_hold_equals_raw_returns_from_the_lookback_point_on(self):
        returns = [0.01, -0.02, 0.015, 0.03, -0.01] * 10
        strategies = fx.build_strategy_returns(returns, momentum_lookback=3, reversion_lookback=5)
        self.assertEqual(strategies["buy_hold"], returns[5:])


class TestSampleCovariance(unittest.TestCase):
    def test_matches_known_hand_calculation(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 1.0, 4.0, 3.0]
        cov = fx.sample_covariance([a, b])
        mean_a, mean_b = 2.5, 2.5
        expected_cov_ab = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / len(a)
        self.assertAlmostEqual(cov[0][1], expected_cov_ab)
        self.assertAlmostEqual(cov[1][0], expected_cov_ab)

    def test_diagonal_matches_variance(self):
        import statistics
        a = [1.0, 5.0, 3.0, 8.0, 2.0]
        cov = fx.sample_covariance([a])
        self.assertAlmostEqual(cov[0][0], statistics.pvariance(a))


class TestMinVariancePortfolios(unittest.TestCase):
    def test_unconstrained_matches_closed_form(self):
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        result = fx.min_variance_portfolio_unconstrained(sigma)
        sigma_inv = la.invert(sigma)
        ones = [1.0, 1.0, 1.0]
        numer = la.mat_vec(sigma_inv, ones)
        denom = la.dot(ones, numer)
        closed_form = [x / denom for x in numer]
        for a, b in zip(result["x"], closed_form):
            self.assertAlmostEqual(a, b, places=6)

    def test_long_only_has_no_negative_weights(self):
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        result = fx.min_variance_portfolio_long_only(sigma, step=1.0)
        for w in result["x"]:
            self.assertGreaterEqual(w, -1e-6)

    def test_long_only_sums_to_one(self):
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        result = fx.min_variance_portfolio_long_only(sigma, step=1.0)
        self.assertAlmostEqual(sum(result["x"]), 1.0, places=6)

    def test_constrained_variance_never_beats_unconstrained(self):
        sigma = [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 0.01]]
        unconstrained = fx.min_variance_portfolio_unconstrained(sigma)
        long_only = fx.min_variance_portfolio_long_only(sigma, step=1.0)
        unconstrained_var = fx.portfolio_variance(sigma, unconstrained["x"])
        self.assertGreaterEqual(long_only["variance"], unconstrained_var - 1e-9)

    def test_long_only_variance_field_is_true_variance_not_half_variance(self):
        sigma = [[0.04, 0.0], [0.0, 0.09]]
        result = fx.min_variance_portfolio_long_only(sigma, step=1.0)
        self.assertAlmostEqual(result["variance"], fx.portfolio_variance(sigma, result["x"]), places=8)


class TestKKTVerificationHelpers(unittest.TestCase):
    def test_unconstrained_kkt_holds(self):
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        result = fx.min_variance_portfolio_unconstrained(sigma)
        kkt = fx.verify_unconstrained_solution_kkt(sigma, result)
        self.assertTrue(kkt["all_conditions_satisfied"])

    def test_long_only_kkt_holds_when_constraint_binds(self):
        sigma = [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 0.01]]
        result = fx.min_variance_portfolio_long_only(sigma, step=1.0, max_iter=50000, tol=1e-10)
        kkt = fx.verify_long_only_solution_kkt(sigma, result["x"])
        self.assertTrue(kkt["all_conditions_satisfied"])

    def test_long_only_kkt_holds_when_constraint_does_not_bind(self):
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        result = fx.min_variance_portfolio_long_only(sigma, step=1.0)
        kkt = fx.verify_long_only_solution_kkt(sigma, result["x"])
        self.assertTrue(kkt["all_conditions_satisfied"])


class TestRealDataApplication(unittest.TestCase):
    def test_long_only_constraint_binds_on_real_data(self):
        returns = fx.load_log_returns(DATA_PATH)
        strategies = fx.build_strategy_returns(returns)
        sigma = fx.sample_covariance(list(strategies.values()))
        unconstrained = fx.min_variance_portfolio_unconstrained(sigma)
        self.assertTrue(any(w < -1e-6 for w in unconstrained["x"]))

    def test_long_only_solution_costs_more_variance_on_real_data(self):
        returns = fx.load_log_returns(DATA_PATH)
        strategies = fx.build_strategy_returns(returns)
        sigma = fx.sample_covariance(list(strategies.values()))
        unconstrained = fx.min_variance_portfolio_unconstrained(sigma)
        long_only = fx.min_variance_portfolio_long_only(sigma)
        unconstrained_var = fx.portfolio_variance(sigma, unconstrained["x"])
        self.assertGreater(long_only["variance"], unconstrained_var)

    def test_kkt_holds_for_both_real_data_solutions(self):
        returns = fx.load_log_returns(DATA_PATH)
        strategies = fx.build_strategy_returns(returns)
        sigma = fx.sample_covariance(list(strategies.values()))

        unconstrained = fx.min_variance_portfolio_unconstrained(sigma)
        kkt_u = fx.verify_unconstrained_solution_kkt(sigma, unconstrained)
        self.assertTrue(kkt_u["all_conditions_satisfied"])

        long_only = fx.min_variance_portfolio_long_only(sigma)
        kkt_lo = fx.verify_long_only_solution_kkt(sigma, long_only["x"])
        self.assertTrue(kkt_lo["all_conditions_satisfied"])


if __name__ == "__main__":
    unittest.main()
