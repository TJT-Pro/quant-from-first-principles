import os
import unittest

import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestDataLoading(unittest.TestCase):
    def test_loads_expected_count(self):
        returns = fx.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)


class TestStrategyConstruction(unittest.TestCase):
    def test_build_strategy_returns_gives_equal_length_series(self):
        returns = fx.load_log_returns(DATA_PATH)
        strategies = fx.build_strategy_returns(returns)
        lengths = {len(s) for s in strategies.values()}
        self.assertEqual(len(lengths), 1)

    def test_momentum_and_mean_reversion_are_not_degenerate_opposites(self):
        import statistics
        returns = fx.load_log_returns(DATA_PATH)
        strategies = fx.build_strategy_returns(returns)
        mom, mr = strategies["momentum"], strategies["mean_reversion"]
        mean_m, mean_r = statistics.mean(mom), statistics.mean(mr)
        cov = sum((mom[i] - mean_m) * (mr[i] - mean_r) for i in range(len(mom))) / len(mom)
        corr = cov / (statistics.pstdev(mom) * statistics.pstdev(mr))
        self.assertGreater(corr, -0.9)

    def test_all_four_strategies_present(self):
        returns = fx.load_log_returns(DATA_PATH)
        strategies = fx.build_strategy_returns(returns)
        self.assertEqual(set(strategies.keys()), {"buy_hold", "momentum", "mean_reversion", "vol_scaled"})


class TestRunFullAnalysis(unittest.TestCase):
    def setUp(self):
        returns = fx.load_log_returns(DATA_PATH)
        self.strategies = fx.build_strategy_returns(returns)

    def test_gmv_weights_sum_to_one(self):
        result = fx.run_full_analysis(self.strategies, rf=0.02)
        self.assertAlmostEqual(sum(result["gmv"]["weights"]), 1.0, places=8)

    def test_tangency_weights_sum_to_one(self):
        result = fx.run_full_analysis(self.strategies, rf=0.02)
        self.assertAlmostEqual(sum(result["tangency"]["weights"]), 1.0, places=8)

    def test_buy_hold_beta_against_itself_is_exactly_one(self):
        result = fx.run_full_analysis(self.strategies, rf=0.02)
        self.assertAlmostEqual(result["betas"]["buy_hold"]["beta"], 1.0, places=8)
        self.assertAlmostEqual(result["betas"]["buy_hold"]["alpha"], 0.0, places=10)
        self.assertAlmostEqual(result["betas"]["buy_hold"]["r_squared"], 1.0, places=8)

    def test_vol_scaled_has_a_meaningfully_higher_beta_than_momentum_or_mean_reversion(self):
        result = fx.run_full_analysis(self.strategies, rf=0.02)
        self.assertGreater(result["betas"]["vol_scaled"]["r_squared"], 0.5)
        self.assertLess(result["betas"]["momentum"]["r_squared"], 0.1)
        self.assertLess(result["betas"]["mean_reversion"]["r_squared"], 0.1)

    def test_euler_risk_decomposition_holds_on_real_data(self):
        result = fx.run_full_analysis(self.strategies, rf=0.02)
        self.assertTrue(result["risk_decomposition"]["matches"])

    def test_risk_contribution_percentages_sum_to_one(self):
        result = fx.run_full_analysis(self.strategies, rf=0.02)
        self.assertAlmostEqual(sum(result["risk_contribution_pct"]), 1.0, places=8)

    def test_kelly_direction_matches_tangency_direction_at_zero_risk_free_rate_on_real_data(self):
        result = fx.run_full_analysis(self.strategies, rf=0.0)
        for a, b in zip(result["kelly_normalized"], result["tangency"]["weights"]):
            self.assertAlmostEqual(a, b, places=6)

    def test_no_strategy_combination_beats_a_realistic_risk_free_rate(self):
        # The honest, thematically consistent finding: with every
        # tested strategy's own expected return near zero or negative
        # (as established via entirely different methods throughout
        # this project family), the theoretically-optimal tangency
        # portfolio, even with full long-short flexibility, cannot
        # produce a positive risk-adjusted excess return over a
        # realistic 2% risk-free rate. This is not a flaw in the
        # portfolio-construction machinery; it is the machinery
        # correctly reporting that no genuine edge exists among these
        # inputs, exactly consistent with the "no linear or nonlinear
        # predictability" finding running throughout this project
        # family via entropy, mutual information, and direct
        # backtesting.
        result = fx.run_full_analysis(self.strategies, rf=0.02)
        self.assertLess(result["tangency"]["sharpe"], 0.0)


if __name__ == "__main__":
    unittest.main()
