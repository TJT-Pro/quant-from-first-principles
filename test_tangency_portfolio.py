import math
import random
import unittest

import tangency_portfolio as tp
import mean_variance as mv

MU = [0.05, 0.10, 0.15]
SIGMA = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
RF = 0.02


class TestSharpeRatio(unittest.TestCase):
    def test_matches_manual_computation(self):
        w = [0.5, 0.3, 0.2]
        expected = (mv.portfolio_return(w, MU) - RF) / math.sqrt(mv.portfolio_variance(w, SIGMA))
        self.assertAlmostEqual(tp.sharpe_ratio(w, MU, SIGMA, RF), expected)


class TestTangencyPortfolio(unittest.TestCase):
    def test_weights_sum_to_one(self):
        tan = tp.tangency_portfolio(MU, SIGMA, RF)
        self.assertAlmostEqual(sum(tan["weights"]), 1.0, places=10)

    def test_maximizes_sharpe_ratio_over_random_search(self):
        tan = tp.tangency_portfolio(MU, SIGMA, RF)
        rng = random.Random(1)
        best_random_sharpe = -999.0
        for _ in range(50000):
            raw = [rng.uniform(-1, 2) for _ in range(3)]
            s = sum(raw)
            w = [x / s for x in raw]
            sharpe = tp.sharpe_ratio(w, MU, SIGMA, RF)
            best_random_sharpe = max(best_random_sharpe, sharpe)
        self.assertGreaterEqual(tan["sharpe"] + 1e-6, best_random_sharpe)

    def test_reported_sharpe_matches_direct_computation(self):
        tan = tp.tangency_portfolio(MU, SIGMA, RF)
        direct = tp.sharpe_ratio(tan["weights"], MU, SIGMA, RF)
        self.assertAlmostEqual(tan["sharpe"], direct, places=10)

    def test_different_risk_free_rate_gives_different_tangency_portfolio(self):
        tan_low_rf = tp.tangency_portfolio(MU, SIGMA, 0.0)
        tan_high_rf = tp.tangency_portfolio(MU, SIGMA, 0.08)
        self.assertNotAlmostEqual(tan_low_rf["weights"][0], tan_high_rf["weights"][0], places=3)


class TestCapitalMarketLine(unittest.TestCase):
    def test_return_at_zero_std_equals_risk_free_rate(self):
        tan = tp.tangency_portfolio(MU, SIGMA, RF)
        self.assertAlmostEqual(tp.capital_market_line(tan, RF, 0.0), RF)

    def test_return_at_tangency_std_equals_tangency_return(self):
        tan = tp.tangency_portfolio(MU, SIGMA, RF)
        tangency_std = math.sqrt(tan["variance"])
        cml_return = tp.capital_market_line(tan, RF, tangency_std)
        self.assertAlmostEqual(cml_return, tan["return"], places=8)

    def test_slope_equals_tangency_sharpe_ratio(self):
        tan = tp.tangency_portfolio(MU, SIGMA, RF)
        r1 = tp.capital_market_line(tan, RF, 0.1)
        r2 = tp.capital_market_line(tan, RF, 0.2)
        slope = (r2 - r1) / 0.1
        self.assertAlmostEqual(slope, tan["sharpe"], places=8)


class TestTwoFundCombination(unittest.TestCase):
    def test_alpha_zero_is_pure_risk_free(self):
        tan = tp.tangency_portfolio(MU, SIGMA, RF)
        combo = tp.two_fund_combination(tan, RF, 0.0)
        self.assertAlmostEqual(combo["return"], RF)
        self.assertAlmostEqual(combo["std"], 0.0)

    def test_alpha_one_is_pure_tangency(self):
        tan = tp.tangency_portfolio(MU, SIGMA, RF)
        combo = tp.two_fund_combination(tan, RF, 1.0)
        self.assertAlmostEqual(combo["return"], tan["return"], places=10)
        self.assertAlmostEqual(combo["variance"], tan["variance"], places=10)

    def test_every_combination_has_the_same_sharpe_ratio_as_tangency(self):
        tan = tp.tangency_portfolio(MU, SIGMA, RF)
        for alpha in (0.3, 0.7, 1.0, 1.5, 2.5):
            combo = tp.two_fund_combination(tan, RF, alpha)
            if combo["std"] > 0:
                combo_sharpe = (combo["return"] - RF) / combo["std"]
                self.assertAlmostEqual(combo_sharpe, tan["sharpe"], places=8)

    def test_leverage_scales_return_and_std_linearly(self):
        tan = tp.tangency_portfolio(MU, SIGMA, RF)
        combo1 = tp.two_fund_combination(tan, RF, 1.0)
        combo2 = tp.two_fund_combination(tan, RF, 2.0)
        self.assertAlmostEqual(combo2["std"], 2 * combo1["std"], places=8)
        self.assertAlmostEqual(combo2["return"] - RF, 2 * (combo1["return"] - RF), places=8)


if __name__ == "__main__":
    unittest.main()
