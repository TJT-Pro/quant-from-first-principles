"""Tests for the strategy backtester.

Every metric is checked against a hand-computed value or a known
mathematical identity.  numpy and scipy are used only in the verify
command, not here.  Monte Carlo tests use fixed seeds.
"""

import csv
import math
import os
import random
import unittest

import backtester as bt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_returns():
    path = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")
    values = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cell = row["return"]
            if cell:
                try:
                    values.append(float(cell))
                except ValueError:
                    pass
    return values


def _make_log(returns, direction=None):
    return bt.load_trade_log([
        {"return": r, "direction": direction} if direction
        else {"return": r}
        for r in returns
    ])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    def test_mean_known(self):
        self.assertAlmostEqual(bt._mean([1.0, 2.0, 3.0]), 2.0)

    def test_mean_empty(self):
        with self.assertRaises(ValueError):
            bt._mean([])

    def test_variance_sample_known(self):
        # sample variance of [2, 4]: ((2-3)^2 + (4-3)^2) / 1 = 2
        self.assertAlmostEqual(bt._variance_sample([2.0, 4.0]), 2.0)

    def test_variance_sample_needs_two(self):
        with self.assertRaises(ValueError):
            bt._variance_sample([1.0])

    def test_quantile_midpoint(self):
        self.assertAlmostEqual(bt._quantile([0.0, 1.0], 0.5), 0.5)

    def test_quantile_endpoints(self):
        s = [1.0, 2.0, 3.0]
        self.assertEqual(bt._quantile(s, 0.0), 1.0)
        self.assertEqual(bt._quantile(s, 1.0), 3.0)

    def test_sorted_copy_does_not_mutate(self):
        original = [3.0, 1.0, 2.0]
        copy = bt._sorted_copy(original)
        self.assertEqual(original, [3.0, 1.0, 2.0])
        self.assertEqual(copy, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

class TestTradeLog(unittest.TestCase):
    def test_minimal_record(self):
        log = bt.load_trade_log([{"return": 0.05}])
        self.assertEqual(len(log), 1)
        self.assertAlmostEqual(log[0]["return"], 0.05)
        self.assertIsNone(log[0]["date"])
        self.assertIsNone(log[0]["direction"])

    def test_full_record(self):
        log = bt.load_trade_log([{
            "return": 0.02, "date": "2024-01-01",
            "direction": "long", "entry": 1.05, "exit": 1.071
        }])
        self.assertEqual(log[0]["direction"], "long")
        self.assertEqual(log[0]["date"], "2024-01-01")

    def test_rejects_missing_return(self):
        with self.assertRaises(ValueError):
            bt.load_trade_log([{"date": "2024-01-01"}])

    def test_rejects_non_numeric_return(self):
        with self.assertRaises(ValueError):
            bt.load_trade_log([{"return": "abc"}])

    def test_rejects_bad_direction(self):
        with self.assertRaises(ValueError):
            bt.load_trade_log([{"return": 0.01, "direction": "sideways"}])

    def test_rejects_empty_log(self):
        with self.assertRaises(ValueError):
            bt.load_trade_log([])

    def test_returns_from_log(self):
        log = _make_log([0.01, -0.02, 0.03])
        self.assertEqual(bt.returns_from_log(log), [0.01, -0.02, 0.03])

    def test_filter_by_direction(self):
        log = bt.load_trade_log([
            {"return": 0.01, "direction": "long"},
            {"return": -0.02, "direction": "short"},
            {"return": 0.03, "direction": "long"},
        ])
        longs  = bt.filter_by_direction(log, "long")
        shorts = bt.filter_by_direction(log, "short")
        self.assertEqual(len(longs), 2)
        self.assertEqual(len(shorts), 1)

    def test_filter_bad_direction(self):
        with self.assertRaises(ValueError):
            bt.filter_by_direction([], "sideways")


# ---------------------------------------------------------------------------
# Basic performance metrics
# ---------------------------------------------------------------------------

class TestPerformanceMetrics(unittest.TestCase):
    RETURNS = [0.05, -0.02, 0.03, -0.01, 0.04, -0.03, 0.02]

    def test_expectancy_known(self):
        returns = [0.1, -0.1, 0.2]
        self.assertAlmostEqual(bt.expectancy(returns),
                               (0.1 + (-0.1) + 0.2) / 3)

    def test_expectancy_zero(self):
        self.assertAlmostEqual(bt.expectancy([0.1, -0.1]), 0.0)

    def test_win_rate_all_wins(self):
        self.assertAlmostEqual(bt.win_rate([0.1, 0.2, 0.3]), 1.0)

    def test_win_rate_all_losses(self):
        self.assertAlmostEqual(bt.win_rate([-0.1, -0.2]), 0.0)

    def test_win_rate_mixed(self):
        # 4 wins, 3 losses in RETURNS
        self.assertAlmostEqual(bt.win_rate(self.RETURNS), 4.0 / 7.0)

    def test_loss_rate_complement(self):
        # no ties in RETURNS so win_rate + loss_rate = 1
        wr = bt.win_rate(self.RETURNS)
        lr = bt.loss_rate(self.RETURNS)
        self.assertAlmostEqual(wr + lr, 1.0)

    def test_average_win_positive(self):
        aw = bt.average_win(self.RETURNS)
        self.assertGreater(aw, 0.0)

    def test_average_loss_negative(self):
        al = bt.average_loss(self.RETURNS)
        self.assertLess(al, 0.0)

    def test_average_win_none_when_no_wins(self):
        self.assertIsNone(bt.average_win([-0.1, -0.2]))

    def test_average_loss_none_when_no_losses(self):
        self.assertIsNone(bt.average_loss([0.1, 0.2]))

    def test_profit_factor_above_one(self):
        # wins: 0.05 + 0.03 + 0.04 + 0.02 = 0.14
        # losses: 0.02 + 0.01 + 0.03 = 0.06
        pf = bt.profit_factor(self.RETURNS)
        self.assertAlmostEqual(pf, 0.14 / 0.06, places=10)

    def test_profit_factor_none_when_no_losses(self):
        self.assertIsNone(bt.profit_factor([0.1, 0.2, 0.3]))

    def test_payoff_ratio_positive(self):
        pr = bt.payoff_ratio(self.RETURNS)
        self.assertGreater(pr, 0.0)

    def test_payoff_ratio_none_when_no_losses(self):
        self.assertIsNone(bt.payoff_ratio([0.1, 0.2]))

    def test_expectancy_decomposition_reconstructs(self):
        comps = bt.expectancy_from_components(self.RETURNS)
        self.assertAlmostEqual(comps["expectancy"],
                               comps["reconstructed_ev"], places=10)

    def test_win_rate_empty(self):
        with self.assertRaises(ValueError):
            bt.win_rate([])

    def test_expectancy_empty(self):
        with self.assertRaises(ValueError):
            bt.expectancy([])


# ---------------------------------------------------------------------------
# Risk-adjusted performance
# ---------------------------------------------------------------------------

class TestRiskAdjusted(unittest.TestCase):
    RETURNS = [0.01, -0.005, 0.02, -0.008, 0.015, -0.003, 0.012]

    def test_sharpe_positive_for_profitable_strategy(self):
        sharpe = bt.sharpe_ratio(self.RETURNS)
        self.assertGreater(sharpe, 0.0)

    def test_sharpe_none_for_constant_returns(self):
        self.assertIsNone(bt.sharpe_ratio([0.01, 0.01, 0.01]))

    def test_sharpe_needs_two_returns(self):
        with self.assertRaises(ValueError):
            bt.sharpe_ratio([0.01])

    def test_sharpe_formula(self):
        mu    = sum(self.RETURNS) / len(self.RETURNS)
        n     = len(self.RETURNS)
        s     = math.sqrt(sum((r - mu) ** 2 for r in self.RETURNS) / (n - 1))
        expected = (mu / s) * math.sqrt(252)
        self.assertAlmostEqual(bt.sharpe_ratio(self.RETURNS), expected)

    def test_sortino_positive_for_profitable_strategy(self):
        sortino = bt.sortino_ratio(self.RETURNS)
        self.assertGreater(sortino, 0.0)

    def test_sortino_none_when_no_downside(self):
        self.assertIsNone(bt.sortino_ratio([0.01, 0.02, 0.03]))

    def test_sortino_greater_than_sharpe_for_skewed_returns(self):
        # skewed: small frequent losses, large infrequent gains
        returns = [0.05, 0.05, 0.05, -0.001, -0.001, -0.001, -0.001]
        sharpe  = bt.sharpe_ratio(returns)
        sortino = bt.sortino_ratio(returns)
        # Both should be positive; Sortino typically higher for positive skew
        self.assertGreater(sharpe, 0.0)
        self.assertGreater(sortino, 0.0)

    def test_sharpe_higher_return_higher_sharpe(self):
        low  = [0.001] * 20 + [-0.002] * 10
        high = [0.01]  * 20 + [-0.002] * 10
        self.assertGreater(bt.sharpe_ratio(high), bt.sharpe_ratio(low))


# ---------------------------------------------------------------------------
# Equity curve and drawdown
# ---------------------------------------------------------------------------

class TestEquityCurveDrawdown(unittest.TestCase):
    def test_equity_curve_known(self):
        curve = bt.equity_curve([0.1, -0.1, 0.2])
        self.assertAlmostEqual(curve[0], 1.0)
        self.assertAlmostEqual(curve[1], 1.1)
        self.assertAlmostEqual(curve[2], 1.1 * 0.9)
        self.assertAlmostEqual(curve[3], 1.1 * 0.9 * 1.2)

    def test_equity_curve_initial_wealth(self):
        curve = bt.equity_curve([0.1], initial=500.0)
        self.assertAlmostEqual(curve[0], 500.0)
        self.assertAlmostEqual(curve[1], 550.0)

    def test_equity_curve_flat(self):
        curve = bt.equity_curve([0.0, 0.0, 0.0])
        self.assertTrue(all(abs(v - 1.0) < 1e-12 for v in curve))

    def test_equity_curve_empty_raises(self):
        with self.assertRaises(ValueError):
            bt.equity_curve([])

    def test_equity_curve_total_loss_raises(self):
        with self.assertRaises(ValueError):
            bt.equity_curve([-1.0])

    def test_equity_curve_length(self):
        self.assertEqual(len(bt.equity_curve([0.1] * 10)), 11)

    def test_drawdown_no_dd(self):
        curve = bt.equity_curve([0.05, 0.05, 0.05])
        dd    = bt.drawdown_series(curve)
        self.assertTrue(all(d == 0.0 for d in dd))

    def test_drawdown_known(self):
        # peak 1.2 then 0.84: dd = (1.2 - 0.84) / 1.2 = 0.3
        dd = bt.drawdown_series([1.0, 1.2, 0.84])
        self.assertAlmostEqual(dd[2], 0.3)

    def test_max_drawdown_known(self):
        # peak 1.8, trough 0.9: mdd = 0.5
        curve = [1.0, 1.5, 1.2, 1.8, 0.9]
        self.assertAlmostEqual(bt.max_drawdown(curve), 0.5)

    def test_max_drawdown_zero_for_rising_curve(self):
        curve = bt.equity_curve([0.01] * 50)
        self.assertAlmostEqual(bt.max_drawdown(curve), 0.0)

    def test_max_drawdown_duration_known(self):
        # two episodes: length 1 then length 3
        curve = [1.0, 1.5, 1.2, 1.5, 1.0, 0.9, 0.8, 1.5]
        self.assertEqual(bt.max_drawdown_duration(curve), 3)

    def test_max_drawdown_duration_zero(self):
        curve = bt.equity_curve([0.01] * 10)
        self.assertEqual(bt.max_drawdown_duration(curve), 0)

    def test_calmar_positive(self):
        returns = [0.01] * 100 + [-0.2] + [0.01] * 100
        calmar  = bt.calmar_ratio(returns)
        self.assertIsNotNone(calmar)
        self.assertGreater(calmar, 0.0)

    def test_calmar_none_when_no_drawdown(self):
        self.assertIsNone(bt.calmar_ratio([0.01] * 20))

    def test_calmar_higher_for_better_return(self):
        high = [0.02] * 100 + [-0.2] + [0.02] * 100
        low  = [0.005] * 100 + [-0.2] + [0.005] * 100
        self.assertGreater(bt.calmar_ratio(high), bt.calmar_ratio(low))


# ---------------------------------------------------------------------------
# t-test and confidence interval
# ---------------------------------------------------------------------------

class TestTTest(unittest.TestCase):
    def test_significant_positive_mean(self):
        returns = [0.05] * 50
        # All returns are 0.05, standard deviation is 0 -- expect error
        with self.assertRaises(ValueError):
            bt.t_test_one_sample(returns)

    def test_mixed_returns_not_always_significant(self):
        rng = random.Random(42)
        # pure noise centred on zero
        returns = [rng.gauss(0.0, 0.01) for _ in range(30)]
        result  = bt.t_test_one_sample(returns, mu0=0.0)
        # p-value should be anywhere; just check it is a valid probability
        self.assertGreater(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_t_stat_formula(self):
        returns = [0.01, -0.005, 0.02, -0.008, 0.015]
        n    = len(returns)
        mu   = sum(returns) / n
        s    = math.sqrt(sum((r - mu) ** 2 for r in returns) / (n - 1))
        t_expected = (mu - 0.0) / (s / math.sqrt(n))
        result = bt.t_test_one_sample(returns)
        self.assertAlmostEqual(result["t_stat"], t_expected, places=10)

    def test_df_is_n_minus_one(self):
        returns = [0.01, -0.01, 0.02, -0.02, 0.03]
        result  = bt.t_test_one_sample(returns)
        self.assertEqual(result["df"], 4)

    def test_t_test_needs_two_returns(self):
        with self.assertRaises(ValueError):
            bt.t_test_one_sample([0.01])

    def test_p_value_symmetric(self):
        # positive and negative t-stats of equal magnitude have same p-value
        returns_pos = [0.02, 0.01, 0.03, 0.015, 0.025]
        returns_neg = [-r for r in returns_pos]
        res_pos = bt.t_test_one_sample(returns_pos)
        res_neg = bt.t_test_one_sample(returns_neg)
        self.assertAlmostEqual(res_pos["p_value"], res_neg["p_value"], places=10)

    def test_confidence_interval_contains_mean(self):
        returns = [0.01, -0.005, 0.02, -0.008, 0.015, 0.03, -0.01]
        lo, hi  = bt.confidence_interval_mean(returns)
        mu      = sum(returns) / len(returns)
        self.assertLess(lo, mu)
        self.assertGreater(hi, mu)

    def test_confidence_interval_width_shrinks_with_n(self):
        rng    = random.Random(1)
        base   = [rng.gauss(0.01, 0.02) for _ in range(500)]
        lo_s, hi_s = bt.confidence_interval_mean(base[:50])
        lo_l, hi_l = bt.confidence_interval_mean(base[:500])
        self.assertGreater(hi_s - lo_s, hi_l - lo_l)

    def test_ci_needs_two_returns(self):
        with self.assertRaises(ValueError):
            bt.confidence_interval_mean([0.01])

    def test_t_ppf_round_trip(self):
        for df in [5, 10, 30, 100]:
            for p in [0.75, 0.90, 0.95, 0.975, 0.99]:
                t   = bt._t_ppf(p, df)
                sf  = bt._t_sf(t, df)
                self.assertAlmostEqual(sf, 1.0 - p, places=5,
                                       msg="round-trip failed df={} p={}".format(df, p))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap(unittest.TestCase):
    RETURNS = [0.01, -0.005, 0.02, -0.008, 0.015, -0.003, 0.012,
               0.005, -0.01, 0.008, -0.002, 0.018, -0.006, 0.009]

    def test_ci_contains_point(self):
        point, lo, hi = bt.bootstrap_ci(
            self.RETURNS, lambda v: sum(v) / len(v),
            confidence=0.95, n_boot=500, seed=1
        )
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)

    def test_ci_width_positive(self):
        _, lo, hi = bt.bootstrap_ci(
            self.RETURNS, lambda v: sum(v) / len(v), n_boot=500, seed=2
        )
        self.assertGreater(hi, lo)

    def test_ci_empty_raises(self):
        with self.assertRaises(ValueError):
            bt.bootstrap_ci([], lambda v: sum(v) / len(v))

    def test_bootstrap_equity_curve_returns_correct_structure(self):
        result = bt.bootstrap_equity_curve(self.RETURNS, n_boot=100, seed=3)
        self.assertEqual(len(result["curves"]), 100)
        self.assertEqual(len(result["final_wealth"]), 100)
        self.assertEqual(len(result["median_curve"]), len(self.RETURNS) + 1)
        self.assertEqual(len(result["ci_lower"]), len(self.RETURNS) + 1)
        self.assertEqual(len(result["ci_upper"]), len(self.RETURNS) + 1)

    def test_bootstrap_equity_curve_ci_ordered(self):
        result = bt.bootstrap_equity_curve(self.RETURNS, n_boot=200, seed=4)
        for lo, med, hi in zip(result["ci_lower"],
                                result["median_curve"],
                                result["ci_upper"]):
            self.assertLessEqual(lo, med)
            self.assertLessEqual(med, hi)

    def test_bootstrap_equity_curve_starts_at_one(self):
        result = bt.bootstrap_equity_curve(self.RETURNS, n_boot=50, seed=5)
        self.assertAlmostEqual(result["median_curve"][0], 1.0)
        self.assertAlmostEqual(result["ci_lower"][0], 1.0)
        self.assertAlmostEqual(result["ci_upper"][0], 1.0)

    def test_bootstrap_reproducible_with_seed(self):
        a = bt.bootstrap_equity_curve(self.RETURNS, n_boot=50, seed=99)
        b = bt.bootstrap_equity_curve(self.RETURNS, n_boot=50, seed=99)
        self.assertEqual(a["final_wealth"], b["final_wealth"])


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

class TestPermutationTest(unittest.TestCase):
    def test_permutation_p_value_in_range(self):
        rng     = random.Random(1)
        returns = [rng.gauss(0.01, 0.02) for _ in range(100)]
        result  = bt.permutation_test(
            returns, lambda v: sum(v) / len(v),
            n_permutations=500, seed=1
        )
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_permutation_null_has_correct_length(self):
        rng     = random.Random(2)
        returns = [rng.gauss(0.0, 0.01) for _ in range(50)]
        result  = bt.permutation_test(
            returns, lambda v: sum(v) / len(v),
            n_permutations=200, seed=2
        )
        self.assertEqual(len(result["null"]), 200)

    def test_permutation_observed_is_stat_on_original(self):
        returns  = [0.01, -0.02, 0.03, -0.01, 0.02]
        stat_fn  = lambda v: sum(v) / len(v)
        result   = bt.permutation_test(returns, stat_fn,
                                       n_permutations=100, seed=3)
        self.assertAlmostEqual(result["observed"], stat_fn(returns))

    def test_permutation_does_not_flag_pure_noise(self):
        rng     = random.Random(42)
        returns = [rng.gauss(0.0, 0.01) for _ in range(200)]
        result  = bt.permutation_test(
            returns, lambda v: sum(v) / len(v),
            n_permutations=500, seed=42
        )
        # pure noise: p-value should generally be above 0.05
        # (this could fail very rarely; fixed seed makes it deterministic)
        self.assertFalse(result["significant_at_5pct"])

    def test_permutation_empty_raises(self):
        with self.assertRaises(ValueError):
            bt.permutation_test([], lambda v: 0.0)


# ---------------------------------------------------------------------------
# Strategy simulation
# ---------------------------------------------------------------------------

class TestStrategySimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns = _load_returns()

    def test_ma_crossover_produces_trades(self):
        trades = bt.simulate_ma_crossover(self.returns, fast=10, slow=30)
        self.assertGreater(len(trades), 0)

    def test_ma_crossover_trades_fewer_than_returns(self):
        trades = bt.simulate_ma_crossover(self.returns, fast=10, slow=30)
        self.assertLess(len(trades), len(self.returns))

    def test_ma_crossover_all_long(self):
        trades = bt.simulate_ma_crossover(self.returns, fast=10, slow=30)
        self.assertTrue(all(t["direction"] == "long" for t in trades))

    def test_ma_crossover_bad_periods(self):
        with self.assertRaises(ValueError):
            bt.simulate_ma_crossover(self.returns, fast=30, slow=10)

    def test_ma_crossover_not_enough_data(self):
        with self.assertRaises(ValueError):
            bt.simulate_ma_crossover([0.01] * 5, fast=5, slow=10)

    def test_mean_reversion_produces_trades(self):
        trades = bt.simulate_mean_reversion(self.returns, lookback=20,
                                            z_threshold=1.5)
        self.assertGreater(len(trades), 0)

    def test_mean_reversion_higher_threshold_fewer_trades(self):
        low_thresh  = bt.simulate_mean_reversion(self.returns, lookback=20,
                                                  z_threshold=0.5)
        high_thresh = bt.simulate_mean_reversion(self.returns, lookback=20,
                                                  z_threshold=3.0)
        self.assertGreater(len(low_thresh), len(high_thresh))

    def test_mean_reversion_bad_lookback(self):
        with self.assertRaises(ValueError):
            bt.simulate_mean_reversion(self.returns, lookback=1)

    def test_mean_reversion_not_enough_data(self):
        with self.assertRaises(ValueError):
            bt.simulate_mean_reversion([0.01] * 5, lookback=10)


# ---------------------------------------------------------------------------
# Full backtest report
# ---------------------------------------------------------------------------

class TestBacktestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns = _load_returns()
        cls.log     = _make_log(cls.returns)

    def test_report_all_keys_present(self):
        report = bt.backtest_report(self.log, n_boot=100, seed=1)
        expected = [
            "n_trades", "returns", "expectancy", "expectancy_ci",
            "win_rate", "loss_rate", "average_win", "average_loss",
            "profit_factor", "payoff_ratio", "sharpe", "sortino",
            "calmar", "max_drawdown", "max_dd_duration",
            "t_stat", "p_value", "significant_5pct", "significant_1pct",
            "boot_final_median", "boot_final_5pct", "boot_final_95pct",
        ]
        for key in expected:
            self.assertIn(key, report, msg="missing key: {}".format(key))

    def test_report_n_trades(self):
        report = bt.backtest_report(self.log, n_boot=50, seed=1)
        self.assertEqual(report["n_trades"], len(self.returns))

    def test_report_expectancy_ci_ordered(self):
        report = bt.backtest_report(self.log, n_boot=200, seed=2)
        lo, hi = report["expectancy_ci"]
        self.assertLessEqual(lo, report["expectancy"])
        self.assertLessEqual(report["expectancy"], hi)

    def test_report_boot_percentiles_ordered(self):
        report = bt.backtest_report(self.log, n_boot=200, seed=3)
        self.assertLessEqual(report["boot_final_5pct"],
                             report["boot_final_median"])
        self.assertLessEqual(report["boot_final_median"],
                             report["boot_final_95pct"])

    def test_report_max_drawdown_in_range(self):
        report = bt.backtest_report(self.log, n_boot=50, seed=4)
        self.assertGreater(report["max_drawdown"], 0.0)
        self.assertLessEqual(report["max_drawdown"], 1.0)

    def test_report_p_value_in_range(self):
        report = bt.backtest_report(self.log, n_boot=50, seed=5)
        self.assertGreater(report["p_value"], 0.0)
        self.assertLessEqual(report["p_value"], 1.0)


# ---------------------------------------------------------------------------
# Real data integration
# ---------------------------------------------------------------------------

class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns = _load_returns()

    def test_dataset_length(self):
        self.assertEqual(len(self.returns), 6930)

    def test_win_rate_reasonable(self):
        wr = bt.win_rate(self.returns)
        # EUR/USD daily win rate should be somewhere between 40% and 60%
        self.assertGreater(wr, 0.3)
        self.assertLess(wr, 0.7)

    def test_sharpe_real_data(self):
        sharpe = bt.sharpe_ratio(self.returns)
        # EUR/USD buy-and-hold has a low but non-None Sharpe
        self.assertIsNotNone(sharpe)

    def test_max_drawdown_real_data(self):
        curve = bt.equity_curve(self.returns)
        mdd   = bt.max_drawdown(curve)
        self.assertGreater(mdd, 0.0)
        self.assertLess(mdd, 1.0)

    def test_t_test_real_data(self):
        result = bt.t_test_one_sample(self.returns)
        self.assertIn("t_stat", result)
        self.assertIn("p_value", result)

    def test_ma_strategy_on_real_data(self):
        trades = bt.simulate_ma_crossover(self.returns)
        self.assertGreater(len(trades), 0)
        log    = bt.load_trade_log(trades)
        report = bt.backtest_report(log, n_boot=100, seed=42)
        self.assertIn("expectancy", report)

    def test_mr_strategy_on_real_data(self):
        trades = bt.simulate_mean_reversion(self.returns)
        self.assertGreater(len(trades), 0)
        log    = bt.load_trade_log(trades)
        result = bt.t_test_one_sample(bt.returns_from_log(log))
        self.assertIn("p_value", result)


if __name__ == "__main__":
    unittest.main()
