import math
import os
import random
import statistics
import unittest

import black_scholes as bs
import hedging_simulation as hs

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestReconstructPricePath(unittest.TestCase):
    def test_starts_at_s0(self):
        prices = hs.reconstruct_price_path([0.01, -0.02, 0.005], s0=2.0)
        self.assertAlmostEqual(prices[0], 2.0)

    def test_length_is_returns_plus_one(self):
        returns = [0.01, -0.01, 0.02, -0.005]
        prices = hs.reconstruct_price_path(returns)
        self.assertEqual(len(prices), len(returns) + 1)

    def test_log_returns_match_reconstructed_prices(self):
        returns = [0.01, -0.02, 0.03]
        prices = hs.reconstruct_price_path(returns)
        for i, r in enumerate(returns):
            self.assertAlmostEqual(math.log(prices[i + 1] / prices[i]), r, places=10)


class TestAnnualizedRealizedVolatility(unittest.TestCase):
    def test_zero_returns_gives_zero_vol(self):
        self.assertAlmostEqual(hs.annualized_realized_volatility([0.0] * 10), 0.0)

    def test_matches_known_scaling(self):
        returns = [0.01, -0.01] * 50
        vol = hs.annualized_realized_volatility(returns, trading_days=252)
        expected = statistics.pstdev(returns) * math.sqrt(252)
        self.assertAlmostEqual(vol, expected)

    def test_too_few_returns_raises(self):
        with self.assertRaises(ValueError):
            hs.annualized_realized_volatility([0.01])


class TestSimulateDeltaHedgeMechanism(unittest.TestCase):
    def _run_gbm_hedge_trials(self, true_sigma, implied_sigma, horizon_days=30, trials=1500):
        pnls = []
        for i in range(trials):
            rng = random.Random(i)
            dt = 1 / 252
            drift = -0.5 * true_sigma ** 2 * dt
            log_returns = [rng.gauss(drift, true_sigma * math.sqrt(dt)) for _ in range(horizon_days + 2)]
            prices = hs.reconstruct_price_path(log_returns, s0=1.0)
            k = prices[0]
            result = hs.simulate_delta_hedge(prices, 0, horizon_days, k, implied_sigma, 0.0, 0.0, "call")
            pnls.append(result["pnl"])
        return pnls

    def test_short_call_loses_when_realized_vol_exceeds_implied(self):
        pnls = self._run_gbm_hedge_trials(true_sigma=0.35, implied_sigma=0.15)
        self.assertLess(statistics.mean(pnls), 0.0)

    def test_short_call_gains_when_realized_vol_below_implied(self):
        pnls = self._run_gbm_hedge_trials(true_sigma=0.08, implied_sigma=0.20)
        self.assertGreater(statistics.mean(pnls), 0.0)

    def test_pnl_near_zero_when_realized_matches_implied(self):
        pnls = self._run_gbm_hedge_trials(true_sigma=0.20, implied_sigma=0.20, trials=3000)
        self.assertAlmostEqual(statistics.mean(pnls), 0.0, delta=0.01)

    def test_invalid_option_type_raises(self):
        prices = hs.reconstruct_price_path([0.01] * 40)
        with self.assertRaises(ValueError):
            hs.simulate_delta_hedge(prices, 0, 10, prices[0], 0.2, 0.0, 0.0, "straddle")

    def test_insufficient_history_raises(self):
        prices = hs.reconstruct_price_path([0.01] * 5)
        with self.assertRaises(ValueError):
            hs.simulate_delta_hedge(prices, 0, 100, prices[0], 0.2, 0.0, 0.0, "call")

    def test_returns_expected_keys(self):
        prices = hs.reconstruct_price_path([0.001] * 40)
        result = hs.simulate_delta_hedge(prices, 0, 20, prices[0], 0.15, 0.0, 0.0, "call")
        for key in ("pnl", "premium", "realized_vol", "implied_vol"):
            self.assertIn(key, result)


class TestRollingHedgeBacktest(unittest.TestCase):
    def test_number_of_windows(self):
        prices = hs.reconstruct_price_path([0.001] * 300)
        results = hs.rolling_hedge_backtest(prices, horizon_days=30, implied_vol=0.1, r=0.0, q=0.0)
        expected_windows = (len(prices) - 1) // 30
        self.assertEqual(len(results), expected_windows)
        self.assertGreater(len(results), 5)

    def test_windows_are_non_overlapping_by_default(self):
        prices = hs.reconstruct_price_path([0.001] * 200)
        results = hs.rolling_hedge_backtest(prices, horizon_days=20, implied_vol=0.1, r=0.0, q=0.0)
        entries = [r["entry_index"] for r in results]
        gaps = [b - a for a, b in zip(entries, entries[1:])]
        self.assertTrue(all(g == 20 for g in gaps))


class TestFXApplication(unittest.TestCase):
    def test_full_backtest_runs_on_real_data(self):
        log_returns = bs.load_log_returns(DATA_PATH)
        prices = hs.reconstruct_price_path(log_returns)
        implied_vol = bs.calibrate_annualized_volatility(log_returns)
        results = hs.rolling_hedge_backtest(prices, horizon_days=30, implied_vol=implied_vol,
                                             r=0.0, q=0.0, option_type="call")
        self.assertGreater(len(results), 100)

    def test_pnl_correlates_negatively_with_realized_minus_implied_variance(self):
        # The central finding: for a short call, hedging P&L should be
        # negatively correlated with (realized variance - implied
        # variance) actually experienced over the option's life, the
        # discrete-hedging analogue of the continuous-time gamma P&L
        # relationship.
        log_returns = bs.load_log_returns(DATA_PATH)
        prices = hs.reconstruct_price_path(log_returns)
        implied_vol = bs.calibrate_annualized_volatility(log_returns)
        results = hs.rolling_hedge_backtest(prices, horizon_days=30, implied_vol=implied_vol,
                                             r=0.0, q=0.0, option_type="call")
        pnls = [r["pnl"] for r in results]
        var_gaps = [r["realized_vol"] ** 2 - implied_vol ** 2 for r in results]

        n = len(pnls)
        mean_g, mean_p = statistics.mean(var_gaps), statistics.mean(pnls)
        cov = sum((g - mean_g) * (p - mean_p) for g, p in zip(var_gaps, pnls)) / n
        corr = cov / (statistics.pstdev(var_gaps) * statistics.pstdev(pnls))
        self.assertLess(corr, -0.5)


if __name__ == "__main__":
    unittest.main()
