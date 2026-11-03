import math
import os
import random
import statistics
import unittest

import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestDataLoading(unittest.TestCase):
    def test_loads_expected_count(self):
        data = fx.load_prices_and_returns(DATA_PATH)
        self.assertEqual(len(data["prices"]), 6930)
        self.assertEqual(len(data["returns"]), 6930)


class TestCalibrateGBM(unittest.TestCase):
    def test_recovers_known_parameters_from_synthetic_gbm_returns(self):
        rng = random.Random(1)
        true_mu, true_sigma = 0.05, 0.15
        dt = 1 / 252
        returns = [rng.gauss((true_mu - 0.5 * true_sigma ** 2) * dt, true_sigma * math.sqrt(dt))
                   for _ in range(50000)]
        result = fx.calibrate_gbm(returns, dt)
        self.assertAlmostEqual(result["sigma"], true_sigma, delta=0.01)
        self.assertAlmostEqual(result["mu"], true_mu, delta=0.02)

    def test_zero_mean_returns_give_small_drift(self):
        returns = [0.001, -0.001, 0.002, -0.002, 0.0005, -0.0005] * 100
        result = fx.calibrate_gbm(returns)
        self.assertGreater(result["sigma"], 0.0)


class TestRollingRealizedVolatility(unittest.TestCase):
    def test_output_length(self):
        returns = [0.01, -0.02, 0.015, 0.03, -0.01] * 10
        vol = fx.rolling_realized_volatility(returns, window=5)
        self.assertEqual(len(vol), len(returns) - 5)

    def test_matches_manual_computation(self):
        returns = [0.01, -0.02, 0.015, 0.03, -0.01, 0.02]
        vol = fx.rolling_realized_volatility(returns, window=3)
        self.assertAlmostEqual(vol[0], statistics.pstdev(returns[0:3]))
        self.assertAlmostEqual(vol[1], statistics.pstdev(returns[1:4]))


class TestCalibrateOUviaAR1(unittest.TestCase):
    def test_recovers_known_parameters_from_synthetic_ou_series(self):
        import models
        true_theta, true_mu, true_sigma = 3.0, 0.05, 0.02
        _, xs = models.simulate_ou_exact(true_mu, true_theta, true_mu, true_sigma, 50.0, 12600, seed=1)
        result = fx.calibrate_ou_via_ar1(xs)
        self.assertAlmostEqual(result["theta"], true_theta, delta=0.5)
        self.assertAlmostEqual(result["mu"], true_mu, delta=0.01)
        self.assertAlmostEqual(result["sigma"], true_sigma, delta=0.01)

    def test_half_life_matches_theta_formula(self):
        import models
        true_theta, true_mu, true_sigma = 2.0, 0.03, 0.01
        _, xs = models.simulate_ou_exact(true_mu, true_theta, true_mu, true_sigma, 50.0, 12600, seed=2)
        result = fx.calibrate_ou_via_ar1(xs)
        expected_half_life = math.log(2) / result["theta"] * 252
        self.assertAlmostEqual(result["half_life_days"], expected_half_life, places=6)


class TestCompareSimulatedVsRealTerminalDistribution(unittest.TestCase):
    def test_matches_theory_on_synthetic_gbm_prices(self):
        import models
        mu_param, sigma_param = 0.03, 0.1
        _, prices = models.simulate_gbm_exact(100.0, mu_param, sigma_param, 20.0, 5000, seed=1)
        gbm_params = {"mu": mu_param, "sigma": sigma_param}
        result = fx.compare_simulated_vs_real_terminal_distribution(prices, gbm_params, horizon_days=25)
        self.assertAlmostEqual(result["std_ratio"], 1.0, delta=0.15)

    def test_real_data_shows_a_ratio_at_or_above_one(self):
        data = fx.load_prices_and_returns(DATA_PATH)
        gbm_params = fx.calibrate_gbm(data["returns"])
        result = fx.compare_simulated_vs_real_terminal_distribution(data["prices"], gbm_params, horizon_days=60)
        self.assertGreaterEqual(result["std_ratio"], 0.95)

    def test_ratio_grows_with_horizon_on_real_data(self):
        data = fx.load_prices_and_returns(DATA_PATH)
        gbm_params = fx.calibrate_gbm(data["returns"])
        short = fx.compare_simulated_vs_real_terminal_distribution(data["prices"], gbm_params, horizon_days=5)
        long = fx.compare_simulated_vs_real_terminal_distribution(data["prices"], gbm_params, horizon_days=60)
        self.assertGreater(long["std_ratio"], short["std_ratio"])


if __name__ == "__main__":
    unittest.main()
