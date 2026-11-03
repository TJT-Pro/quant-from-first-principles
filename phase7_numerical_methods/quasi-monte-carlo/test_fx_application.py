import math
import os
import unittest

import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestDataLoading(unittest.TestCase):
    def test_loads_expected_count(self):
        data = fx.load_prices_and_returns(DATA_PATH)
        self.assertEqual(len(data["prices"]), 6930)
        self.assertEqual(len(data["returns"]), 6930)


class TestCalibrateVolatility(unittest.TestCase):
    def test_real_eurusd_volatility_is_plausible(self):
        data = fx.load_prices_and_returns(DATA_PATH)
        sigma = fx.calibrate_volatility(data["returns"])
        self.assertGreater(sigma, 0.03)
        self.assertLess(sigma, 0.25)


class TestBlackScholesClosedForm(unittest.TestCase):
    def test_positive_and_reasonable(self):
        price = fx.black_scholes_closed_form(100.0, 100.0, 0.05, 0.2, 1.0)
        self.assertGreater(price, 0.0)
        self.assertLess(price, 100.0)

    def test_deep_itm_approaches_intrinsic_plus_carry(self):
        s0, k, r, sigma, t = 1000.0, 100.0, 0.05, 0.2, 1.0
        price = fx.black_scholes_closed_form(s0, k, r, sigma, t)
        expected = s0 - k * math.exp(-r * t)
        self.assertAlmostEqual(price, expected, delta=0.01)


class TestMCAndQMCOptionPricing(unittest.TestCase):
    def test_mc_price_close_to_closed_form_at_large_n(self):
        s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
        exact = fx.black_scholes_closed_form(s0, k, r, sigma, t)
        price = fx.mc_option_price(s0, k, r, sigma, t, n_paths=200000, seed=1)
        self.assertAlmostEqual(price, exact, delta=0.05)

    def test_qmc_price_close_to_closed_form(self):
        s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
        exact = fx.black_scholes_closed_form(s0, k, r, sigma, t)
        price = fx.qmc_option_price(s0, k, r, sigma, t, n_paths=20000)
        self.assertAlmostEqual(price, exact, delta=0.01)

    def test_qmc_is_deterministic(self):
        s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
        p1 = fx.qmc_option_price(s0, k, r, sigma, t, n_paths=5000)
        p2 = fx.qmc_option_price(s0, k, r, sigma, t, n_paths=5000)
        self.assertEqual(p1, p2)

    def test_qmc_more_accurate_than_single_mc_run_at_matched_n(self):
        s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
        exact = fx.black_scholes_closed_form(s0, k, r, sigma, t)
        n = 2000
        mc_price = fx.mc_option_price(s0, k, r, sigma, t, n, seed=1)
        qmc_price = fx.qmc_option_price(s0, k, r, sigma, t, n)
        self.assertLess(abs(qmc_price - exact), abs(mc_price - exact))


class TestConvergenceComparison(unittest.TestCase):
    def test_mc_order_near_half(self):
        result = fx.convergence_comparison(100.0, 100.0, 0.05, 0.2, 1.0,
                                            [200, 800, 3200], n_mc_repeats=15, seed=1)
        self.assertAlmostEqual(result["mc_order"], 0.5, delta=0.2)

    def test_qmc_order_exceeds_mc_order(self):
        result = fx.convergence_comparison(100.0, 100.0, 0.05, 0.2, 1.0,
                                            [200, 800, 3200], n_mc_repeats=15, seed=1)
        self.assertGreater(result["qmc_order"], result["mc_order"])

    def test_qmc_achieves_smaller_absolute_errors_at_every_n(self):
        result = fx.convergence_comparison(100.0, 100.0, 0.05, 0.2, 1.0,
                                            [200, 800, 3200], n_mc_repeats=15, seed=1)
        for mc_err, qmc_err in zip(result["mc_errors"], result["qmc_errors"]):
            self.assertLess(qmc_err, mc_err)


class TestRealDataApplication(unittest.TestCase):
    def test_convergence_comparison_on_real_calibrated_option(self):
        data = fx.load_prices_and_returns(DATA_PATH)
        sigma = fx.calibrate_volatility(data["returns"])
        s0 = data["prices"][-1]
        k = round(s0, 2)
        result = fx.convergence_comparison(s0, k, 0.03, sigma, 0.25, [200, 800, 3200], n_mc_repeats=15, seed=1)
        self.assertGreater(result["qmc_order"], result["mc_order"])
        for mc_err, qmc_err in zip(result["mc_errors"], result["qmc_errors"]):
            self.assertLess(qmc_err, mc_err)

    def test_qmc_kinked_payoff_order_is_lower_than_smooth_integrand_order(self):
        data = fx.load_prices_and_returns(DATA_PATH)
        sigma = fx.calibrate_volatility(data["returns"])
        s0 = data["prices"][-1]
        k = round(s0, 2)
        result = fx.convergence_comparison(s0, k, 0.03, sigma, 0.25, [200, 800, 3200, 12800],
                                            n_mc_repeats=15, seed=1)
        self.assertLess(result["qmc_order"], 0.85)
        self.assertGreater(result["qmc_order"], result["mc_order"])


if __name__ == "__main__":
    unittest.main()
