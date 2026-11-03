import math
import os
import statistics
import unittest

import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestDataLoading(unittest.TestCase):
    def test_loads_expected_count(self):
        data = fx.load_prices_and_returns(DATA_PATH)
        self.assertEqual(len(data["prices"]), 6930)
        self.assertEqual(len(data["returns"]), 6930)


class TestCalibrateVolatility(unittest.TestCase):
    def test_matches_manual_computation(self):
        returns = [0.01, -0.02, 0.015, 0.03, -0.01, 0.02, -0.005]
        expected = statistics.pstdev(returns) * math.sqrt(252)
        self.assertAlmostEqual(fx.calibrate_volatility(returns), expected)

    def test_scales_with_periods_per_year(self):
        returns = [0.01, -0.02, 0.015, 0.03, -0.01]
        vol_252 = fx.calibrate_volatility(returns, periods_per_year=252)
        vol_12 = fx.calibrate_volatility(returns, periods_per_year=12)
        self.assertAlmostEqual(vol_252 / vol_12, math.sqrt(252 / 12), places=6)

    def test_real_eurusd_volatility_is_plausible(self):
        data = fx.load_prices_and_returns(DATA_PATH)
        sigma = fx.calibrate_volatility(data["returns"])
        self.assertGreater(sigma, 0.03)
        self.assertLess(sigma, 0.25)


class TestPriceOptionAllSchemes(unittest.TestCase):
    def test_all_schemes_agree_closely_with_closed_form(self):
        s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
        result = fx.price_option_all_schemes(s0, k, r, sigma, t)
        for scheme in ("explicit", "implicit", "crank_nicolson"):
            self.assertAlmostEqual(result[scheme], result["closed_form"], delta=0.01)

    def test_put_pricing_also_agrees(self):
        s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
        result = fx.price_option_all_schemes(s0, k, r, sigma, t, option_type="put")
        for scheme in ("explicit", "implicit", "crank_nicolson"):
            self.assertAlmostEqual(result[scheme], result["closed_form"], delta=0.01)

    def test_deep_otm_call_is_cheap(self):
        s0, k, r, sigma, t = 50.0, 150.0, 0.05, 0.2, 0.5
        result = fx.price_option_all_schemes(s0, k, r, sigma, t, n_space=200, n_time=500)
        self.assertLess(result["closed_form"], 0.5)
        self.assertLess(result["crank_nicolson"], 1.0)


class TestPDEConvergenceStudy(unittest.TestCase):
    def test_error_generally_decreases_with_finer_spatial_grid(self):
        s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
        results = fx.pde_convergence_study(s0, k, r, sigma, t, [25, 50, 100], n_time=500)
        errors = [r["error"] for r in results]
        self.assertLess(errors[-1], errors[0])

    def test_returns_expected_structure(self):
        s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
        results = fx.pde_convergence_study(s0, k, r, sigma, t, [50, 100], n_time=200)
        self.assertEqual(len(results), 2)
        for r_ in results:
            self.assertIn("n_space", r_)
            self.assertIn("crank_nicolson_price", r_)
            self.assertIn("error", r_)


class TestRealDataApplication(unittest.TestCase):
    def test_prices_a_real_eurusd_atm_option_consistently_across_schemes(self):
        data = fx.load_prices_and_returns(DATA_PATH)
        sigma = fx.calibrate_volatility(data["returns"])
        s0 = data["prices"][-1]
        k = round(s0, 2)
        result = fx.price_option_all_schemes(s0, k, 0.03, sigma, 0.25)
        for scheme in ("explicit", "implicit", "crank_nicolson"):
            self.assertAlmostEqual(result[scheme], result["closed_form"], delta=0.001)

    def test_real_data_option_price_is_positive_and_plausible(self):
        data = fx.load_prices_and_returns(DATA_PATH)
        sigma = fx.calibrate_volatility(data["returns"])
        s0 = data["prices"][-1]
        k = round(s0, 2)
        result = fx.price_option_all_schemes(s0, k, 0.03, sigma, 0.25)
        self.assertGreater(result["closed_form"], 0.0)
        self.assertLess(result["closed_form"], s0 * 0.2)


if __name__ == "__main__":
    unittest.main()
