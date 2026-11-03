import math
import os
import unittest

import black_scholes as bs

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestNormalHelpers(unittest.TestCase):
    def test_cdf_at_zero(self):
        self.assertAlmostEqual(bs.standard_normal_cdf(0.0), 0.5)

    def test_pdf_matches_known_value(self):
        # phi(0) = 1/sqrt(2 pi)
        self.assertAlmostEqual(bs.standard_normal_pdf(0.0), 1.0 / math.sqrt(2 * math.pi))

    def test_cdf_symmetry(self):
        for x in (0.3, 1.1, 2.4):
            self.assertAlmostEqual(bs.standard_normal_cdf(-x), 1.0 - bs.standard_normal_cdf(x), places=10)


class TestGBM(unittest.TestCase):
    def test_zero_vol_is_deterministic_growth(self):
        s0, mu, t = 100.0, 0.05, 2.0
        result = bs.simulate_gbm_terminal(s0, mu, 0.0, t, z=0.0)
        self.assertAlmostEqual(result, s0 * math.exp(mu * t))

    def test_z_shifts_terminal_value_correctly(self):
        s0, mu, sigma, t = 100.0, 0.0, 0.2, 1.0
        up = bs.simulate_gbm_terminal(s0, mu, sigma, t, z=1.0)
        down = bs.simulate_gbm_terminal(s0, mu, sigma, t, z=-1.0)
        self.assertGreater(up, s0)
        self.assertLess(down, s0)


class TestD1D2(unittest.TestCase):
    def test_at_the_money_zero_rate_zero_dividend(self):
        s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.0, 0.2
        d1, d2 = bs.d1_d2(s, k, t, r, sigma)
        self.assertAlmostEqual(d1, sigma * math.sqrt(t) / 2, places=8)
        self.assertAlmostEqual(d2, -sigma * math.sqrt(t) / 2, places=8)

    def test_d1_minus_d2_equals_sigma_sqrt_t(self):
        d1, d2 = bs.d1_d2(105.0, 100.0, 0.5, 0.03, 0.25, 0.01)
        self.assertAlmostEqual(d1 - d2, 0.25 * math.sqrt(0.5), places=8)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            bs.d1_d2(0.0, 100.0, 1.0, 0.0, 0.2)
        with self.assertRaises(ValueError):
            bs.d1_d2(100.0, 100.0, 0.0, 0.0, 0.2)


class TestPricingKnownValues(unittest.TestCase):
    def test_textbook_example(self):
        # Hull's standard worked example: S=42, K=40, r=0.10, sigma=0.20,
        # T=0.5, q=0, call ~ 4.76, put ~ 0.81.
        s, k, t, r, sigma = 42.0, 40.0, 0.5, 0.10, 0.20
        self.assertAlmostEqual(bs.call_price(s, k, t, r, sigma), 4.76, places=2)
        self.assertAlmostEqual(bs.put_price(s, k, t, r, sigma), 0.81, places=2)

    def test_call_price_nonnegative_and_above_intrinsic(self):
        s, k, t, r, sigma = 100.0, 90.0, 1.0, 0.02, 0.15
        price = bs.call_price(s, k, t, r, sigma)
        intrinsic = max(s - k * math.exp(-r * t), 0.0)
        self.assertGreaterEqual(price, intrinsic - 1e-9)

    def test_deep_otm_call_near_zero(self):
        price = bs.call_price(50.0, 500.0, 0.1, 0.02, 0.1)
        self.assertLess(price, 1e-6)

    def test_deep_itm_put_near_intrinsic(self):
        s, k, t, r, sigma = 10.0, 500.0, 0.05, 0.02, 0.1
        price = bs.put_price(s, k, t, r, sigma)
        intrinsic = k * math.exp(-r * t) - s
        self.assertAlmostEqual(price, intrinsic, delta=0.5)


class TestPutCallParity(unittest.TestCase):
    def test_parity_holds_across_parameter_grid(self):
        for s in (80.0, 100.0, 120.0):
            for k in (90.0, 100.0, 110.0):
                for t in (0.1, 1.0, 3.0):
                    for r in (0.0, 0.02, 0.08):
                        for sigma in (0.1, 0.3):
                            for q in (0.0, 0.01, 0.05):
                                gap = bs.put_call_parity_gap(s, k, t, r, sigma, q)
                                self.assertAlmostEqual(gap, 0.0, places=8,
                                                        msg=f"s={s},k={k},t={t},r={r},sigma={sigma},q={q}")


class TestGreeksAnalyticalVsNumerical(unittest.TestCase):
    def setUp(self):
        self.params = dict(s=100.0, k=105.0, t=0.75, r=0.03, sigma=0.22, q=0.015)

    def test_call_delta_matches_numerical(self):
        p = self.params
        analytical = bs.call_delta(**p)
        numerical = bs.numerical_delta(bs.call_price, p["s"], p["k"], p["t"], p["r"], p["sigma"], p["q"])
        self.assertAlmostEqual(analytical, numerical, places=5)

    def test_put_delta_matches_numerical(self):
        p = self.params
        analytical = bs.put_delta(**p)
        numerical = bs.numerical_delta(bs.put_price, p["s"], p["k"], p["t"], p["r"], p["sigma"], p["q"])
        self.assertAlmostEqual(analytical, numerical, places=5)

    def test_gamma_matches_numerical_and_shared_by_call_and_put(self):
        p = self.params
        analytical = bs.gamma(**p)
        numerical_call = bs.numerical_gamma(bs.call_price, p["s"], p["k"], p["t"], p["r"], p["sigma"], p["q"])
        numerical_put = bs.numerical_gamma(bs.put_price, p["s"], p["k"], p["t"], p["r"], p["sigma"], p["q"])
        self.assertAlmostEqual(analytical, numerical_call, places=3)
        self.assertAlmostEqual(analytical, numerical_put, places=3)

    def test_vega_matches_numerical_and_shared(self):
        p = self.params
        analytical = bs.vega(**p)
        numerical_call = bs.numerical_vega(bs.call_price, p["s"], p["k"], p["t"], p["r"], p["sigma"], p["q"])
        numerical_put = bs.numerical_vega(bs.put_price, p["s"], p["k"], p["t"], p["r"], p["sigma"], p["q"])
        self.assertAlmostEqual(analytical, numerical_call, places=4)
        self.assertAlmostEqual(analytical, numerical_put, places=4)

    def test_call_theta_matches_numerical(self):
        p = self.params
        analytical = bs.call_theta(**p)
        numerical = bs.numerical_theta(bs.call_price, p["s"], p["k"], p["t"], p["r"], p["sigma"], p["q"])
        self.assertAlmostEqual(analytical, numerical, places=3)

    def test_put_theta_matches_numerical(self):
        p = self.params
        analytical = bs.put_theta(**p)
        numerical = bs.numerical_theta(bs.put_price, p["s"], p["k"], p["t"], p["r"], p["sigma"], p["q"])
        self.assertAlmostEqual(analytical, numerical, places=3)

    def test_delta_put_call_relationship(self):
        p = self.params
        self.assertAlmostEqual(bs.call_delta(**p) - bs.put_delta(**p), math.exp(-p["q"] * p["t"]), places=8)


class TestRho(unittest.TestCase):
    def test_call_rho_domestic_positive(self):
        self.assertGreater(bs.call_rho_domestic(100.0, 100.0, 1.0, 0.03, 0.2), 0.0)

    def test_put_rho_domestic_negative(self):
        self.assertLess(bs.put_rho_domestic(100.0, 100.0, 1.0, 0.03, 0.2), 0.0)

    def test_call_rho_foreign_negative(self):
        self.assertLess(bs.call_rho_foreign(100.0, 100.0, 1.0, 0.03, 0.2, 0.01), 0.0)

    def test_put_rho_foreign_positive(self):
        self.assertGreater(bs.put_rho_foreign(100.0, 100.0, 1.0, 0.03, 0.2, 0.01), 0.0)


class TestFXCalibration(unittest.TestCase):
    def test_load_log_returns_reads_real_data(self):
        returns = bs.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)

    def test_annualized_volatility_reasonable(self):
        returns = bs.load_log_returns(DATA_PATH)
        sigma = bs.calibrate_annualized_volatility(returns)
        self.assertGreater(sigma, 0.03)
        self.assertLess(sigma, 0.20)

    def test_annualized_drift_near_zero(self):
        returns = bs.load_log_returns(DATA_PATH)
        mu = bs.calibrate_annualized_drift(returns)
        self.assertAlmostEqual(mu, 0.0, delta=0.01)

    def test_calibrated_sigma_used_in_pricer_gives_sane_price(self):
        returns = bs.load_log_returns(DATA_PATH)
        sigma = bs.calibrate_annualized_volatility(returns)
        price = bs.call_price(s=1.10, k=1.10, t=0.25, r=0.04, sigma=sigma, q=0.03)
        self.assertGreater(price, 0.0)
        self.assertLess(price, 1.10)


if __name__ == "__main__":
    unittest.main()
