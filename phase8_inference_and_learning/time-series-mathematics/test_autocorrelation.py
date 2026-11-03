import math
import os
import random
import statistics
import unittest

import autocorrelation as ac

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


def simulate_ar1(n: int, phi: float, sigma: float = 1.0, seed: int = 0) -> list:
    rng = random.Random(seed)
    x = [0.0]
    for _ in range(n - 1):
        x.append(phi * x[-1] + rng.gauss(0, sigma))
    return x


def simulate_ma1(n: int, theta: float, sigma: float = 1.0, seed: int = 0) -> list:
    rng = random.Random(seed)
    eps = [rng.gauss(0, sigma) for _ in range(n + 1)]
    return [eps[t + 1] + theta * eps[t] for t in range(n)]


class TestAutocovarianceAndACF(unittest.TestCase):
    def test_lag_zero_is_variance(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean = statistics.mean(x)
        expected = sum((v - mean) ** 2 for v in x) / len(x)
        self.assertAlmostEqual(ac.autocovariance(x, 0), expected)

    def test_acf_at_lag_zero_is_one(self):
        x = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]
        self.assertAlmostEqual(ac.autocorrelation(x, 0), 1.0)

    def test_acf_known_hand_calculation(self):
        x = [1.0, 2.0, 3.0, 4.0]
        mean = 2.5
        gamma0 = sum((v - mean) ** 2 for v in x) / 4
        gamma1 = sum((x[t] - mean) * (x[t + 1] - mean) for t in range(3)) / 4
        self.assertAlmostEqual(ac.autocorrelation(x, 1), gamma1 / gamma0)

    def test_invalid_lag_raises(self):
        with self.assertRaises(ValueError):
            ac.autocovariance([1.0, 2.0], 5)

    def test_acf_list_length(self):
        x = list(range(20))
        result = ac.acf([float(v) for v in x], 5)
        self.assertEqual(len(result), 6)


class TestBartlettAndInverseNormal(unittest.TestCase):
    def test_inverse_normal_cdf_known_quantiles(self):
        self.assertAlmostEqual(ac._inverse_normal_cdf(0.975), 1.959964, places=4)
        self.assertAlmostEqual(ac._inverse_normal_cdf(0.5), 0.0, places=6)
        self.assertAlmostEqual(ac._inverse_normal_cdf(0.025), -1.959964, places=4)

    def test_bartlett_band_matches_known_z(self):
        n = 100
        band = ac.bartlett_confidence_band(n, confidence=0.95)
        self.assertAlmostEqual(band, 1.959964 / math.sqrt(n), places=4)

    def test_band_shrinks_with_more_data(self):
        self.assertGreater(ac.bartlett_confidence_band(50), ac.bartlett_confidence_band(500))


class TestPACFDurbinLevinson(unittest.TestCase):
    def test_ar1_pacf_cuts_off_after_lag_one(self):
        x = simulate_ar1(3000, phi=0.6, seed=1)
        pacf = ac.pacf_durbin_levinson(x, max_lag=8)
        self.assertGreater(abs(pacf[0]), 0.4)
        for k in range(1, 8):
            self.assertLess(abs(pacf[k]), 0.15)

    def test_ma1_acf_cuts_off_after_lag_one(self):
        x = simulate_ma1(3000, theta=0.5, seed=1)
        acf_vals = ac.acf(x, 8)
        # theoretical rho_1 = theta / (1 + theta^2)
        expected_rho1 = 0.5 / (1 + 0.25)
        self.assertAlmostEqual(acf_vals[1], expected_rho1, delta=0.05)
        for k in range(2, 9):
            self.assertLess(abs(acf_vals[k]), 0.08)

    def test_pacf_matches_yule_walker_final_coefficient(self):
        x = simulate_ar1(2000, phi=0.4, seed=2)
        pacf = ac.pacf_durbin_levinson(x, max_lag=5)
        for k in range(1, 6):
            yw_coeffs = ac.yule_walker_coefficients(x, order=k)
            self.assertAlmostEqual(pacf[k - 1], yw_coeffs[-1], places=8)

    def test_zero_max_lag_raises(self):
        with self.assertRaises(ValueError):
            ac.pacf_durbin_levinson([1.0, 2.0, 3.0], max_lag=0)


class TestYuleWalker(unittest.TestCase):
    def test_recovers_ar1_coefficient_approximately(self):
        true_phi = 0.7
        x = simulate_ar1(5000, phi=true_phi, seed=3)
        estimated = ac.yule_walker_coefficients(x, order=1)
        self.assertAlmostEqual(estimated[0], true_phi, delta=0.05)

    def test_recovers_ar2_coefficients_approximately(self):
        rng = random.Random(4)
        phi1, phi2 = 0.5, -0.3
        x = [0.0, 0.0]
        for _ in range(5000):
            x.append(phi1 * x[-1] + phi2 * x[-2] + rng.gauss(0, 1))
        estimated = ac.yule_walker_coefficients(x, order=2)
        self.assertAlmostEqual(estimated[0], phi1, delta=0.05)
        self.assertAlmostEqual(estimated[1], phi2, delta=0.05)

    def test_zero_order_raises(self):
        with self.assertRaises(ValueError):
            ac.yule_walker_coefficients([1.0, 2.0, 3.0], order=0)


class TestChiSquareSurvival(unittest.TestCase):
    def test_known_exponential_case_df2(self):
        for x in (1.0, 3.0, 6.0):
            self.assertAlmostEqual(ac.chi2_sf(x, 2), math.exp(-x / 2), places=8)

    def test_sf_at_zero_is_one(self):
        self.assertEqual(ac.chi2_sf(0.0, 5), 1.0)

    def test_sf_decreasing(self):
        vals = [ac.chi2_sf(x, 4) for x in (1.0, 5.0, 10.0)]
        for a, b in zip(vals, vals[1:]):
            self.assertGreater(a, b)


class TestLjungBox(unittest.TestCase):
    def test_white_noise_not_rejected(self):
        rng = random.Random(5)
        x = [rng.gauss(0, 1) for _ in range(2000)]
        result = ac.ljung_box_test(x, max_lag=10)
        self.assertGreater(result["p_value"], 0.05)

    def test_strongly_autocorrelated_series_rejected(self):
        x = simulate_ar1(2000, phi=0.8, seed=6)
        result = ac.ljung_box_test(x, max_lag=10)
        self.assertLess(result["p_value"], 0.001)

    def test_degrees_of_freedom_adjustment(self):
        x = [1.0] * 50  # irrelevant values; only checking df bookkeeping via a valid case
        rng = random.Random(7)
        x = [rng.gauss(0, 1) for _ in range(500)]
        result = ac.ljung_box_test(x, max_lag=10, n_params_fitted=2)
        self.assertEqual(result["df"], 8)

    def test_insufficient_degrees_of_freedom_raises(self):
        rng = random.Random(8)
        x = [rng.gauss(0, 1) for _ in range(200)]
        with self.assertRaises(ValueError):
            ac.ljung_box_test(x, max_lag=3, n_params_fitted=5)


class TestFXDataLoading(unittest.TestCase):
    def test_load_log_returns_reads_real_data(self):
        returns = ac.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)

    def test_load_prices_reads_real_data(self):
        prices = ac.load_prices(DATA_PATH)
        self.assertEqual(len(prices), 6930)
        self.assertTrue(all(p > 0 for p in prices))


if __name__ == "__main__":
    unittest.main()
