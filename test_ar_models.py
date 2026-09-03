import math
import unittest

import ar_models as ar


class TestSimulateAR(unittest.TestCase):
    def test_length(self):
        x = ar.simulate_ar(500, c=0.0, phis=[0.5], seed=1)
        self.assertEqual(len(x), 500)

    def test_deterministic_with_seed(self):
        a = ar.simulate_ar(200, 0.0, [0.5], seed=7)
        b = ar.simulate_ar(200, 0.0, [0.5], seed=7)
        self.assertEqual(a, b)

    def test_mean_matches_theoretical_stationary_mean(self):
        c, phi = 2.0, 0.5
        x = ar.simulate_ar(20000, c, [phi], sigma=1.0, seed=1)
        theoretical_mean = c / (1 - phi)
        self.assertAlmostEqual(sum(x) / len(x), theoretical_mean, delta=0.15)


class TestDurandKernerRoots(unittest.TestCase):
    def test_known_quadratic_roots(self):
        # z^2 - 5z + 6 = 0 has roots 2 and 3.
        roots = ar.durand_kerner_roots([6.0, -5.0, 1.0])
        real_parts = sorted(r.real for r in roots)
        self.assertAlmostEqual(real_parts[0], 2.0, places=6)
        self.assertAlmostEqual(real_parts[1], 3.0, places=6)

    def test_known_complex_roots(self):
        # z^2 + 1 = 0 has roots +-i.
        roots = ar.durand_kerner_roots([1.0, 0.0, 1.0])
        imag_parts = sorted(r.imag for r in roots)
        self.assertAlmostEqual(imag_parts[0], -1.0, places=6)
        self.assertAlmostEqual(imag_parts[1], 1.0, places=6)

    def test_linear_polynomial(self):
        # 2z - 4 = 0 has root 2.
        roots = ar.durand_kerner_roots([-4.0, 2.0])
        self.assertAlmostEqual(roots[0].real, 2.0, places=6)

    def test_cubic_known_roots(self):
        # (z-1)(z-2)(z-3) = z^3 - 6z^2 + 11z - 6
        roots = ar.durand_kerner_roots([-6.0, 11.0, -6.0, 1.0])
        real_parts = sorted(r.real for r in roots)
        for expected, got in zip([1.0, 2.0, 3.0], real_parts):
            self.assertAlmostEqual(expected, got, places=5)

    def test_zero_leading_coefficient_raises(self):
        with self.assertRaises(ValueError):
            ar.durand_kerner_roots([1.0, 2.0, 0.0])


class TestIsStationary(unittest.TestCase):
    def test_ar1_stationary_case(self):
        self.assertTrue(ar.is_stationary([0.5]))
        self.assertTrue(ar.is_stationary([-0.9]))

    def test_ar1_nonstationary_case(self):
        self.assertFalse(ar.is_stationary([1.0]))
        self.assertFalse(ar.is_stationary([1.5]))

    def test_ar2_stationary_case(self):
        # A textbook stationary AR(2) region example.
        self.assertTrue(ar.is_stationary([0.5, -0.3]))

    def test_ar2_nonstationary_case(self):
        # phi1 + phi2 >= 1 violates a necessary stationarity condition.
        self.assertFalse(ar.is_stationary([0.9, 0.5]))

    def test_empty_phis_raises(self):
        with self.assertRaises(ValueError):
            ar.is_stationary([])


class TestFitYuleWalkerAndOLS(unittest.TestCase):
    def test_yule_walker_recovers_ar1(self):
        true_c, true_phi = 1.0, 0.6
        x = ar.simulate_ar(8000, true_c, [true_phi], seed=1)
        result = ar.fit_yule_walker(x, order=1)
        self.assertAlmostEqual(result["phis"][0], true_phi, delta=0.05)
        self.assertAlmostEqual(result["c"], true_c, delta=0.2)

    def test_ols_recovers_ar1(self):
        true_c, true_phi = 1.0, 0.6
        x = ar.simulate_ar(8000, true_c, [true_phi], seed=1)
        result = ar.fit_ols(x, order=1)
        self.assertAlmostEqual(result["phis"][0], true_phi, delta=0.05)
        self.assertAlmostEqual(result["c"], true_c, delta=0.2)

    def test_yule_walker_and_ols_roughly_agree(self):
        x = ar.simulate_ar(5000, 0.5, [0.4, -0.2], seed=2)
        yw = ar.fit_yule_walker(x, order=2)
        ols = ar.fit_ols(x, order=2)
        for i in range(2):
            self.assertAlmostEqual(yw["phis"][i], ols["phis"][i], delta=0.05)

    def test_ols_too_short_series_raises(self):
        with self.assertRaises(ValueError):
            ar.fit_ols([1.0, 2.0], order=5)

    def test_ols_residuals_length(self):
        x = ar.simulate_ar(200, 0.0, [0.3], seed=1)
        result = ar.fit_ols(x, order=1)
        self.assertEqual(len(result["residuals"]), len(x) - 1)


class TestForecast(unittest.TestCase):
    def test_ar1_forecast_converges_to_mean(self):
        c, phi = 1.0, 0.5
        x = ar.simulate_ar(1000, c, [phi], seed=1)
        forecasts = ar.forecast_ar(x, c, [phi], steps=50)
        theoretical_mean = c / (1 - phi)
        self.assertAlmostEqual(forecasts[-1], theoretical_mean, delta=0.01)

    def test_forecast_length(self):
        x = ar.simulate_ar(200, 0.0, [0.3, 0.1], seed=1)
        forecasts = ar.forecast_ar(x, 0.0, [0.3, 0.1], steps=10)
        self.assertEqual(len(forecasts), 10)

    def test_forecast_one_step_matches_formula(self):
        x = [1.0, 2.0, 3.0]
        c, phis = 0.5, [0.2, 0.1]
        forecasts = ar.forecast_ar(x, c, phis, steps=1)
        expected = c + phis[0] * 3.0 + phis[1] * 2.0
        self.assertAlmostEqual(forecasts[0], expected)


if __name__ == "__main__":
    unittest.main()
