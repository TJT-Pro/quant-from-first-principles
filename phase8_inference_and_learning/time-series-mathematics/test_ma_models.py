import statistics
import unittest

import ma_models as ma


class TestSimulateMA(unittest.TestCase):
    def test_length(self):
        x = ma.simulate_ma(500, 0.0, [0.5], seed=1)
        self.assertEqual(len(x), 500)

    def test_deterministic_with_seed(self):
        a = ma.simulate_ma(200, 0.0, [0.4], seed=7)
        b = ma.simulate_ma(200, 0.0, [0.4], seed=7)
        self.assertEqual(a, b)

    def test_mean_matches_mu(self):
        x = ma.simulate_ma(20000, 3.0, [0.5, -0.2], seed=1)
        self.assertAlmostEqual(statistics.mean(x), 3.0, delta=0.1)

    def test_theoretical_lag_one_autocorrelation(self):
        # For MA(1), rho_1 = theta / (1 + theta^2).
        theta = 0.6
        x = ma.simulate_ma(30000, 0.0, [theta], seed=2)
        import autocorrelation as ac
        rho1 = ac.autocorrelation(x, 1)
        expected = theta / (1 + theta ** 2)
        self.assertAlmostEqual(rho1, expected, delta=0.03)


class TestIsInvertible(unittest.TestCase):
    def test_ma1_invertible_case(self):
        self.assertTrue(ma.is_invertible([0.5]))
        self.assertTrue(ma.is_invertible([-0.8]))

    def test_ma1_noninvertible_case(self):
        self.assertFalse(ma.is_invertible([1.5]))

    def test_empty_thetas_is_invertible(self):
        self.assertTrue(ma.is_invertible([]))

    def test_ma2_case(self):
        self.assertTrue(ma.is_invertible([0.3, 0.2]))


class TestCSSResiduals(unittest.TestCase):
    def test_zero_theta_gives_demeaned_series(self):
        x = [1.0, 2.0, 3.0, 4.0]
        residuals = ma.css_residuals(x, mu=2.5, thetas=[])
        self.assertEqual(residuals, [-1.5, -0.5, 0.5, 1.5])

    def test_residual_recursion_matches_manual_calculation(self):
        x = [1.0, 2.0, 3.0]
        mu, theta = 0.0, 0.5
        residuals = ma.css_residuals(x, mu, [theta])
        self.assertAlmostEqual(residuals[0], 1.0)
        self.assertAlmostEqual(residuals[1], 1.5)
        self.assertAlmostEqual(residuals[2], 2.25)

    def test_residuals_length_matches_series(self):
        x = list(range(10))
        residuals = ma.css_residuals([float(v) for v in x], mu=0.0, thetas=[0.3, 0.2])
        self.assertEqual(len(residuals), 10)


class TestFitCSS(unittest.TestCase):
    def test_recovers_ma1_coefficient(self):
        true_mu, true_theta = 0.0, 0.6
        x = ma.simulate_ma(5000, true_mu, [true_theta], seed=1)
        result = ma.fit_css(x, q=1)
        self.assertAlmostEqual(result["thetas"][0], true_theta, delta=0.1)
        self.assertAlmostEqual(result["mu"], true_mu, delta=0.1)

    def test_recovers_ma1_negative_coefficient(self):
        true_mu, true_theta = 1.0, -0.5
        x = ma.simulate_ma(5000, true_mu, [true_theta], seed=2)
        result = ma.fit_css(x, q=1)
        self.assertAlmostEqual(result["thetas"][0], true_theta, delta=0.1)

    def test_recovers_ma2_coefficients_approximately(self):
        true_mu, true_thetas = 0.0, [0.4, 0.2]
        x = ma.simulate_ma(8000, true_mu, true_thetas, seed=3)
        result = ma.fit_css(x, q=2)
        self.assertAlmostEqual(result["thetas"][0], true_thetas[0], delta=0.15)
        self.assertAlmostEqual(result["thetas"][1], true_thetas[1], delta=0.15)

    def test_sigma2_close_to_true_variance(self):
        x = ma.simulate_ma(8000, 0.0, [0.5], sigma=1.5, seed=4)
        result = ma.fit_css(x, q=1)
        self.assertAlmostEqual(result["sigma2"], 1.5 ** 2, delta=0.2)


if __name__ == "__main__":
    unittest.main()
