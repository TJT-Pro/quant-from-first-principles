import random
import statistics
import unittest

import var_backtesting as vb


class TestInverseNormalCDF(unittest.TestCase):
    def test_known_quantiles(self):
        self.assertAlmostEqual(vb.inverse_normal_cdf(0.5), 0.0, places=6)
        self.assertAlmostEqual(vb.inverse_normal_cdf(0.95), 1.6449, places=3)
        self.assertAlmostEqual(vb.inverse_normal_cdf(0.99), 2.3263, places=3)

    def test_matches_python_normaldist(self):
        nd = statistics.NormalDist()
        for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
            self.assertAlmostEqual(vb.inverse_normal_cdf(p), nd.inv_cdf(p), places=5)


class TestParametricGaussianVaR(unittest.TestCase):
    def test_matches_manual_computation(self):
        returns = [0.01, -0.02, 0.015, -0.01, 0.005, -0.008, 0.02, -0.015]
        mu = statistics.mean(returns)
        sigma = statistics.pstdev(returns)
        z = vb.inverse_normal_cdf(0.05)
        expected = -(mu + z * sigma)
        self.assertAlmostEqual(vb.parametric_gaussian_var(returns, 0.95), expected)

    def test_higher_confidence_gives_larger_var(self):
        rng = random.Random(1)
        returns = [rng.gauss(0, 0.01) for _ in range(2000)]
        var_95 = vb.parametric_gaussian_var(returns, 0.95)
        var_99 = vb.parametric_gaussian_var(returns, 0.99)
        self.assertGreater(var_99, var_95)


class TestHistoricalVaR(unittest.TestCase):
    def test_matches_manual_percentile(self):
        returns = list(range(-100, 100))
        returns = [r / 1000 for r in returns]
        result = vb.historical_var(returns, 0.95)
        sorted_returns = sorted(returns)
        idx = int(0.05 * len(returns))
        self.assertAlmostEqual(result, -sorted_returns[idx])

    def test_matches_gaussian_var_closely_on_genuinely_gaussian_data(self):
        rng = random.Random(1)
        returns = [rng.gauss(0, 0.01) for _ in range(20000)]
        gaussian = vb.parametric_gaussian_var(returns, 0.95)
        historical = vb.historical_var(returns, 0.95)
        self.assertAlmostEqual(gaussian, historical, delta=0.001)


class TestCountExceptions(unittest.TestCase):
    def test_matches_manual_count(self):
        returns = [0.01, -0.05, 0.02, -0.08, -0.01, -0.03]
        var_estimate = 0.04
        # -0.05 < -0.04 and -0.08 < -0.04 both exceed the threshold;
        # -0.01 and -0.03 do not.
        self.assertEqual(vb.count_exceptions(returns, var_estimate), 2)


class TestChiSquared1CDF(unittest.TestCase):
    def test_known_critical_value_for_95_percent(self):
        self.assertAlmostEqual(vb.chi_squared_1_cdf(3.841), 0.95, places=3)

    def test_known_critical_value_for_99_percent(self):
        self.assertAlmostEqual(vb.chi_squared_1_cdf(6.635), 0.99, places=3)

    def test_zero_gives_zero(self):
        self.assertAlmostEqual(vb.chi_squared_1_cdf(0.0), 0.0)

    def test_monotonically_increasing(self):
        values = [vb.chi_squared_1_cdf(x) for x in (0.5, 1.0, 2.0, 5.0, 10.0)]
        for earlier, later in zip(values, values[1:]):
            self.assertLess(earlier, later)


class TestKupiecPOFTest(unittest.TestCase):
    def test_expected_exceptions_matches_formula(self):
        result = vb.kupiec_pof_test(1000, 50, 0.95)
        self.assertAlmostEqual(result["expected_exceptions"], 50.0)

    def test_exact_expected_rate_gives_small_lr_statistic(self):
        result = vb.kupiec_pof_test(1000, 50, 0.95)
        self.assertAlmostEqual(result["lr_statistic"], 0.0, places=6)
        self.assertFalse(result["rejected_at_5_percent"])

    def test_far_too_many_exceptions_is_rejected(self):
        result = vb.kupiec_pof_test(1000, 200, 0.95)
        self.assertTrue(result["rejected_at_5_percent"])
        self.assertLess(result["p_value"], 0.001)

    def test_correctly_specified_model_passes_most_of_the_time(self):
        rng = random.Random(1)
        n, confidence = 1000, 0.95
        p_true = 1 - confidence
        n_rejected = 0
        n_trials = 500
        for _ in range(n_trials):
            exceptions = sum(1 for _ in range(n) if rng.random() < p_true)
            result = vb.kupiec_pof_test(n, exceptions, confidence)
            if result["rejected_at_5_percent"]:
                n_rejected += 1
        rejection_rate = n_rejected / n_trials
        self.assertLess(rejection_rate, 0.12)


if __name__ == "__main__":
    unittest.main()
