import random
import statistics
import unittest

import parameter_uncertainty as pu


class TestBootstrapResample(unittest.TestCase):
    def test_same_length_as_original(self):
        data = list(range(50))
        resample = pu.bootstrap_resample(data, seed=1)
        self.assertEqual(len(resample), len(data))

    def test_deterministic_given_seed(self):
        data = list(range(50))
        r1 = pu.bootstrap_resample(data, seed=42)
        r2 = pu.bootstrap_resample(data, seed=42)
        self.assertEqual(r1, r2)

    def test_all_values_come_from_original_data(self):
        data = list(range(50))
        resample = pu.bootstrap_resample(data, seed=1)
        self.assertTrue(all(v in data for v in resample))


class TestBootstrapParameterDistribution(unittest.TestCase):
    def test_bootstrap_mean_close_to_true_mean_for_large_sample(self):
        rng = random.Random(1)
        data = [rng.gauss(0.001, 0.01) for _ in range(2000)]
        estimates = pu.bootstrap_parameter_distribution(data, statistics.mean, n_bootstrap=500, seed=1)
        self.assertAlmostEqual(statistics.mean(estimates), statistics.mean(data), delta=0.0005)

    def test_bootstrap_distribution_has_positive_spread(self):
        rng = random.Random(2)
        data = [rng.gauss(0, 0.01) for _ in range(500)]
        estimates = pu.bootstrap_parameter_distribution(data, statistics.pstdev, n_bootstrap=500, seed=1)
        self.assertGreater(statistics.pstdev(estimates), 0.0)

    def test_larger_sample_gives_tighter_bootstrap_distribution(self):
        rng = random.Random(3)
        small_data = [rng.gauss(0, 0.01) for _ in range(100)]
        large_data = [rng.gauss(0, 0.01) for _ in range(5000)]
        small_estimates = pu.bootstrap_parameter_distribution(small_data, statistics.pstdev, 500, seed=1)
        large_estimates = pu.bootstrap_parameter_distribution(large_data, statistics.pstdev, 500, seed=1)
        self.assertGreater(statistics.pstdev(small_estimates), statistics.pstdev(large_estimates))


class TestBlackScholesCall(unittest.TestCase):
    def test_positive_and_reasonable(self):
        price = pu.black_scholes_call(100.0, 100.0, 0.05, 0.2, 1.0)
        self.assertGreater(price, 0.0)
        self.assertLess(price, 100.0)

    def test_zero_volatility_gives_intrinsic_discounted_value(self):
        import math
        s0, k, r, t = 110.0, 100.0, 0.05, 1.0
        price = pu.black_scholes_call(s0, k, r, 0.0, t)
        expected = max(s0 - k * math.exp(-r * t), 0.0)
        self.assertAlmostEqual(price, expected, places=6)

    def test_higher_volatility_gives_higher_price(self):
        price_low_vol = pu.black_scholes_call(100.0, 100.0, 0.05, 0.1, 1.0)
        price_high_vol = pu.black_scholes_call(100.0, 100.0, 0.05, 0.4, 1.0)
        self.assertGreater(price_high_vol, price_low_vol)


class TestPropagateToModelOutput(unittest.TestCase):
    def test_output_point_estimate_matches_direct_computation(self):
        rng = random.Random(1)
        returns = [rng.gauss(0.0005, 0.01) for _ in range(2000)]
        s0, k, r, t = 100.0, 100.0, 0.03, 0.25
        output_fn = lambda sigma: pu.black_scholes_call(s0, k, r, sigma, t)
        result = pu.propagate_to_model_output(returns, pu.annualized_volatility_estimator, output_fn,
                                               n_bootstrap=300, seed=1)
        direct_sigma = pu.annualized_volatility_estimator(returns)
        direct_price = pu.black_scholes_call(s0, k, r, direct_sigma, t)
        self.assertAlmostEqual(result["output_point_estimate"], direct_price, places=8)

    def test_output_uncertainty_is_positive(self):
        rng = random.Random(2)
        returns = [rng.gauss(0.0005, 0.01) for _ in range(2000)]
        output_fn = lambda sigma: pu.black_scholes_call(100.0, 100.0, 0.03, sigma, 0.25)
        result = pu.propagate_to_model_output(returns, pu.annualized_volatility_estimator, output_fn,
                                               n_bootstrap=300, seed=1)
        self.assertGreater(result["output_std"], 0.0)

    def test_smaller_sample_gives_larger_output_uncertainty(self):
        rng = random.Random(3)
        small_returns = [rng.gauss(0.0005, 0.01) for _ in range(200)]
        large_returns = [rng.gauss(0.0005, 0.01) for _ in range(5000)]
        output_fn = lambda sigma: pu.black_scholes_call(100.0, 100.0, 0.03, sigma, 0.25)

        small_result = pu.propagate_to_model_output(small_returns, pu.annualized_volatility_estimator,
                                                      output_fn, n_bootstrap=300, seed=1)
        large_result = pu.propagate_to_model_output(large_returns, pu.annualized_volatility_estimator,
                                                      output_fn, n_bootstrap=300, seed=1)
        self.assertGreater(small_result["output_std"], large_result["output_std"])

    def test_confidence_interval_ordering(self):
        rng = random.Random(4)
        returns = [rng.gauss(0.0005, 0.01) for _ in range(2000)]
        output_fn = lambda sigma: pu.black_scholes_call(100.0, 100.0, 0.03, sigma, 0.25)
        result = pu.propagate_to_model_output(returns, pu.annualized_volatility_estimator, output_fn,
                                               n_bootstrap=500, seed=1)
        self.assertLess(result["output_5th_percentile"], result["output_95th_percentile"])
        self.assertLess(result["parameter_5th_percentile"], result["parameter_95th_percentile"])


if __name__ == "__main__":
    unittest.main()
