import math
import os
import unittest

import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestDataLoading(unittest.TestCase):
    def test_loads_expected_count(self):
        returns = fx.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)


class TestBandAveragePower(unittest.TestCase):
    def test_matches_manual_average(self):
        power = [1.0, 2.0, 3.0, 4.0, 5.0]
        freqs = [0.0, 0.1, 0.2, 0.3, 0.4]
        result = fx.band_average_power(power, freqs, 0.1, 0.3)
        self.assertAlmostEqual(result, (2.0 + 3.0 + 4.0) / 3)

    def test_empty_band_raises(self):
        power = [1.0, 2.0]
        freqs = [0.0, 0.1]
        with self.assertRaises(ValueError):
            fx.band_average_power(power, freqs, 0.5, 0.6)


class TestLowVsHighFrequencyRatio(unittest.TestCase):
    def test_white_noise_gives_ratio_near_one(self):
        import random
        rng = random.Random(1)
        x = [rng.gauss(0, 1) for _ in range(2048)]
        result = fx.low_vs_high_frequency_ratio(x)
        self.assertAlmostEqual(result["ratio"], 1.0, delta=0.5)

    def test_persistent_ar1_process_gives_ratio_above_one(self):
        import random
        rng = random.Random(2)
        phi = 0.95
        x = [0.0]
        for _ in range(2047):
            x.append(phi * x[-1] + rng.gauss(0, 1))
        result = fx.low_vs_high_frequency_ratio(x)
        self.assertGreater(result["ratio"], 3.0)


class TestWeeklyCyclePower(unittest.TestCase):
    def test_synthetic_weekly_cycle_is_detected(self):
        n = 2048
        x = [math.sin(2 * math.pi * 0.2 * t) + 0.1 * (t % 3) for t in range(n)]
        result = fx.weekly_cycle_power(x, sample_rate=1.0)
        self.assertGreater(result["ratio_to_neighbors"], 5.0)

    def test_white_noise_shows_no_weekly_cycle(self):
        import random
        rng = random.Random(3)
        x = [rng.gauss(0, 1) for _ in range(2048)]
        result = fx.weekly_cycle_power(x)
        self.assertLess(result["ratio_to_neighbors"], 3.0)


class TestExtractVolatilityTrend(unittest.TestCase):
    def test_output_length_matches_input(self):
        x = [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 4.0, 2.0]
        trend = fx.extract_volatility_trend(x, cutoff_period_days=4.0)
        self.assertEqual(len(trend), len(x))

    def test_trend_is_smoother_than_raw_series(self):
        import random
        rng = random.Random(4)
        n = 512
        x = [max(0.0, math.sin(2 * math.pi * t / 100) + 0.5 + rng.gauss(0, 0.3)) for t in range(n)]
        trend = fx.extract_volatility_trend(x, cutoff_period_days=20.0)

        def roughness(series):
            return sum(abs(series[i + 1] - series[i]) for i in range(len(series) - 1))

        self.assertLess(roughness(trend), roughness(x) / 3)

    def test_ideal_filter_can_produce_small_negative_values(self):
        x = [0.0] * 20 + [1.0] + [0.0] * 20
        trend = fx.extract_volatility_trend(x, cutoff_period_days=8.0)
        self.assertLess(min(trend), 0.0)


class TestRealDataApplication(unittest.TestCase):
    def test_squared_returns_show_a_red_spectrum(self):
        returns = fx.load_log_returns(DATA_PATH)
        squared = [r ** 2 for r in returns[:4096]]
        result = fx.low_vs_high_frequency_ratio(squared)
        self.assertGreater(result["ratio"], 2.0)

    def test_raw_returns_show_a_close_to_flat_spectrum(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.low_vs_high_frequency_ratio(returns[:4096])
        self.assertLess(result["ratio"], 1.5)

    def test_no_weekly_cycle_in_real_returns(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.weekly_cycle_power(returns[:4096])
        self.assertLess(result["ratio_to_neighbors"], 3.0)


if __name__ == "__main__":
    unittest.main()
