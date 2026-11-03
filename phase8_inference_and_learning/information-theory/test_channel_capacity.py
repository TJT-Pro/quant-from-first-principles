import math
import unittest

import channel_capacity as cc


class TestBinaryEntropy(unittest.TestCase):
    def test_maximum_at_half(self):
        self.assertAlmostEqual(cc.binary_entropy(0.5), 1.0)

    def test_zero_at_extremes(self):
        self.assertAlmostEqual(cc.binary_entropy(0.0), 0.0)
        self.assertAlmostEqual(cc.binary_entropy(1.0), 0.0)

    def test_symmetric_around_half(self):
        self.assertAlmostEqual(cc.binary_entropy(0.3), cc.binary_entropy(0.7))

    def test_matches_manual_formula(self):
        p = 0.2
        expected = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        self.assertAlmostEqual(cc.binary_entropy(p), expected)


class TestBSCCapacity(unittest.TestCase):
    def test_perfect_channel_has_capacity_one(self):
        self.assertAlmostEqual(cc.bsc_capacity(0.0), 1.0)

    def test_perfectly_invertible_channel_has_capacity_one(self):
        # p=1 always flips the bit, but the flip is perfectly known and
        # invertible, so no information is actually lost.
        self.assertAlmostEqual(cc.bsc_capacity(1.0), 1.0)

    def test_pure_noise_has_zero_capacity(self):
        self.assertAlmostEqual(cc.bsc_capacity(0.5), 0.0)

    def test_capacity_decreases_as_noise_increases_toward_half(self):
        self.assertGreater(cc.bsc_capacity(0.1), cc.bsc_capacity(0.3))
        self.assertGreater(cc.bsc_capacity(0.3), cc.bsc_capacity(0.5))

    def test_symmetric_around_half(self):
        self.assertAlmostEqual(cc.bsc_capacity(0.2), cc.bsc_capacity(0.8))


class TestGaussianChannelCapacity(unittest.TestCase):
    def test_zero_snr_gives_zero_capacity(self):
        self.assertAlmostEqual(cc.gaussian_channel_capacity(0.0), 0.0)

    def test_matches_manual_formula(self):
        snr = 7.0
        self.assertAlmostEqual(cc.gaussian_channel_capacity(snr), 0.5 * math.log2(1 + snr))

    def test_increases_with_snr(self):
        self.assertLess(cc.gaussian_channel_capacity(1.0), cc.gaussian_channel_capacity(10.0))

    def test_known_value_at_snr_15(self):
        # 0.5*log2(16) = 0.5*4 = 2.0 bits exactly.
        self.assertAlmostEqual(cc.gaussian_channel_capacity(15.0), 2.0)


class TestRepetitionCode(unittest.TestCase):
    def test_error_rate_decreases_with_code_length(self):
        results = [cc.simulate_repetition_code(0.1, n, 20000, seed=1) for n in (1, 5, 15)]
        error_rates = [r["empirical_error_rate"] for r in results]
        for earlier, later in zip(error_rates, error_rates[1:]):
            self.assertGreater(earlier, later)

    def test_uncoded_error_rate_matches_crossover_probability(self):
        result = cc.simulate_repetition_code(0.15, 1, 30000, seed=2)
        self.assertAlmostEqual(result["empirical_error_rate"], 0.15, delta=0.01)

    def test_rate_matches_inverse_code_length(self):
        result = cc.simulate_repetition_code(0.1, 7, 1000, seed=3)
        self.assertAlmostEqual(result["rate"], 1 / 7)

    def test_long_code_achieves_near_zero_error_below_capacity(self):
        result = cc.simulate_repetition_code(0.1, 25, 20000, seed=4)
        self.assertLess(result["empirical_error_rate"], 0.001)


if __name__ == "__main__":
    unittest.main()
