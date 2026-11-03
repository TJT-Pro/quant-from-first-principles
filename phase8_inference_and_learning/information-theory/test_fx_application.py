import os
import unittest

import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestDataLoading(unittest.TestCase):
    def test_loads_expected_count(self):
        returns = fx.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)


class TestSignSequence(unittest.TestCase):
    def test_correct_length_and_values(self):
        returns = [0.01, -0.02, 0.0, 0.005]
        signs = fx.sign_sequence(returns)
        self.assertEqual(signs, ["up", "down", "down", "up"])


class TestSignEntropyAnalysis(unittest.TestCase):
    def test_real_data_entropy_close_to_maximum(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.sign_entropy_analysis(returns)
        self.assertGreater(result["fraction_of_max"], 0.99)
        self.assertAlmostEqual(result["max_entropy"], 1.0)

    def test_biased_synthetic_sequence_shows_lower_fraction(self):
        returns = [0.01] * 900 + [-0.01] * 100
        result = fx.sign_entropy_analysis(returns)
        self.assertLess(result["fraction_of_max"], 0.6)


class TestMutualInformationAtLags(unittest.TestCase):
    def test_returns_expected_lag_keys(self):
        seq = ["a", "b"] * 100
        result = fx.mutual_information_at_lags(seq, [1, 2, 3])
        self.assertEqual(set(result.keys()), {1, 2, 3})

    def test_alternating_sequence_shows_strong_lag_one_dependence(self):
        seq = ["a", "b"] * 500
        result = fx.mutual_information_at_lags(seq, [1, 2])
        self.assertGreater(result[1], 0.9)  # perfectly predictable at lag 1


class TestPermutationNullMI(unittest.TestCase):
    def test_genuine_dependence_exceeds_null(self):
        seq = ["a", "b"] * 500
        result = fx.permutation_null_mi(seq, lag=1, n_permutations=200, seed=1)
        self.assertTrue(result["exceeds_95th_percentile"])
        self.assertLess(result["p_value"], 0.05)

    def test_independent_sequence_does_not_exceed_null(self):
        import random
        rng = random.Random(1)
        seq = [rng.choice(["a", "b"]) for _ in range(3000)]
        result = fx.permutation_null_mi(seq, lag=1, n_permutations=200, seed=2)
        self.assertFalse(result["exceeds_95th_percentile"])

    def test_real_sign_sequence_shows_no_significant_lag_one_dependence(self):
        returns = fx.load_log_returns(DATA_PATH)
        signs = fx.sign_sequence(returns)
        result = fx.permutation_null_mi(signs, lag=1, n_permutations=300, seed=1)
        self.assertFalse(result["exceeds_95th_percentile"])

    def test_real_volatility_state_shows_significant_lag_one_dependence(self):
        returns = fx.load_log_returns(DATA_PATH)
        vol_states = fx.rolling_volatility_states(returns, window=20)
        result = fx.permutation_null_mi(vol_states, lag=1, n_permutations=300, seed=1)
        self.assertTrue(result["exceeds_95th_percentile"])
        self.assertLess(result["p_value"], 0.01)


class TestRollingVolatilityStates(unittest.TestCase):
    def test_output_length(self):
        returns = [0.01, -0.02, 0.015, 0.03, -0.01] * 10
        states = fx.rolling_volatility_states(returns, window=5, n_buckets=3)
        self.assertEqual(len(states), len(returns) - 5)

    def test_produces_expected_number_of_buckets(self):
        import random
        rng = random.Random(1)
        returns = [rng.gauss(0, 0.01) for _ in range(500)]
        states = fx.rolling_volatility_states(returns, window=20, n_buckets=4)
        self.assertEqual(len(set(states)), 4)

    def test_buckets_are_roughly_equal_frequency(self):
        import random
        rng = random.Random(1)
        returns = [rng.gauss(0, 0.01) for _ in range(2000)]
        states = fx.rolling_volatility_states(returns, window=20, n_buckets=3)
        from collections import Counter
        counts = Counter(states)
        total = len(states)
        for count in counts.values():
            self.assertAlmostEqual(count / total, 1 / 3, delta=0.05)


class TestHuffmanCompressionAnalysis(unittest.TestCase):
    def test_binary_alphabet_always_achieves_exactly_one_bit_per_symbol(self):
        # The central, precise finding: Huffman coding on any 2-symbol
        # alphabet is mechanically forced to assign one-bit codewords
        # to each symbol (there are only two possible codewords, "0"
        # and "1"), regardless of how skewed the true distribution is,
        # so it cannot discriminate a near-fair sequence from a
        # heavily skewed one the way the entropy computation itself can.
        fair = ["a", "b"] * 500
        skewed = ["a"] * 950 + ["b"] * 50
        fair_result = fx.huffman_compression_analysis(fair)
        skewed_result = fx.huffman_compression_analysis(skewed)
        self.assertAlmostEqual(fair_result["average_code_length"], 1.0)
        self.assertAlmostEqual(skewed_result["average_code_length"], 1.0)
        self.assertLess(skewed_result["entropy"], fair_result["entropy"])

    def test_three_symbol_alphabet_can_beat_naive_two_bit_encoding(self):
        # With 3+ symbols, Huffman genuinely can and does exploit skew.
        skewed = ["a"] * 900 + ["b"] * 80 + ["c"] * 20
        result = fx.huffman_compression_analysis(skewed)
        self.assertLess(result["average_code_length"], 2.0)

    def test_real_volatility_state_compression_close_to_entropy_bound(self):
        returns = fx.load_log_returns(DATA_PATH)
        vol_states = fx.rolling_volatility_states(returns, window=20, n_buckets=3)
        result = fx.huffman_compression_analysis(vol_states)
        self.assertLess(result["average_code_length"] - result["entropy"], 1.0)


class TestKLDivergenceFromGaussian(unittest.TestCase):
    def test_gaussian_samples_show_small_divergence(self):
        import random
        rng = random.Random(1)
        samples = [rng.gauss(0, 1) for _ in range(20000)]
        result = fx.kl_divergence_from_gaussian(samples, n_bins=50)
        self.assertLess(result["kl_divergence_bits"], 0.02)

    def test_real_returns_show_a_real_but_modest_divergence(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.kl_divergence_from_gaussian(returns)
        self.assertGreater(result["kl_divergence_bits"], 0.01)
        self.assertLess(result["kl_divergence_bits"], 0.5)

    def test_heavier_tailed_distribution_shows_larger_divergence(self):
        import random
        rng = random.Random(2)
        gaussian_samples = [rng.gauss(0, 1) for _ in range(20000)]
        # A crude fat-tailed mixture: mostly tight, occasionally wide.
        fat_tailed_samples = [rng.gauss(0, 1) if rng.random() > 0.05 else rng.gauss(0, 5)
                               for _ in range(20000)]
        gaussian_kl = fx.kl_divergence_from_gaussian(gaussian_samples)["kl_divergence_bits"]
        fat_tailed_kl = fx.kl_divergence_from_gaussian(fat_tailed_samples)["kl_divergence_bits"]
        self.assertGreater(fat_tailed_kl, gaussian_kl)


class TestMultiLagSignificanceWithCorrection(unittest.TestCase):
    def test_real_sign_sequence_no_lag_survives_correction(self):
        # The specific real finding this function exists to catch:
        # lag=10 alone looks nominally significant (p~0.03) but should
        # not survive Bonferroni correction across the 5 lags tested.
        returns = fx.load_log_returns(DATA_PATH)
        signs = fx.sign_sequence(returns)
        result = fx.multi_lag_significance_with_correction(signs, [1, 2, 5, 10, 20],
                                                             n_permutations=300, seed=1)
        self.assertFalse(result["any_survives_correction"])
        self.assertAlmostEqual(result["corrected_alpha"], 0.01)

    def test_genuine_strong_dependence_survives_correction(self):
        seq = ["a", "b"] * 500
        result = fx.multi_lag_significance_with_correction(seq, [1, 2, 3], n_permutations=200, seed=1)
        self.assertTrue(result["any_survives_correction"])

    def test_corrected_alpha_matches_bonferroni_formula(self):
        seq = ["a", "b"] * 200
        result = fx.multi_lag_significance_with_correction(seq, [1, 2, 3, 4], n_permutations=100, seed=1)
        self.assertAlmostEqual(result["corrected_alpha"], 0.05 / 4)


if __name__ == "__main__":
    unittest.main()
