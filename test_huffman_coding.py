import random
import unittest

import huffman_coding as hc
import entropy as ent


class TestHuffmanEncodeDecode(unittest.TestCase):
    def test_decode_recovers_original_sequence(self):
        symbols = list("mississippi")
        encoded, codes = hc.huffman_encode(symbols)
        decoded = hc.huffman_decode(encoded, codes)
        self.assertEqual(decoded, symbols)

    def test_decode_recovers_original_on_random_data(self):
        rng = random.Random(1)
        symbols = [rng.choice("abcdef") for _ in range(2000)]
        encoded, codes = hc.huffman_encode(symbols)
        decoded = hc.huffman_decode(encoded, codes)
        self.assertEqual(decoded, symbols)

    def test_single_symbol_alphabet(self):
        symbols = ["x"] * 50
        encoded, codes = hc.huffman_encode(symbols)
        decoded = hc.huffman_decode(encoded, codes)
        self.assertEqual(decoded, symbols)
        self.assertEqual(len(encoded), 50)  # 1 bit per symbol

    def test_two_symbol_alphabet(self):
        symbols = ["a", "b"] * 100
        encoded, codes = hc.huffman_encode(symbols)
        decoded = hc.huffman_decode(encoded, codes)
        self.assertEqual(decoded, symbols)
        self.assertEqual(len(set(codes.values())), 2)


class TestPrefixFreeProperty(unittest.TestCase):
    def test_no_codeword_is_a_prefix_of_another(self):
        symbols = ["a"] * 45 + ["b"] * 13 + ["c"] * 12 + ["d"] * 16 + ["e"] * 9 + ["f"] * 5
        _, codes = hc.huffman_encode(symbols)
        code_list = list(codes.values())
        for i, c1 in enumerate(code_list):
            for j, c2 in enumerate(code_list):
                if i != j:
                    self.assertFalse(c2.startswith(c1))

    def test_prefix_free_holds_for_skewed_distribution(self):
        symbols = ["a"] * 950 + ["b"] * 30 + ["c"] * 15 + ["d"] * 5
        _, codes = hc.huffman_encode(symbols)
        code_list = list(codes.values())
        for i, c1 in enumerate(code_list):
            for j, c2 in enumerate(code_list):
                if i != j:
                    self.assertFalse(c2.startswith(c1))


class TestSourceCodingBound(unittest.TestCase):
    def test_average_length_at_least_entropy(self):
        rng = random.Random(2)
        symbols = [rng.choice("abcdefgh") for _ in range(5000)]
        result = hc.verify_source_coding_bound(symbols)
        self.assertTrue(result["satisfies_lower_bound"])

    def test_average_length_under_entropy_plus_one(self):
        rng = random.Random(3)
        symbols = [rng.choice("abcdefgh") for _ in range(5000)]
        result = hc.verify_source_coding_bound(symbols)
        self.assertTrue(result["satisfies_upper_bound"])

    def test_bound_holds_across_various_skew_levels(self):
        distributions = [
            ["a"] * 25 + ["b"] * 25 + ["c"] * 25 + ["d"] * 25,
            ["a"] * 70 + ["b"] * 15 + ["c"] * 10 + ["d"] * 5,
            ["a"] * 950 + ["b"] * 30 + ["c"] * 15 + ["d"] * 5,
        ]
        for symbols in distributions:
            result = hc.verify_source_coding_bound(symbols)
            self.assertTrue(result["satisfies_lower_bound"])
            self.assertTrue(result["satisfies_upper_bound"])

    def test_uniform_distribution_achieves_near_optimal_compression(self):
        # For a uniform distribution over a power-of-two alphabet size,
        # Huffman coding achieves exactly the entropy bound (fixed-
        # length codes are already optimal in this special case).
        symbols = list(range(4)) * 1000
        result = hc.verify_source_coding_bound(symbols)
        self.assertAlmostEqual(result["average_code_length"], result["entropy"], places=6)

    def test_highly_skewed_distribution_shows_large_relative_overhead(self):
        # A real, known limitation: when entropy itself is small, the
        # Shannon bound's "+1 bit" slack is large in *relative* terms,
        # even though the absolute gap stays under 1 bit.
        symbols = ["a"] * 950 + ["b"] * 30 + ["c"] * 15 + ["d"] * 5
        result = hc.verify_source_coding_bound(symbols)
        absolute_gap = result["average_code_length"] - result["entropy"]
        self.assertLess(absolute_gap, 1.0)
        relative_overhead = absolute_gap / result["entropy"]
        self.assertGreater(relative_overhead, 1.0)  # more than 100% relative overhead


class TestBuildHuffmanTree(unittest.TestCase):
    def test_codes_have_expected_alphabet(self):
        frequencies = {"a": 5, "b": 3, "c": 2}
        tree = hc.build_huffman_tree(frequencies)
        codes = hc.build_codes(tree)
        self.assertEqual(set(codes.keys()), {"a", "b", "c"})

    def test_more_frequent_symbol_gets_shorter_or_equal_code(self):
        frequencies = {"a": 100, "b": 1}
        tree = hc.build_huffman_tree(frequencies)
        codes = hc.build_codes(tree)
        self.assertLessEqual(len(codes["a"]), len(codes["b"]))


if __name__ == "__main__":
    unittest.main()
