import math
import unittest

import entropy as ent


class TestEntropy(unittest.TestCase):
    def test_fair_coin_is_one_bit(self):
        self.assertAlmostEqual(ent.entropy({"H": 0.5, "T": 0.5}), 1.0)

    def test_deterministic_outcome_has_zero_entropy(self):
        self.assertAlmostEqual(ent.entropy({"H": 1.0, "T": 0.0}), 0.0)

    def test_biased_coin_entropy_matches_manual_formula(self):
        p = 0.9
        expected = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        self.assertAlmostEqual(ent.entropy({"H": p, "T": 1 - p}), expected)

    def test_uniform_distribution_achieves_max_entropy(self):
        n = 8
        uniform = {i: 1.0 / n for i in range(n)}
        self.assertAlmostEqual(ent.entropy(uniform), ent.max_entropy(n))

    def test_natural_log_base(self):
        p = {"H": 0.5, "T": 0.5}
        self.assertAlmostEqual(ent.entropy(p, base=math.e), math.log(2))

    def test_zero_probability_does_not_error(self):
        self.assertAlmostEqual(ent.entropy({"a": 1.0, "b": 0.0}), 0.0)


class TestEntropyFromCounts(unittest.TestCase):
    def test_matches_direct_entropy_computation(self):
        counts = ["a", "a", "a", "b", "b", "c"]
        result = ent.entropy_from_counts(counts)
        expected = ent.entropy({"a": 3 / 6, "b": 2 / 6, "c": 1 / 6})
        self.assertAlmostEqual(result, expected)

    def test_all_identical_symbols_has_zero_entropy(self):
        self.assertAlmostEqual(ent.entropy_from_counts(["x"] * 100), 0.0)

    def test_uniform_counts_approach_max_entropy(self):
        counts = [i % 4 for i in range(4000)]
        self.assertAlmostEqual(ent.entropy_from_counts(counts), ent.max_entropy(4), places=6)


class TestJointAndConditionalEntropy(unittest.TestCase):
    def test_chain_rule_holds(self):
        pairs = [("a", 1), ("a", 2), ("b", 1), ("b", 2), ("a", 1), ("b", 2), ("a", 2), ("b", 1)]
        h_xy = ent.joint_entropy_from_pairs(pairs)
        h_x = ent.entropy_from_counts([p[0] for p in pairs])
        h_y_given_x = ent.conditional_entropy(pairs)
        self.assertAlmostEqual(h_xy, h_x + h_y_given_x, places=10)

    def test_conditional_entropy_is_nonnegative(self):
        pairs = [("a", 1), ("a", 1), ("b", 2), ("b", 1), ("a", 2)]
        self.assertGreaterEqual(ent.conditional_entropy(pairs), -1e-10)

    def test_deterministic_y_given_x_has_zero_conditional_entropy(self):
        pairs = [("a", 1), ("a", 1), ("a", 1), ("b", 2), ("b", 2)]
        self.assertAlmostEqual(ent.conditional_entropy(pairs), 0.0, places=10)

    def test_independent_x_y_gives_conditional_entropy_equal_to_marginal(self):
        import random
        rng = random.Random(1)
        pairs = [(rng.choice(["a", "b"]), rng.choice([1, 2, 3])) for _ in range(20000)]
        h_y = ent.entropy_from_counts([p[1] for p in pairs])
        h_y_given_x = ent.conditional_entropy(pairs)
        self.assertAlmostEqual(h_y_given_x, h_y, delta=0.02)


class TestMaxEntropy(unittest.TestCase):
    def test_matches_log_n(self):
        for n in (2, 3, 5, 16):
            self.assertAlmostEqual(ent.max_entropy(n), math.log2(n))

    def test_increases_with_alphabet_size(self):
        self.assertLess(ent.max_entropy(2), ent.max_entropy(4))
        self.assertLess(ent.max_entropy(4), ent.max_entropy(8))


if __name__ == "__main__":
    unittest.main()
