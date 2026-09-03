import math
import random
import unittest

import divergence as div
import entropy as ent


class TestKLDivergence(unittest.TestCase):
    def test_zero_for_identical_distributions(self):
        p = {"a": 0.3, "b": 0.7}
        self.assertAlmostEqual(div.kl_divergence(p, p), 0.0)

    def test_nonnegative_gibbs_inequality(self):
        rng = random.Random(1)
        for _ in range(50):
            raw_p = [rng.random() for _ in range(4)]
            raw_q = [rng.random() for _ in range(4)]
            sp, sq = sum(raw_p), sum(raw_q)
            p = {i: v / sp for i, v in enumerate(raw_p)}
            q = {i: v / sq for i, v in enumerate(raw_q)}
            self.assertGreaterEqual(div.kl_divergence(p, q), -1e-10)

    def test_matches_manual_computation(self):
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.9, "b": 0.1}
        expected = 0.5 * math.log2(0.5 / 0.9) + 0.5 * math.log2(0.5 / 0.1)
        self.assertAlmostEqual(div.kl_divergence(p, q), expected)

    def test_infinite_when_q_assigns_zero_to_a_p_supported_outcome(self):
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 1.0, "b": 0.0}
        self.assertEqual(div.kl_divergence(p, q), float("inf"))

    def test_asymmetric_in_general(self):
        p = {"a": 0.9, "b": 0.1}
        q = {"a": 0.5, "b": 0.5}
        self.assertNotAlmostEqual(div.kl_divergence(p, q), div.kl_divergence(q, p), places=4)


class TestCrossEntropy(unittest.TestCase):
    def test_identity_h_pq_equals_h_p_plus_kl(self):
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.9, "b": 0.1}
        h_p = ent.entropy(p)
        d_pq = div.kl_divergence(p, q)
        h_pq = div.cross_entropy(p, q)
        self.assertAlmostEqual(h_pq, h_p + d_pq, places=10)

    def test_equals_entropy_when_q_equals_p(self):
        p = {"a": 0.3, "b": 0.7}
        self.assertAlmostEqual(div.cross_entropy(p, p), ent.entropy(p))

    def test_cross_entropy_at_least_entropy(self):
        # H(P,Q) >= H(P) always, since D(P||Q) >= 0.
        rng = random.Random(2)
        for _ in range(30):
            raw_p = [rng.random() for _ in range(3)]
            raw_q = [rng.random() for _ in range(3)]
            sp, sq = sum(raw_p), sum(raw_q)
            p = {i: v / sp for i, v in enumerate(raw_p)}
            q = {i: v / sq for i, v in enumerate(raw_q)}
            self.assertGreaterEqual(div.cross_entropy(p, q), ent.entropy(p) - 1e-10)


class TestMutualInformation(unittest.TestCase):
    def test_independent_variables_have_near_zero_mi(self):
        rng = random.Random(1)
        pairs = [(rng.choice(["a", "b"]), rng.choice([1, 2, 3])) for _ in range(50000)]
        mi = div.mutual_information_from_pairs(pairs)
        self.assertAlmostEqual(mi, 0.0, delta=0.005)

    def test_deterministic_dependence_gives_mi_equal_to_entropy(self):
        pairs = [("a", 1)] * 100 + [("b", 2)] * 100
        mi = div.mutual_information_from_pairs(pairs)
        h_x = ent.entropy_from_counts([p[0] for p in pairs])
        self.assertAlmostEqual(mi, h_x, places=10)

    def test_entropy_formula_matches_kl_formula(self):
        rng = random.Random(3)
        pairs = [(rng.choice(["a", "b", "c"]), rng.choice([1, 2])) for _ in range(5000)]
        mi_entropy = div.mutual_information_from_pairs(pairs)
        mi_kl = div.mutual_information_via_kl(pairs)
        self.assertAlmostEqual(mi_entropy, mi_kl, places=8)

    def test_mutual_information_is_nonnegative(self):
        rng = random.Random(4)
        for _ in range(20):
            pairs = [(rng.choice(["a", "b"]), rng.choice(["x", "y", "z"])) for _ in range(500)]
            self.assertGreaterEqual(div.mutual_information_from_pairs(pairs), -1e-10)

    def test_detects_nonlinear_dependence_correlation_would_miss(self):
        # Y = X^2 for X uniform on {-2,-1,0,1,2}: Pearson correlation
        # between X and Y is exactly zero (Y is an even function of a
        # symmetric X), but they are strongly dependent, and mutual
        # information should reflect that (unlike correlation, which
        # reports exactly zero here).
        #
        # A precise point worth getting right rather than glossing
        # over: since x^2 is not injective on this symmetric domain
        # (X=-1 and X=1 both map to Y=1; X=-2 and X=2 both map to
        # Y=4), Y is a deterministic function of X (H(Y|X)=0) but X is
        # NOT recoverable from Y (H(X|Y)>0, since Y=1 leaves X's sign
        # ambiguous). So MI = H(Y) - H(Y|X) = H(Y), not H(X); the two
        # differ here specifically because of that sign-collision, and
        # asserting MI=H(X) (only valid when the map is
        # bijective/invertible) would be the wrong claim for this
        # specific, deliberately non-injective example.
        pairs = []
        for x in (-2, -1, 0, 1, 2):
            pairs.append((x, x ** 2))
        mi = div.mutual_information_from_pairs(pairs)
        h_x = ent.entropy_from_counts([p[0] for p in pairs])
        h_y = ent.entropy_from_counts([p[1] for p in pairs])
        self.assertAlmostEqual(mi, h_y, places=10)
        self.assertLess(mi, h_x)
        self.assertGreater(mi, 0.0)


class TestNormalizedMutualInformation(unittest.TestCase):
    def test_independent_variables_give_near_zero(self):
        rng = random.Random(5)
        pairs = [(rng.choice(["a", "b"]), rng.choice([1, 2, 3])) for _ in range(50000)]
        self.assertAlmostEqual(div.normalized_mutual_information(pairs), 0.0, delta=0.01)

    def test_deterministic_dependence_gives_one(self):
        pairs = [("a", 1)] * 100 + [("b", 2)] * 100
        self.assertAlmostEqual(div.normalized_mutual_information(pairs), 1.0, places=8)

    def test_stays_within_zero_one_bounds(self):
        rng = random.Random(6)
        for _ in range(20):
            pairs = [(rng.choice(["a", "b", "c"]), rng.choice(["x", "y"])) for _ in range(500)]
            nmi = div.normalized_mutual_information(pairs)
            self.assertGreaterEqual(nmi, -1e-9)
            self.assertLessEqual(nmi, 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
