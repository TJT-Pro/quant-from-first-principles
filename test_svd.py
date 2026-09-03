import math
import random
import unittest

import svd
import linear_algebra as la


def random_matrix(m, n, seed):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(n)] for _ in range(m)]


class TestSVDReconstruction(unittest.TestCase):
    def test_reconstructs_random_matrix(self):
        a = random_matrix(6, 3, seed=1)
        u, s, v = svd.singular_value_decompose(a)
        recon = svd.reconstruct(u, s, v)
        error = la.frobenius_norm(la.subtract(a, recon))
        self.assertLess(error, 1e-8)

    def test_reconstructs_square_matrix(self):
        a = random_matrix(4, 4, seed=2)
        u, s, v = svd.singular_value_decompose(a)
        recon = svd.reconstruct(u, s, v)
        error = la.frobenius_norm(la.subtract(a, recon))
        self.assertLess(error, 1e-8)

    def test_singular_values_are_nonnegative_and_descending(self):
        a = random_matrix(8, 4, seed=3)
        u, s, v = svd.singular_value_decompose(a)
        for si in s:
            self.assertGreaterEqual(si, 0.0)
        for earlier, later in zip(s, s[1:]):
            self.assertGreaterEqual(earlier, later)

    def test_m_less_than_n_raises(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with self.assertRaises(ValueError):
            svd.singular_value_decompose(a)


class TestOrthogonality(unittest.TestCase):
    def test_u_columns_orthonormal(self):
        a = random_matrix(7, 3, seed=4)
        u, s, v = svd.singular_value_decompose(a)
        utu = la.matmul(la.transpose(u), u)
        n = len(s)
        for i in range(n):
            for j in range(n):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(utu[i][j], expected, places=6)

    def test_v_is_orthogonal(self):
        a = random_matrix(6, 4, seed=5)
        u, s, v = svd.singular_value_decompose(a)
        vtv = la.matmul(la.transpose(v), v)
        n = len(s)
        for i in range(n):
            for j in range(n):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(vtv[i][j], expected, places=8)


class TestRankDeficientMatrix(unittest.TestCase):
    def test_detects_rank_deficiency(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 9.0], [7.0, 8.0, 15.0],
             [1.0, 1.0, 2.0], [0.0, 3.0, 3.0]]
        u, s, v = svd.singular_value_decompose(a)
        self.assertAlmostEqual(s[-1], 0.0, places=6)
        self.assertEqual(svd.rank_estimate(s), 2)

    def test_reconstruction_still_exact_when_rank_deficient(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 9.0], [7.0, 8.0, 15.0],
             [1.0, 1.0, 2.0], [0.0, 3.0, 3.0]]
        u, s, v = svd.singular_value_decompose(a)
        recon = svd.reconstruct(u, s, v)
        error = la.frobenius_norm(la.subtract(a, recon))
        self.assertLess(error, 1e-6)

    def test_gram_schmidt_completed_u_columns_still_orthonormal(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 9.0], [7.0, 8.0, 15.0],
             [1.0, 1.0, 2.0], [0.0, 3.0, 3.0]]
        u, s, v = svd.singular_value_decompose(a)
        utu = la.matmul(la.transpose(u), u)
        n = len(s)
        for i in range(n):
            for j in range(n):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(utu[i][j], expected, places=6)


class TestLowRankApproximation(unittest.TestCase):
    def test_full_rank_approximation_recovers_original(self):
        a = random_matrix(6, 3, seed=6)
        u, s, v = svd.singular_value_decompose(a)
        full = svd.low_rank_approximation(u, s, v, k=3)
        error = la.frobenius_norm(la.subtract(a, full))
        self.assertLess(error, 1e-8)

    def test_zero_rank_approximation_is_zero_matrix(self):
        a = random_matrix(5, 3, seed=7)
        u, s, v = svd.singular_value_decompose(a)
        zero_approx = svd.low_rank_approximation(u, s, v, k=0)
        self.assertAlmostEqual(la.frobenius_norm(zero_approx), 0.0)

    def test_error_matches_dropped_singular_values_eckart_young(self):
        a = random_matrix(6, 4, seed=8)
        u, s, v = svd.singular_value_decompose(a)
        for k in range(len(s)):
            approx = svd.low_rank_approximation(u, s, v, k)
            error = la.frobenius_norm(la.subtract(a, approx))
            expected_error = math.sqrt(sum(si ** 2 for si in s[k:]))
            self.assertAlmostEqual(error, expected_error, places=6)

    def test_truncated_svd_beats_arbitrary_rank_k_alternative(self):
        a = random_matrix(6, 4, seed=9)
        u, s, v = svd.singular_value_decompose(a)
        optimal = svd.low_rank_approximation(u, s, v, k=2)
        optimal_error = la.frobenius_norm(la.subtract(a, optimal))

        u_alt = [[row[1], row[2]] for row in u]
        s_alt = [s[1], s[2]]
        v_alt = [[row[1], row[2]] for row in v]
        alternative = svd.reconstruct(u_alt, s_alt, v_alt)
        alternative_error = la.frobenius_norm(la.subtract(a, alternative))

        self.assertLessEqual(optimal_error, alternative_error)

    def test_invalid_k_raises(self):
        a = random_matrix(5, 3, seed=10)
        u, s, v = svd.singular_value_decompose(a)
        with self.assertRaises(ValueError):
            svd.low_rank_approximation(u, s, v, k=-1)
        with self.assertRaises(ValueError):
            svd.low_rank_approximation(u, s, v, k=10)


class TestPseudoinverse(unittest.TestCase):
    def test_moore_penrose_condition_a_aplus_a_equals_a(self):
        a = random_matrix(6, 3, seed=11)
        u, s, v = svd.singular_value_decompose(a)
        a_plus = svd.pseudoinverse(u, s, v)
        check = la.matmul(la.matmul(a, a_plus), a)
        error = la.frobenius_norm(la.subtract(a, check))
        self.assertLess(error, 1e-6)

    def test_moore_penrose_condition_aplus_a_aplus_equals_aplus(self):
        a = random_matrix(6, 3, seed=12)
        u, s, v = svd.singular_value_decompose(a)
        a_plus = svd.pseudoinverse(u, s, v)
        check = la.matmul(la.matmul(a_plus, a), a_plus)
        error = la.frobenius_norm(la.subtract(a_plus, check))
        self.assertLess(error, 1e-6)

    def test_matches_normal_equations_for_full_rank(self):
        a = random_matrix(6, 3, seed=13)
        u, s, v = svd.singular_value_decompose(a)
        a_plus = svd.pseudoinverse(u, s, v)
        should_be_identity = la.matmul(a_plus, a)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(should_be_identity[i][j], expected, places=5)


class TestRankAndConditionNumber(unittest.TestCase):
    def test_full_rank_matrix(self):
        a = random_matrix(6, 3, seed=14)
        u, s, v = svd.singular_value_decompose(a)
        self.assertEqual(svd.rank_estimate(s), 3)

    def test_condition_number_of_well_conditioned_matrix_is_small(self):
        a = random_matrix(6, 3, seed=15)
        u, s, v = svd.singular_value_decompose(a)
        cond = svd.condition_number(s)
        self.assertGreater(cond, 1.0)
        self.assertLess(cond, 100.0)

    def test_near_duplicate_columns_can_underflow_to_exact_rank_deficiency(self):
        # A real, instructive finding from testing this specific
        # construction: two columns differing by only 1e-8 imply a
        # true smallest singular value on the order of 1e-8 (not
        # exactly zero), but forming A^T*A explicitly squares that
        # gap. The corresponding eigenvalue gap in A^T*A works out to
        # roughly 1.5e-15 against entries of magnitude ~15, right at
        # the edge of double-precision resolution, so the smallest
        # singular value here does not come back as "small," it comes
        # back as exactly 0.0, and the matrix is detected as rank 2,
        # not the ill-conditioned-but-full-rank 3 that a numerically
        # stable direct SVD algorithm (Golub-Kahan bidiagonalisation,
        # which never explicitly forms A^T*A) would report. This is
        # exactly why professional numerical libraries avoid the
        # "eigendecompose A^T*A" approach used throughout this project
        # for anything but well-conditioned, moderately-sized problems.
        a = [[1.0, 1.0 + 1e-8, 2.0], [2.0, 2.0 + 1e-8, 1.0],
             [3.0, 3.0 + 1e-8, 4.0], [1.0, 1.0 + 1e-8, 0.5]]
        u, s, v = svd.singular_value_decompose(a)
        self.assertAlmostEqual(s[-1], 0.0, places=6)
        self.assertEqual(svd.rank_estimate(s, rel_tol=1e-10), 2)

    def test_rank_estimate_of_empty_vector(self):
        self.assertEqual(svd.rank_estimate([]), 0)

    def test_condition_number_undefined_for_rank_zero(self):
        self.assertEqual(svd.condition_number([0.0, 0.0]), float("inf"))


if __name__ == "__main__":
    unittest.main()
