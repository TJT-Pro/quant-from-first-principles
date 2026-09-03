import random
import unittest

import jacobi_eigen as je
import linear_algebra as la


def random_symmetric_psd(n, seed):
    rng = random.Random(seed)
    m = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
    return la.matmul(m, la.transpose(m))


class TestJacobiEigenKnownCases(unittest.TestCase):
    def test_two_by_two_known_eigenvalues(self):
        a = [[2.0, 1.0], [1.0, 2.0]]
        vals, vecs = je.sorted_eigen(a)
        self.assertAlmostEqual(vals[0], 3.0, places=10)
        self.assertAlmostEqual(vals[1], 1.0, places=10)

    def test_diagonal_matrix_returns_itself(self):
        a = [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]]
        vals, vecs = je.sorted_eigen(a)
        self.assertAlmostEqual(vals[0], 5.0, places=10)
        self.assertAlmostEqual(vals[1], 3.0, places=10)
        self.assertAlmostEqual(vals[2], 1.0, places=10)

    def test_requires_symmetric_input(self):
        with self.assertRaises(ValueError):
            je.jacobi_eigen([[1.0, 2.0], [3.0, 4.0]])


class TestEigenpairCorrectness(unittest.TestCase):
    """Regression tests for a real bug found during development: an
    earlier version of jacobi_eigen produced correct eigenvalues (the
    diagonal converged correctly, matching the trace) but incorrect
    eigenvectors, because the rotation angle's sign did not match the
    sign convention implicit in the row/column update formulas. The
    bug was invisible from the eigenvalues alone and invisible from
    checking V's orthogonality alone (V was a valid, perfectly
    orthogonal rotation product either way); it only showed up when
    checking A*v = lambda*v directly. These tests check exactly that,
    for every matrix size exercised, rather than only checking
    eigenvalues or orthogonality.
    """

    def test_eigenvectors_satisfy_av_equals_lambda_v(self):
        for n in (2, 3, 4, 5, 6):
            a = random_symmetric_psd(n, seed=n)
            vals, vecs = je.jacobi_eigen(a)
            for i in range(n):
                v = la.column(vecs, i)
                av = la.mat_vec(a, v)
                lv = [vals[i] * x for x in v]
                for x, y in zip(av, lv):
                    self.assertAlmostEqual(x, y, places=5, msg=f"n={n}, eigenvector {i}")

    def test_full_reconstruction_a_equals_v_lambda_vt(self):
        for n in (2, 3, 4, 5):
            a = random_symmetric_psd(n, seed=100 + n)
            vals, vecs = je.jacobi_eigen(a)
            d = [[vals[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
            reconstructed = la.matmul(la.matmul(vecs, d), la.transpose(vecs))
            error = la.frobenius_norm(la.subtract(a, reconstructed))
            self.assertLess(error, 1e-6, msg=f"n={n}")

    def test_v_transpose_a_v_is_diagonal(self):
        a = random_symmetric_psd(4, seed=7)
        vals, vecs = je.jacobi_eigen(a)
        vtav = la.matmul(la.matmul(la.transpose(vecs), a), vecs)
        n = len(a)
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.assertAlmostEqual(vtav[i][j], 0.0, places=6)

    def test_eigenvectors_are_orthonormal(self):
        a = random_symmetric_psd(5, seed=9)
        vals, vecs = je.jacobi_eigen(a)
        n = len(a)
        vtv = la.matmul(la.transpose(vecs), vecs)
        for i in range(n):
            for j in range(n):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(vtv[i][j], expected, places=8)


class TestSortedEigen(unittest.TestCase):
    def test_eigenvalues_descending(self):
        a = random_symmetric_psd(5, seed=3)
        vals, vecs = je.sorted_eigen(a)
        for earlier, later in zip(vals, vals[1:]):
            self.assertGreaterEqual(earlier, later)

    def test_sorted_eigenvectors_still_satisfy_av_equals_lambda_v(self):
        a = random_symmetric_psd(4, seed=11)
        vals, vecs = je.sorted_eigen(a)
        for i in range(4):
            v = la.column(vecs, i)
            av = la.mat_vec(a, v)
            lv = [vals[i] * x for x in v]
            for x, y in zip(av, lv):
                self.assertAlmostEqual(x, y, places=5)

    def test_psd_matrix_has_nonnegative_eigenvalues(self):
        a = random_symmetric_psd(6, seed=13)
        vals, vecs = je.sorted_eigen(a)
        for v in vals:
            self.assertGreaterEqual(v, -1e-8)


if __name__ == "__main__":
    unittest.main()
