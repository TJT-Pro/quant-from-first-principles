"""Tests for PCA from scratch.

All expected values are derived analytically or from known small examples.
No numpy or scipy.
"""

import math
import os
import sys
import unittest

import pca as pc

_HERE = os.path.dirname(os.path.abspath(__file__))
_P12  = os.path.join(_HERE, "..", "project12_matrix_engine")
sys.path.insert(0, _P12)
import matrix_engine as mx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allclose(a, b, tol=1e-6):
    if isinstance(a, list) and isinstance(b, list):
        if isinstance(a[0], list):
            return all(allclose(a[i], b[i], tol) for i in range(len(a)))
        return all(abs(x - y) <= tol for x, y in zip(a, b))
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing(unittest.TestCase):
    def test_column_means(self):
        X = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        self.assertAlmostEqual(pc.column_means(X)[0], 2.0)
        self.assertAlmostEqual(pc.column_means(X)[1], 5.0)

    def test_centre_zero_mean(self):
        X   = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        Xc, means = pc.centre(X)
        for j in range(2):
            col_mean = sum(Xc[i][j] for i in range(3)) / 3
            self.assertAlmostEqual(col_mean, 0.0, places=10)

    def test_centre_returns_means(self):
        X = [[1.0, 2.0], [3.0, 4.0]]
        _, means = pc.centre(X)
        self.assertAlmostEqual(means[0], 2.0)
        self.assertAlmostEqual(means[1], 3.0)

    def test_standardise_unit_variance(self):
        X = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        Xs, means, stds = pc.standardise(X)
        for j in range(2):
            var = sum(Xs[i][j]**2 for i in range(3)) / 2   # sample var of centred
            self.assertAlmostEqual(var, 1.0, places=8)

    def test_column_stds(self):
        # x = [2,4,4,4,5,5,7,9]: mean=5, sample std ≈ 2.138
        X   = [[2.0], [4.0], [4.0], [4.0], [5.0], [5.0], [7.0], [9.0]]
        std = pc.column_stds(X)[0]
        import math
        expected = math.sqrt(sum((v[0] - 5.0)**2 for v in X) / 7)
        self.assertAlmostEqual(std, expected, places=8)


# ---------------------------------------------------------------------------
# Covariance matrix
# ---------------------------------------------------------------------------

class TestCovarianceMatrix(unittest.TestCase):
    def test_identity_data(self):
        # Two uncorrelated features, same variance
        X = [[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]]
        Xc, _ = pc.centre(X)
        C = pc.covariance_matrix(Xc)
        self.assertAlmostEqual(C[0][1], 0.0, places=8)

    def test_symmetric(self):
        X  = [[1.0, 2.0], [3.0, 4.0], [5.0, 1.0]]
        Xc, _ = pc.centre(X)
        C  = pc.covariance_matrix(Xc)
        self.assertAlmostEqual(C[0][1], C[1][0], places=12)

    def test_known_covariance(self):
        # x = [1,2,3], y = [1,2,3]: perfect correlation, cov = var = 1
        X  = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        Xc, _ = pc.centre(X)
        C  = pc.covariance_matrix(Xc)
        self.assertAlmostEqual(C[0][0], 1.0, places=8)
        self.assertAlmostEqual(C[0][1], 1.0, places=8)

    def test_single_obs_raises(self):
        with self.assertRaises(ValueError):
            pc.covariance_matrix([[1.0, 2.0]])


# ---------------------------------------------------------------------------
# Jacobi eigenvalue algorithm
# ---------------------------------------------------------------------------

class TestJacobiEigen(unittest.TestCase):
    def test_diagonal_matrix(self):
        C  = [[3.0, 0.0], [0.0, 1.0]]
        ev, V = pc.jacobi_eigen(C)
        self.assertIn(round(3.0, 6), [round(e, 6) for e in ev])
        self.assertIn(round(1.0, 6), [round(e, 6) for e in ev])

    def test_symmetric_2x2(self):
        # [[2,1],[1,2]]: eigenvalues 3 and 1
        C  = [[2.0, 1.0], [1.0, 2.0]]
        ev, V = pc.jacobi_eigen(C)
        self.assertTrue(allclose(sorted(ev), [1.0, 3.0], tol=1e-8))

    def test_eigenvectors_orthonormal(self):
        C  = [[2.0, 1.0], [1.0, 2.0]]
        ev, V = pc.jacobi_eigen(C)
        # VᵀV should be identity
        Vt  = mx.transpose(V)
        VtV = mx.matmul(Vt, V)
        I   = mx.identity(2)
        self.assertTrue(allclose(VtV, I, tol=1e-8))

    def test_cv_equals_lambda_v(self):
        C  = [[2.0, 1.0], [1.0, 2.0]]
        ev, V = pc.jacobi_eigen(C)
        for k in range(2):
            vk  = [V[r][k] for r in range(2)]
            Cvk = mx.matvec(C, vk)
            lv  = mx.vec_scale(ev[k], vk)
            self.assertTrue(allclose(Cvk, lv, tol=1e-7))

    def test_3x3_diagonal(self):
        C  = [[5.0,0,0],[0,3.0,0],[0,0,1.0]]
        ev, V = pc.jacobi_eigen(C)
        self.assertTrue(allclose(sorted(ev), [1.0,3.0,5.0], tol=1e-8))

    def test_3x3_eigenvectors_orthonormal(self):
        C  = [[4.0,1.0,0.0],[1.0,3.0,1.0],[0.0,1.0,2.0]]
        ev, V = pc.jacobi_eigen(C)
        Vt  = mx.transpose(V)
        VtV = mx.matmul(Vt, V)
        I   = mx.identity(3)
        self.assertTrue(allclose(VtV, I, tol=1e-7))

    def test_sum_eigenvalues_equals_trace(self):
        C  = [[4.0,1.0,0.0],[1.0,3.0,1.0],[0.0,1.0,2.0]]
        ev, _ = pc.jacobi_eigen(C)
        self.assertAlmostEqual(sum(ev), mx.trace(C), places=8)


# ---------------------------------------------------------------------------
# Full PCA pipeline
# ---------------------------------------------------------------------------

class TestPCA(unittest.TestCase):
    def _make_two_component_data(self):
        # Data lies along two known directions
        # PC1: [1,1]/sqrt(2) variance=4, PC2: [1,-1]/sqrt(2) variance=1
        import random
        rng = random.Random(42)
        rows = []
        for _ in range(200):
            a = rng.gauss(0, 2)   # component 1 score
            b = rng.gauss(0, 1)   # component 2 score
            rows.append([a + b, a - b])
        return rows

    def test_n_components_in_result(self):
        X = self._make_two_component_data()
        r = pc.pca(X, n_components=1)
        self.assertEqual(r["n_components"], 1)

    def test_scores_shape(self):
        X = self._make_two_component_data()
        r = pc.pca(X, n_components=2)
        self.assertEqual(len(r["scores"]), 200)
        self.assertEqual(len(r["scores"][0]), 2)

    def test_components_shape(self):
        X  = self._make_two_component_data()
        r  = pc.pca(X, n_components=2)
        self.assertEqual(len(r["components"]), 2)    # p=2 features
        self.assertEqual(len(r["components"][0]), 2) # k=2 components

    def test_explained_ratio_sums_to_one(self):
        X = self._make_two_component_data()
        r = pc.pca(X, n_components=2)
        self.assertAlmostEqual(sum(r["explained_ratio"]), 1.0, places=6)

    def test_cumulative_ratio_is_monotone(self):
        X = self._make_two_component_data()
        r = pc.pca(X, n_components=2)
        cum = r["cumulative_ratio"]
        for a, b in zip(cum, cum[1:]):
            self.assertLessEqual(a, b + 1e-10)

    def test_first_component_largest_variance(self):
        X = self._make_two_component_data()
        r = pc.pca(X, n_components=2)
        self.assertGreater(r["eigenvalues"][0], r["eigenvalues"][1])

    def test_scores_zero_mean(self):
        X = self._make_two_component_data()
        r = pc.pca(X, n_components=2)
        n = len(r["scores"])
        for k in range(2):
            mean_k = sum(r["scores"][i][k] for i in range(n)) / n
            self.assertAlmostEqual(mean_k, 0.0, places=6)

    def test_components_orthogonal(self):
        X = self._make_two_component_data()
        r = pc.pca(X, n_components=2)
        c = r["components"]
        # dot(component_0, component_1)
        dot_01 = sum(c[j][0] * c[j][1] for j in range(2))
        self.assertAlmostEqual(dot_01, 0.0, places=6)

    def test_components_unit_length(self):
        X = self._make_two_component_data()
        r = pc.pca(X, n_components=2)
        c = r["components"]
        for k in range(2):
            length = math.sqrt(sum(c[j][k]**2 for j in range(2)))
            self.assertAlmostEqual(length, 1.0, places=6)

    def test_standardise_option(self):
        X = [[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]]
        r = pc.pca(X, standardise_data=True)
        self.assertIsNotNone(r["stds"])

    def test_n_components_defaults_to_p(self):
        X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        r = pc.pca(X)
        self.assertEqual(r["n_components"], 3)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

class TestReconstruction(unittest.TestCase):
    def test_full_reconstruction_exact(self):
        # Using all components: reconstruction should be exact
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]]
        r = pc.pca(X, n_components=2)
        err = pc.reconstruction_error(X, r, n_components=2)
        self.assertAlmostEqual(err, 0.0, places=8)

    def test_partial_reconstruction_error_positive(self):
        import random
        rng = random.Random(0)
        X = [[rng.gauss(0,1) for _ in range(3)] for _ in range(50)]
        r   = pc.pca(X, n_components=3)
        err = pc.reconstruction_error(X, r, n_components=1)
        self.assertGreater(err, 0.0)

    def test_more_components_less_error(self):
        import random
        rng = random.Random(1)
        X   = [[rng.gauss(0,1) for _ in range(4)] for _ in range(50)]
        r   = pc.pca(X, n_components=4)
        e1  = pc.reconstruction_error(X, r, n_components=1)
        e2  = pc.reconstruction_error(X, r, n_components=2)
        self.assertGreater(e1, e2)

    def test_reconstruct_shape(self):
        X    = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        r    = pc.pca(X, n_components=2)
        Xhat = pc.reconstruct(r, n_components=1)
        self.assertEqual(len(Xhat), 3)
        self.assertEqual(len(Xhat[0]), 3)

    def test_too_many_components_raises(self):
        X = [[1.0, 2.0], [3.0, 4.0]]
        r = pc.pca(X, n_components=1)
        with self.assertRaises(ValueError):
            pc.reconstruct(r, n_components=2)


# ---------------------------------------------------------------------------
# Whitening and variance threshold
# ---------------------------------------------------------------------------

class TestWhiteningAndThreshold(unittest.TestCase):
    def test_whitened_unit_variance(self):
        import random
        rng = random.Random(7)
        X   = [[rng.gauss(0,1) + rng.gauss(0,3) for _ in range(2)]
               for _ in range(100)]
        r   = pc.pca(X, n_components=2)
        w   = pc.whitened_scores(r)
        n   = len(w)
        for k in range(2):
            var = sum(w[i][k]**2 for i in range(n)) / (n - 1)
            self.assertAlmostEqual(var, 1.0, places=2)

    def test_n_components_for_variance_full(self):
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]]
        r = pc.pca(X, n_components=2)
        k = pc.n_components_for_variance(r, threshold=0.99)
        self.assertLessEqual(k, 2)

    def test_n_components_for_variance_ordering(self):
        import random
        rng = random.Random(2)
        X   = [[rng.gauss(0,3) + rng.gauss(0,0.1) for _ in range(4)]
               for _ in range(100)]
        r   = pc.pca(X, n_components=4)
        k90 = pc.n_components_for_variance(r, threshold=0.90)
        k99 = pc.n_components_for_variance(r, threshold=0.99)
        self.assertLessEqual(k90, k99)


# ---------------------------------------------------------------------------
# Lag matrix
# ---------------------------------------------------------------------------

class TestLagMatrix(unittest.TestCase):
    def test_shape(self):
        s = list(range(10))
        L = pc.lag_matrix(s, n_lags=3)
        self.assertEqual(len(L), 8)      # 10 - 3 + 1
        self.assertEqual(len(L[0]), 3)

    def test_values(self):
        s = [10, 20, 30, 40, 50]
        L = pc.lag_matrix(s, n_lags=2)
        # Row 0: [20, 10], Row 1: [30, 20], ...
        self.assertEqual(L[0], [20, 10])
        self.assertEqual(L[1], [30, 20])

    def test_lag_1_is_original(self):
        s = [1.0, 2.0, 3.0, 4.0]
        L = pc.lag_matrix(s, n_lags=1)
        self.assertEqual(len(L), 4)
        self.assertEqual([r[0] for r in L], s)

    def test_bad_n_lags(self):
        with self.assertRaises(ValueError):
            pc.lag_matrix([1, 2, 3], n_lags=0)


# ---------------------------------------------------------------------------
# Integration: EUR/USD data
# ---------------------------------------------------------------------------

class TestEURUSDPCA(unittest.TestCase):
    _data = None

    @classmethod
    def setUpClass(cls):
        data_path = os.path.join(os.path.dirname(__file__),
                                  "data", "DEXUSEU_returns.csv")
        if os.path.exists(data_path):
            cls._data = pc.load_csv(data_path)

    def test_pca_on_lag_matrix(self):
        if self._data is None:
            self.skipTest("data file not found")
        X = pc.lag_matrix(self._data, n_lags=5)
        r = pc.pca(X, n_components=3)
        self.assertEqual(r["n_components"], 3)
        self.assertGreater(r["cumulative_ratio"][0], 0.0)
        self.assertLessEqual(r["cumulative_ratio"][-1], 1.0 + 1e-8)

    def test_first_pc_captures_most_variance(self):
        if self._data is None:
            self.skipTest("data file not found")
        X = pc.lag_matrix(self._data, n_lags=5)
        r = pc.pca(X, n_components=5)
        self.assertGreater(r["explained_ratio"][0], r["explained_ratio"][1])

    def test_reconstruction_improves_with_components(self):
        if self._data is None:
            self.skipTest("data file not found")
        X   = pc.lag_matrix(self._data, n_lags=5)
        r   = pc.pca(X, n_components=5)
        e1  = pc.reconstruction_error(X, r, n_components=1)
        e5  = pc.reconstruction_error(X, r, n_components=5)
        self.assertGreater(e1, e5)


if __name__ == "__main__":
    unittest.main()
