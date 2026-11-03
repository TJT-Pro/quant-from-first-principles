import math
import os
import random
import statistics
import unittest

import fx_application as fx
import linear_algebra as la

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestDataLoading(unittest.TestCase):
    def test_loads_expected_count(self):
        returns = fx.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)


class TestBuildVolatilityMatrix(unittest.TestCase):
    def test_shape(self):
        returns = [0.01, -0.02, 0.015, 0.03, -0.01, 0.02, -0.005, 0.01, 0.008, -0.012] * 5
        matrix = fx.build_volatility_matrix(returns, windows=[3, 5])
        self.assertEqual(len(matrix[0]), 2)
        self.assertEqual(len(matrix), len(returns) - 5)

    def test_columns_are_nonnegative(self):
        returns = [0.01, -0.02, 0.015, 0.03, -0.01, 0.02] * 10
        matrix = fx.build_volatility_matrix(returns, windows=[3, 5])
        for row in matrix:
            for v in row:
                self.assertGreaterEqual(v, 0.0)


class TestCenterColumns(unittest.TestCase):
    def test_columns_have_zero_mean(self):
        x = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
        centered = fx.center_columns(x)
        for j in range(2):
            column_mean = statistics.mean([row[j] for row in centered])
            self.assertAlmostEqual(column_mean, 0.0, places=10)


class TestPCAviaCovarianceVsSVD(unittest.TestCase):
    def test_agree_on_well_scaled_synthetic_data(self):
        rng = random.Random(1)
        true_loadings = [0.6, 0.5, 0.45, 0.4]
        norm = la.vector_norm(true_loadings)
        true_loadings = [x / norm for x in true_loadings]
        x = []
        for _ in range(1000):
            common = rng.gauss(0, 1)
            x.append([true_loadings[j] * common * 3 + rng.gauss(0, 0.3) for j in range(4)])

        result = fx.compare_pca_methods(x)
        self.assertLess(result["naive_vs_svd_leading_vector_gap"], 0.01)

    def test_naive_covariance_pca_fails_at_small_realistic_scale(self):
        rng = random.Random(1)
        true_loadings = [0.6, 0.5, 0.45, 0.4]
        norm = la.vector_norm(true_loadings)
        true_loadings = [x / norm for x in true_loadings]
        scale = 1e-5
        x = []
        for _ in range(2000):
            common = rng.gauss(0, 1)
            x.append([scale * (true_loadings[j] * common * 3 + rng.gauss(0, 0.3)) for j in range(4)])

        result = fx.compare_pca_methods(x)
        naive_v0 = la.column(result["naive"]["eigenvectors"], 0)

        naive_error = la.vector_norm([a - b for a, b in zip(naive_v0, true_loadings)])
        self.assertGreater(naive_error, 0.5)

    def test_rescaled_pca_recovers_true_loadings_at_small_scale(self):
        rng = random.Random(1)
        true_loadings = [0.6, 0.5, 0.45, 0.4]
        norm = la.vector_norm(true_loadings)
        true_loadings = [x / norm for x in true_loadings]
        scale = 1e-5
        x = []
        for _ in range(2000):
            common = rng.gauss(0, 1)
            x.append([scale * (true_loadings[j] * common * 3 + rng.gauss(0, 0.3)) for j in range(4)])

        result = fx.compare_pca_methods(x)
        rescaled_v0 = la.column(result["rescaled"]["eigenvectors"], 0)
        if la.dot(rescaled_v0, true_loadings) < 0:
            rescaled_v0 = [-v for v in rescaled_v0]
        rescaled_error = la.vector_norm([a - b for a, b in zip(rescaled_v0, true_loadings)])
        self.assertLess(rescaled_error, 0.05)

    def test_svd_recovers_true_loadings_at_small_scale_without_rescaling(self):
        rng = random.Random(1)
        true_loadings = [0.6, 0.5, 0.45, 0.4]
        norm = la.vector_norm(true_loadings)
        true_loadings = [x / norm for x in true_loadings]
        scale = 1e-5
        x = []
        for _ in range(2000):
            common = rng.gauss(0, 1)
            x.append([scale * (true_loadings[j] * common * 3 + rng.gauss(0, 0.3)) for j in range(4)])

        result = fx.compare_pca_methods(x)
        svd_v0 = la.column(result["svd"]["eigenvectors"], 0)
        if la.dot(svd_v0, true_loadings) < 0:
            svd_v0 = [-v for v in svd_v0]
        svd_error = la.vector_norm([a - b for a, b in zip(svd_v0, true_loadings)])
        self.assertLess(svd_error, 0.05)

    def test_variance_explained_sums_to_one(self):
        rng = random.Random(2)
        x = [[rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(200)]
        result = fx.compare_pca_methods(x)
        self.assertAlmostEqual(sum(result["svd"]["variance_explained"]), 1.0, places=8)
        self.assertAlmostEqual(sum(result["rescaled"]["variance_explained"]), 1.0, places=8)

    def test_rescaled_and_svd_eigenvalues_agree_closely(self):
        rng = random.Random(3)
        x = [[rng.gauss(0, 1), rng.gauss(0, 1) + 0.5 * rng.gauss(0, 1)] for _ in range(500)]
        result = fx.compare_pca_methods(x)
        for a, b in zip(result["rescaled"]["eigenvalues"], result["svd"]["eigenvalues"]):
            self.assertAlmostEqual(a, b, delta=abs(a) * 0.01 + 1e-12)


class TestRealDataApplication(unittest.TestCase):
    def test_first_component_dominates_variance(self):
        returns = fx.load_log_returns(DATA_PATH)
        x = fx.build_volatility_matrix(returns, windows=[5, 10, 20, 60])
        result = fx.compare_pca_methods(x)
        self.assertGreater(result["svd"]["variance_explained"][0], 0.7)

    def test_rescaled_pca_matches_svd_almost_exactly_on_real_data(self):
        returns = fx.load_log_returns(DATA_PATH)
        x = fx.build_volatility_matrix(returns, windows=[5, 10, 20, 60])
        result = fx.compare_pca_methods(x)
        self.assertLess(result["rescaled_vs_svd_leading_vector_gap"], 1e-8)

    def test_naive_pca_shows_a_real_measurable_gap_on_real_data(self):
        returns = fx.load_log_returns(DATA_PATH)
        x = fx.build_volatility_matrix(returns, windows=[5, 10, 20, 60])
        result = fx.compare_pca_methods(x)
        self.assertGreater(result["naive_vs_svd_leading_vector_gap"], 1e-4)


if __name__ == "__main__":
    unittest.main()
