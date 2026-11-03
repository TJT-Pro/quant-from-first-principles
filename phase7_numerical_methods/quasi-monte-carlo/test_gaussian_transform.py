import statistics
import unittest

import gaussian_transform as gt


class TestInverseNormalCDF(unittest.TestCase):
    def test_median_is_zero(self):
        self.assertAlmostEqual(gt.inverse_normal_cdf(0.5), 0.0, places=8)

    def test_matches_python_normaldist_across_bulk(self):
        nd = statistics.NormalDist()
        for i in range(1, 1000):
            u = i / 1000
            self.assertAlmostEqual(gt.inverse_normal_cdf(u), nd.inv_cdf(u), places=6)

    def test_matches_python_normaldist_in_tails(self):
        nd = statistics.NormalDist()
        for u in (1e-6, 1e-4, 0.001, 0.999, 0.9999, 1 - 1e-6):
            self.assertAlmostEqual(gt.inverse_normal_cdf(u), nd.inv_cdf(u), places=5)

    def test_symmetric_around_median(self):
        for u in (0.1, 0.3, 0.45, 0.7, 0.95):
            self.assertAlmostEqual(gt.inverse_normal_cdf(u), -gt.inverse_normal_cdf(1 - u), places=8)

    def test_monotonically_increasing(self):
        values = [gt.inverse_normal_cdf(u) for u in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)]
        for earlier, later in zip(values, values[1:]):
            self.assertLess(earlier, later)

    def test_boundary_values_raise(self):
        with self.assertRaises(ValueError):
            gt.inverse_normal_cdf(0.0)
        with self.assertRaises(ValueError):
            gt.inverse_normal_cdf(1.0)

    def test_transformed_uniform_samples_have_standard_normal_moments(self):
        import random
        rng = random.Random(1)
        samples = [gt.inverse_normal_cdf(rng.random()) for _ in range(50000)]
        self.assertAlmostEqual(statistics.mean(samples), 0.0, delta=0.02)
        self.assertAlmostEqual(statistics.pstdev(samples), 1.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
