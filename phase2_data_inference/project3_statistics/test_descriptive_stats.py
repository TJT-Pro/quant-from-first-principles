"""Tests for Project 3 - Descriptive Statistics Library.

Run with:  python3 -m unittest -v
"""

import math
import os
import unittest

import descriptive_stats as ds


class TestLocation(unittest.TestCase):

    def test_mean(self):
        self.assertEqual(ds.mean([1, 2, 3, 4]), 2.5)
        self.assertEqual(ds.mean([5]), 5.0)
        self.assertEqual(ds.mean([1.5, 2.5]), 2.0)

    def test_mean_empty_raises(self):
        with self.assertRaises(ValueError):
            ds.mean([])

    def test_median_odd(self):
        self.assertEqual(ds.median([3, 1, 2]), 2.0)

    def test_median_even(self):
        self.assertEqual(ds.median([1, 2, 3, 4]), 2.5)

    def test_median_single(self):
        self.assertEqual(ds.median([7]), 7.0)

    def test_median_empty_raises(self):
        with self.assertRaises(ValueError):
            ds.median([])

    def test_median_matches_mean_for_symmetric(self):
        self.assertEqual(ds.median([2, 4, 6, 8]), 5.0)

    def test_mode_single(self):
        self.assertEqual(ds.mode([1, 2, 2, 3]), [2.0])

    def test_mode_multiple(self):
        self.assertEqual(ds.mode([1, 1, 2, 2, 3]), [1.0, 2.0])

    def test_mode_all_unique_returns_all(self):
        modes = ds.mode([1, 2, 3])
        self.assertEqual(set(modes), {1.0, 2.0, 3.0})

    def test_mode_first_appearance_order(self):
        self.assertEqual(ds.mode([4, 4, 1, 1, 2]), [4.0, 1.0])

    def test_mode_empty_raises(self):
        with self.assertRaises(ValueError):
            ds.mode([])

    def test_data_range(self):
        self.assertEqual(ds.data_range([4, 1, 9]), 8.0)
        self.assertEqual(ds.data_range([5]), 0.0)

    def test_data_range_empty_raises(self):
        with self.assertRaises(ValueError):
            ds.data_range([])


class TestQuantiles(unittest.TestCase):

    def test_quantile_endpoints(self):
        self.assertEqual(ds.quantile([1, 2, 3, 4], 0.0), 1.0)
        self.assertEqual(ds.quantile([1, 2, 3, 4], 1.0), 4.0)

    def test_quantile_median_even(self):
        self.assertEqual(ds.quantile([1, 2, 3, 4], 0.5), 2.5)

    def test_quantile_linear_interpolation(self):
        self.assertEqual(ds.quantile([1, 2, 3, 4], 0.25), 1.75)
        self.assertEqual(ds.quantile([1, 2, 3, 4], 0.75), 3.25)

    def test_quantile_single(self):
        self.assertEqual(ds.quantile([8], 0.5), 8.0)

    def test_quantile_invalid_q(self):
        with self.assertRaises(ValueError):
            ds.quantile([1, 2], -0.1)
        with self.assertRaises(ValueError):
            ds.quantile([1, 2], 1.1)

    def test_quantile_empty_raises(self):
        with self.assertRaises(ValueError):
            ds.quantile([], 0.5)

    def test_iqr(self):
        self.assertEqual(ds.iqr([1, 2, 3, 4, 5, 6]), 2.5)


class TestSpread(unittest.TestCase):

    def test_variance_population(self):
        self.assertAlmostEqual(ds.variance([1, 2, 3, 4], False), 1.25, places=12)

    def test_variance_sample(self):
        self.assertAlmostEqual(ds.variance([1, 2, 3, 4], True), 5.0 / 3.0, places=12)

    def test_variance_constant(self):
        self.assertEqual(ds.variance([3, 3, 3], False), 0.0)
        self.assertEqual(ds.variance([3, 3, 3], True), 0.0)

    def test_variance_single(self):
        self.assertEqual(ds.variance([3], False), 0.0)
        with self.assertRaises(ValueError):
            ds.variance([3], True)

    def test_variance_empty_raises(self):
        with self.assertRaises(ValueError):
            ds.variance([])

    def test_stddev_is_sqrt_of_variance(self):
        data = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(ds.stddev(data, True),
                               math.sqrt(ds.variance(data, True)), places=12)
        self.assertAlmostEqual(ds.stddev(data, False),
                               math.sqrt(ds.variance(data, False)), places=12)

    def test_sample_variance_consistent_with_known(self):
        # population vs sample: sample should be larger for n < inf
        data = [1.1, 2.2, 3.3, 4.4, 5.5]
        self.assertGreater(ds.variance(data, True), ds.variance(data, False))


class TestShape(unittest.TestCase):

    def test_skewness_symmetric_is_zero(self):
        self.assertAlmostEqual(ds.skewness([1, 2, 3]), 0.0, places=12)
        self.assertAlmostEqual(ds.skewness([1, 2, 3, 4, 5]), 0.0, places=12)

    def test_skewness_positive_for_right_tail(self):
        self.assertGreater(ds.skewness([1, 1, 1, 2, 10]), 0.0)

    def test_skewness_negative_for_left_tail(self):
        self.assertLess(ds.skewness([10, 10, 10, 10, 1]), 0.0)

    def test_skewness_unbiased_needs_three(self):
        with self.assertRaises(ValueError):
            ds.skewness([1, 2], False)

    def test_skewness_zero_variance_raises(self):
        with self.assertRaises(ValueError):
            ds.skewness([5, 5, 5])

    def test_kurtosis_excess_zero_variance_raises(self):
        with self.assertRaises(ValueError):
            ds.kurtosis([5, 5, 5])

    def test_kurtosis_unbiased_needs_four(self):
        with self.assertRaises(ValueError):
            ds.kurtosis([1, 2, 3], False)

    def test_kurtosis_fisher_relationship(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        excess = ds.kurtosis(data, bias=True, fisher=True)
        raw = ds.kurtosis(data, bias=True, fisher=False)
        self.assertAlmostEqual(excess, raw - 3.0, places=12)

    def test_kurtosis_large_for_heavy_tail(self):
        heavy = [0.0] * 200 + [20.0, 21.0, 22.0]
        self.assertGreater(ds.kurtosis(heavy, bias=True, fisher=True), 0.0)


class TestJoint(unittest.TestCase):

    def test_covariance_perfect_linear(self):
        x = [1, 2, 3]
        y = [4, 5, 6]
        self.assertAlmostEqual(ds.covariance(x, y, False), 2.0 / 3.0, places=12)
        self.assertAlmostEqual(ds.covariance(x, y, True), 1.0, places=12)

    def test_covariance_sample_larger_than_population(self):
        x = [1, 3, 5, 7]
        y = [2, 4, 6, 8]
        self.assertGreater(ds.covariance(x, y, True), ds.covariance(x, y, False))

    def test_covariance_length_mismatch(self):
        with self.assertRaises(ValueError):
            ds.covariance([1, 2], [1, 2, 3])

    def test_covariance_sample_one_pair_raises(self):
        with self.assertRaises(ValueError):
            ds.covariance([1], [2], True)

    def test_pearson_perfect_positive(self):
        self.assertAlmostEqual(ds.pearson_correlation([1, 2, 3], [4, 5, 6]),
                               1.0, places=12)

    def test_pearson_perfect_negative(self):
        self.assertAlmostEqual(ds.pearson_correlation([1, 2, 3], [6, 5, 4]),
                               -1.0, places=12)

    def test_pearson_independent(self):
        # y = [1, 0, 1] is uncorrelated with x = [1, 2, 3], yet not constant
        self.assertAlmostEqual(ds.pearson_correlation([1, 2, 3], [1, 0, 1]), 0.0,
                               places=12)

    def test_pearson_constant_raises(self):
        with self.assertRaises(ValueError):
            ds.pearson_correlation([5, 5, 5], [1, 2, 3])
        with self.assertRaises(ValueError):
            ds.pearson_correlation([1, 2, 3], [7, 7, 7])

    def test_pearson_bounded_by_one(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 1, 5, 3]
        r = ds.pearson_correlation(x, y)
        self.assertLessEqual(abs(r), 1.0)


class TestPctChange(unittest.TestCase):

    def test_pct_change_known(self):
        result = ds.pct_change([100, 110, 121])
        self.assertEqual(len(result), 2)
        for value, expected in zip(result, [0.1, 0.1]):
            self.assertAlmostEqual(value, expected, places=12)

    def test_pct_change_length(self):
        self.assertEqual(len(ds.pct_change([1, 2, 3, 4])), 3)

    def test_pct_change_negative_prices(self):
        result = ds.pct_change([2, 1])
        self.assertEqual(result, [-0.5])

    def test_pct_change_too_short(self):
        with self.assertRaises(ValueError):
            ds.pct_change([5])

    def test_pct_change_zero_previous(self):
        with self.assertRaises(ValueError):
            ds.pct_change([0, 5])


class TestSummary(unittest.TestCase):

    def test_summary_keys(self):
        s = ds.summary([1, 2, 3, 4, 5])
        expected = {'count', 'mean', 'median', 'mode', 'min', 'max', 'range',
                    'q1', 'q3', 'iqr', 'variance_population',
                    'variance_sample', 'stddev_population',
                    'stddev_sample', 'skewness', 'kurtosis_excess'}
        self.assertEqual(set(s.keys()), expected)


class TestAgainstNumpyScipy(unittest.TestCase):

    def test_cross_check_random_data(self):
        try:
            import numpy as np
            from scipy import stats as scipy_stats
        except ImportError:
            self.skipTest("numpy/scipy not installed")
        rng = np.random.RandomState(7)
        data = rng.normal(0.002, 0.01, 1000).tolist()
        a = np.array(data)

        def near(x, y, tol=1e-9):
            return abs(x - y) <= tol * max(1.0, abs(y))

        self.assertTrue(near(ds.mean(data), a.mean()))
        self.assertTrue(near(ds.median(data), np.median(a)))
        self.assertTrue(near(ds.quantile(data, 0.25), np.percentile(a, 25)))
        self.assertTrue(near(ds.quantile(data, 0.75), np.percentile(a, 75)))
        self.assertTrue(near(ds.variance(data, False), a.var(ddof=0)))
        self.assertTrue(near(ds.variance(data, True), a.var(ddof=1)))
        self.assertTrue(near(ds.stddev(data, False), a.std(ddof=0)))
        self.assertTrue(near(ds.stddev(data, True), a.std(ddof=1)))
        self.assertTrue(near(ds.skewness(data, True), scipy_stats.skew(a, bias=True)))
        self.assertTrue(near(ds.skewness(data, False), scipy_stats.skew(a, bias=False)))
        self.assertTrue(near(ds.kurtosis(data, True, True),
                             scipy_stats.kurtosis(a, fisher=True, bias=True)))
        self.assertTrue(near(ds.kurtosis(data, False, True),
                             scipy_stats.kurtosis(a, fisher=True, bias=False)))
        self.assertTrue(near(ds.kurtosis(data, True, False),
                             scipy_stats.kurtosis(a, fisher=False, bias=True)))


class TestRealData(unittest.TestCase):

    def setUp(self):
        base = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(base, 'data', 'DEXUSEU_returns.csv')

    def test_dataset_exists_and_loads(self):
        import csv
        with open(self.path) as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ['date', 'return'])
        self.assertEqual(len(rows) - 1, 6930)

    def test_returns_are_well_behaved(self):
        import csv
        with open(self.path) as handle:
            rows = list(csv.reader(handle))
        returns = [float(r[1]) for r in rows[1:]]
        self.assertLess(abs(ds.mean(returns)), 0.001)
        self.assertGreater(ds.variance(returns, True), 0.0)
        self.assertLess(ds.stddev(returns, True), 0.05)
        self.assertGreater(max(returns), 0.0)
        self.assertLess(min(returns), 0.0)


if __name__ == '__main__':
    unittest.main()
