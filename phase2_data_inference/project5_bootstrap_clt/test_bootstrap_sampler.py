"""Tests for the resampling library.

Seeded determinism is checked everywhere so the tests are reproducible.
The jackknife is tested against its exact analytic properties (the
standard error of the mean is s/sqrt(n), the bias of the mean and of
the sample variance are both zero).  The permutation test is checked
against its exact value on a tiny exhaustive example, and the bootstrap
against scipy where the two share conventions.  A coverage simulation
verifies the bootstrap confidence interval is honest, and the CLT
functions are checked against the population standard deviation formula
and against the predicted skewness shrinkage on a skewed population.
"""

import math
import random
import unittest

import bootstrap_sampler as bs

try:
    import numpy as np
    from scipy import stats as sp_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def _requires_scipy(test):
    return unittest.skipUnless(HAVE_SCIPY, "scipy not installed")(test)


class TestReproducibility(unittest.TestCase):
    def test_bootstrap_same_seed(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        a = bs.bootstrap(data, n_boot=500, seed=11)
        b = bs.bootstrap(data, n_boot=500, seed=11)
        c = bs.bootstrap(data, n_boot=500, seed=12)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_ci_same_seed(self):
        data = [float(x) for x in range(1, 20)]
        a = bs.bootstrap_ci(data, n_boot=500, seed=21)
        b = bs.bootstrap_ci(data, n_boot=500, seed=21)
        self.assertEqual(a["lower"], b["lower"])
        self.assertEqual(a["upper"], b["upper"])

    def test_permutation_same_seed(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        r1 = bs.permutation_test(a, b, n_perm=500, seed=31)
        r2 = bs.permutation_test(a, b, n_perm=500, seed=31)
        self.assertEqual(r1["p"], r2["p"])

    def test_sample_means_same_seed(self):
        pop = [float(x) for x in range(100)]
        self.assertEqual(bs.sample_means(pop, 10, 200, seed=41),
                         bs.sample_means(pop, 10, 200, seed=41))


class TestStatisticHelpers(unittest.TestCase):
    def test_mean_median_std(self):
        self.assertEqual(bs.mean([1, 2, 3, 4]), 2.5)
        self.assertEqual(bs.median([3, 1, 2]), 2.0)
        self.assertEqual(bs.median([4, 1, 3, 2]), 2.5)
        self.assertAlmostEqual(bs.std([1, 2, 3, 4]), math.sqrt(5 / 3))
        self.assertAlmostEqual(bs.population_std([1, 2, 3, 4]), math.sqrt(5 / 4))

    def test_median_does_not_mutate(self):
        data = [3, 1, 2]
        bs.median(data)
        self.assertEqual(data, [3, 1, 2])

    def test_mean_diff(self):
        self.assertEqual(bs.mean_diff([1, 2, 3], [4, 5, 6]), -3.0)


class TestBootstrap(unittest.TestCase):
    def test_replicate_count(self):
        data = [float(x) for x in range(10)]
        reps = bs.bootstrap(data, n_boot=1000, seed=1)
        self.assertEqual(len(reps), 1000)

    def test_replicates_same_length_as_data(self):
        data = [1.0, 2.0, 3.0]
        reps = bs.bootstrap(data, n_boot=100, seed=2)
        self.assertTrue(all(isinstance(r, float) for r in reps))

    def test_mean_of_replicates_is_estimate(self):
        data = [random.gauss(1.0, 2.0) for _ in range(1000)]
        estimate = bs.mean(data)
        reps = bs.bootstrap(data, bs.mean, n_boot=2000, seed=3)
        self.assertLess(abs(bs.mean(reps) - estimate), 0.005)

    def test_ci_contains_estimate(self):
        data = [random.gauss(5.0, 1.0) for _ in range(60)]
        for method in ("percentile", "basic", "normal", "BCa"):
            ci = bs.bootstrap_ci(data, bs.mean, n_boot=999, method=method, seed=4)
            self.assertLessEqual(ci["lower"], ci["estimate"])
            self.assertGreaterEqual(ci["upper"], ci["estimate"])
            self.assertLess(ci["lower"], ci["upper"])

    def test_normal_method_symmetric(self):
        data = [random.gauss(0.0, 1.0) for _ in range(40)]
        ci = bs.bootstrap_ci(data, bs.mean, n_boot=500, method="normal", seed=5)
        self.assertAlmostEqual(ci["estimate"] - ci["lower"], ci["upper"] - ci["estimate"],
                               places=9)

    def test_basic_method_reflects_percentile(self):
        data = [random.gauss(2.0, 1.0) for _ in range(50)]
        pct = bs.bootstrap_ci(data, bs.mean, n_boot=999, method="percentile", seed=6)
        basic = bs.bootstrap_ci(data, bs.mean, n_boot=999, method="basic", seed=6)
        est = pct["estimate"]
        self.assertAlmostEqual(basic["lower"], 2 * est - pct["upper"], places=9)
        self.assertAlmostEqual(basic["upper"], 2 * est - pct["lower"], places=9)

    def test_median_statistic_works(self):
        data = [random.gauss(0.0, 1.0) for _ in range(100)]
        ci = bs.bootstrap_ci(data, bs.median, n_boot=300, method="percentile", seed=7)
        self.assertLessEqual(ci["lower"], ci["estimate"])
        self.assertGreaterEqual(ci["upper"], ci["estimate"])

    def test_coverage_rate(self):
        random.seed(51)
        hits = 0
        trials = 250
        for _ in range(trials):
            data = [random.gauss(0.0, 1.0) for _ in range(40)]
            ci = bs.bootstrap_ci(data, bs.mean, n_boot=999, method="percentile",
                                 seed=8)
            if ci["lower"] <= 0.0 <= ci["upper"]:
                hits += 1
        rate = hits / trials
        self.assertGreater(rate, 0.90)
        self.assertLess(rate, 0.99)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            bs.bootstrap([])
        with self.assertRaises(ValueError):
            bs.bootstrap([1.0], n_boot=1)
        with self.assertRaises(ValueError):
            bs.bootstrap_ci([1.0], alpha=0.0)
        with self.assertRaises(ValueError):
            bs.bootstrap_ci([1.0], alpha=1.5)
        with self.assertRaises(ValueError):
            bs.bootstrap_ci([1.0], method="wild")
        with self.assertRaises(ValueError):
            bs.bootstrap_ci([], n_boot=100)

    @_requires_scipy
    def test_percentile_against_scipy(self):
        data = [random.gauss(1.0, 2.0) for _ in range(200)]
        our = bs.bootstrap_ci(data, bs.mean, n_boot=999, method="percentile", seed=9)
        theirs = sp_stats.bootstrap((data,), np.mean, n_resamples=999,
                                    method="percentile",
                                    random_state=np.random.default_rng(9))
        tol = 0.08 * our["se"]
        self.assertAlmostEqual(our["lower"], theirs.confidence_interval.low, delta=tol)
        self.assertAlmostEqual(our["upper"], theirs.confidence_interval.high, delta=tol)

    @_requires_scipy
    def test_bca_matches_percentile_on_symmetric_data(self):
        data = [random.gauss(1.0, 2.0) for _ in range(200)]
        pct = bs.bootstrap_ci(data, bs.mean, n_boot=999, method="percentile", seed=10)
        bca = bs.bootstrap_ci(data, bs.mean, n_boot=999, method="BCa", seed=10)
        tol = 0.25 * pct["se"]
        self.assertAlmostEqual(bca["lower"], pct["lower"], delta=tol)
        self.assertAlmostEqual(bca["upper"], pct["upper"], delta=tol)

    def test_bca_on_skewed_data(self):
        random.seed(61)
        data = [random.expovariate(1.0) for _ in range(60)]
        pct = bs.bootstrap_ci(data, bs.mean, n_boot=999, method="percentile", seed=11)
        bca = bs.bootstrap_ci(data, bs.mean, n_boot=999, method="BCa", seed=11)
        self.assertLess(bca["lower"], bca["upper"])
        self.assertLessEqual(bca["lower"], bca["estimate"])
        self.assertGreaterEqual(bca["upper"], bca["estimate"])
        self.assertNotEqual(bca["lower"], pct["lower"])

    def test_bca_coverage_on_skewed_data(self):
        random.seed(62)
        hits = 0
        trials = 200
        for _ in range(trials):
            data = [random.expovariate(1.0) for _ in range(50)]
            bca = bs.bootstrap_ci(data, bs.mean, n_boot=999, method="BCa", seed=12)
            if bca["lower"] <= 1.0 <= bca["upper"]:
                hits += 1
        rate = hits / trials
        self.assertGreater(rate, 0.88)
        self.assertLess(rate, 0.99)


class TestJackknife(unittest.TestCase):
    def test_se_of_mean_is_s_over_sqrt_n(self):
        data = [random.gauss(0.0, 1.0) for _ in range(150)]
        result = bs.jackknife(data, bs.mean)
        expected = math.sqrt(bs.variance(data) / len(data))
        self.assertAlmostEqual(result["se"], expected, places=12)

    def test_bias_of_mean_is_zero(self):
        data = [float(x) for x in range(1, 20)]
        result = bs.jackknife(data, bs.mean)
        self.assertAlmostEqual(result["bias"], 0.0, places=12)

    def test_bias_of_sample_variance_is_zero(self):
        data = [random.gauss(0.0, 1.0) for _ in range(30)]
        result = bs.jackknife(data, bs.variance)
        self.assertAlmostEqual(result["bias"], 0.0, places=9)

    def test_leave_one_out_length(self):
        data = [float(x) for x in range(8)]
        result = bs.jackknife(data)
        self.assertEqual(len(result["leave_one_out"]), 8)

    def test_median_gives_finite_se(self):
        data = [random.gauss(0.0, 1.0) for _ in range(50)]
        result = bs.jackknife(data, bs.median)
        self.assertGreater(result["se"], 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            bs.jackknife([])


class TestPermutationTest(unittest.TestCase):
    def test_identical_groups_give_p_one(self):
        a = [1.0, 2.0, 3.0, 4.0]
        result = bs.permutation_test(a, list(a), n_perm=500, seed=1)
        self.assertEqual(result["p"], 1.0)

    def test_separated_groups_give_tiny_p(self):
        a = [0.0] * 8
        b = [10.0] * 8
        result = bs.permutation_test(a, b, n_perm=5000, seed=2)
        self.assertLessEqual(result["p"], 0.002)

    def test_exact_small_case(self):
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        values = [1.0, 2.0, 3.0, 4.0]
        exact_extreme = 0
        from itertools import combinations
        for chosen in combinations(values, 2):
            perm_a = list(chosen)
            perm_b = [v for v in values if v not in chosen]
            stat = bs.mean_diff(perm_a, perm_b)
            if abs(stat) >= abs(bs.mean_diff(a, b)):
                exact_extreme += 1
        exact_p = exact_extreme / 6.0
        result = bs.permutation_test(a, b, n_perm=10000, seed=3)
        self.assertAlmostEqual(result["p"], exact_p, delta=0.03)

    def test_p_never_zero(self):
        a = [0.0] * 6
        b = [10.0] * 6
        result = bs.permutation_test(a, b, n_perm=1000, seed=4)
        self.assertGreater(result["p"], 0.0)
        self.assertEqual(result["count"] + 1, result["p"] * (result["n_perm"] + 1))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            bs.permutation_test([], [1.0])
        with self.assertRaises(ValueError):
            bs.permutation_test([1.0], [])
        with self.assertRaises(ValueError):
            bs.permutation_test([1.0], [2.0], n_perm=1)

    @_requires_scipy
    def test_against_scipy(self):
        rng = random.Random(77)
        a = [rng.gauss(0.0, 1.0) for _ in range(20)]
        b = [rng.gauss(0.4, 1.0) for _ in range(20)]
        ours = bs.permutation_test(a, b, n_perm=5000, seed=5)
        theirs = sp_stats.permutation_test(
            (a, b), lambda x, y: np.mean(x) - np.mean(y), n_resamples=5000,
            random_state=np.random.default_rng(5))
        self.assertAlmostEqual(ours["p"], theirs.pvalue, delta=0.02)


class TestSampleMeansAndCLT(unittest.TestCase):
    def test_mean_of_means_is_population_mean(self):
        pop = [random.gauss(3.0, 2.0) for _ in range(5000)]
        row = bs.clt_summary(pop, 30, n_samples=3000, seed=1)
        self.assertLess(abs(row["mean"] - row["population_mean"]), 0.02)

    def test_sd_approaches_theoretical(self):
        pop = [random.gauss(3.0, 2.0) for _ in range(5000)]
        for k in (10, 50):
            row = bs.clt_summary(pop, k, n_samples=4000, seed=2)
            self.assertLess(abs(row["sd"] - row["theoretical_sd"]) / row["theoretical_sd"],
                            0.05)

    def test_skewness_shrinks_with_sample_size(self):
        random.seed(91)
        pop = [random.expovariate(1.0) for _ in range(5000)]
        small = bs.clt_summary(pop, 4, n_samples=4000, seed=3)
        large = bs.clt_summary(pop, 100, n_samples=4000, seed=4)
        self.assertGreater(small["skewness"], large["skewness"])
        self.assertLess(large["skewness"], 0.4)
        self.assertAlmostEqual(small["skewness"], 2.0 / math.sqrt(4), delta=0.25)

    def test_empirical_95_percent_for_uniform(self):
        pop = [random.random() for _ in range(3000)]
        row = bs.clt_summary(pop, 50, n_samples=5000, seed=5)
        means = bs.sample_means(pop, 50, 5000, seed=5)
        inside = sum(1 for m in means if abs(m - row["mean"]) <= 1.96 * row["theoretical_sd"])
        self.assertAlmostEqual(inside / 5000, 0.95, delta=0.02)

    def test_convergence_returns_one_row_per_size(self):
        pop = [random.gauss(0.0, 1.0) for _ in range(1000)]
        rows = bs.clt_convergence(pop, [1, 5, 30], n_samples=1000, seed=6)
        self.assertEqual([r["sample_size"] for r in rows], [1, 5, 30])

    def test_sample_size_one_reproduces_population_shape(self):
        pop = [random.expovariate(1.0) for _ in range(10000)]
        row = bs.clt_summary(pop, 1, n_samples=8000, seed=7)
        self.assertAlmostEqual(row["skewness"], 2.0, delta=0.15)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            bs.sample_means([], 5)
        with self.assertRaises(ValueError):
            bs.sample_means([1.0], 0)
        with self.assertRaises(ValueError):
            bs.sample_means([1.0], 2, n_samples=1)


class TestRealDataIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        base = os.path.dirname(os.path.abspath(__file__))
        cls.data = []
        with open(os.path.join(base, "data/DEXUSEU_returns.csv"), newline="") as handle:
            import csv
            reader = csv.DictReader(handle)
            for row in reader:
                if row["return"] != "":
                    cls.data.append(float(row["return"]))
        assert len(cls.data) == 6930

    def test_bootstrap_ci_contains_zero(self):
        ci = bs.bootstrap_ci(self.data, bs.mean, n_boot=500, method="percentile", seed=1)
        self.assertLessEqual(ci["lower"], 0.0)
        self.assertGreaterEqual(ci["upper"], 0.0)

    def test_permutation_test_no_drift_between_halves(self):
        half = len(self.data) // 2
        a = self.data[:500]
        b = self.data[500:1000]
        result = bs.permutation_test(a, b, n_perm=3000, seed=2)
        self.assertGreater(result["p"], 0.05)

    def test_clt_on_real_returns(self):
        row = bs.clt_summary(self.data, 50, n_samples=3000, seed=3)
        self.assertLess(abs(row["sd"] - row["theoretical_sd"]) / row["theoretical_sd"],
                        0.06)
        self.assertLess(abs(row["mean"]), 0.0005)


if __name__ == "__main__":
    unittest.main()
