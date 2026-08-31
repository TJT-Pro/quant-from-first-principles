"""Tests for the hypothesis testing library.

The distribution functions are checked against known values and against
scipy over dense grids.  The tests themselves are checked against scipy
where an equivalent exists (t tests, chi-squared tests) and against
hand-computed textbook numbers otherwise.  A coverage simulation checks
the mean confidence interval is honest, and a boundary property checks
the Wilson interval really is the inversion of the z test it claims to
be.
"""

import math
import random
import unittest

import hypothesis_tests as ht

try:
    from scipy import stats as sp_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def _requires_scipy(test):
    return unittest.skipUnless(HAVE_SCIPY, "scipy not installed")(test)


class TestNormal(unittest.TestCase):
    def test_cdf_known_values(self):
        self.assertAlmostEqual(ht.normal_cdf(0.0), 0.5)
        self.assertAlmostEqual(ht.normal_cdf(1.959963984540054), 0.975, places=12)
        self.assertAlmostEqual(ht.normal_cdf(-1.959963984540054), 0.025, places=12)
        self.assertAlmostEqual(ht.normal_cdf(3.0), 0.9986501019683699, places=12)

    def test_pdf_symmetry_and_peak(self):
        self.assertAlmostEqual(ht.normal_pdf(0.0), 0.3989422804014327)
        self.assertAlmostEqual(ht.normal_pdf(2.0), ht.normal_pdf(-2.0))

    def test_ppf_round_trip(self):
        for p in (0.001, 0.01, 0.05, 0.25, 0.5, 0.9, 0.99, 0.999):
            x = ht.normal_cdf_inverse(p)
            self.assertAlmostEqual(ht.normal_cdf(x), p, places=10)

    def test_ppf_known_values(self):
        self.assertAlmostEqual(ht.normal_cdf_inverse(0.975), 1.959963984540054,
                               places=10)
        self.assertAlmostEqual(ht.normal_cdf_inverse(0.5), 0.0, places=12)
        self.assertAlmostEqual(ht.normal_cdf_inverse(0.025), -1.959963984540054,
                               places=10)

    def test_ppf_bad_probability(self):
        for p in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                ht.normal_cdf_inverse(p)

    @_requires_scipy
    def test_cdf_against_scipy_grid(self):
        for x in [x / 100.0 for x in range(-600, 601)]:
            self.assertAlmostEqual(ht.normal_cdf(x), sp_stats.norm.cdf(x), places=14)


class TestStudentT(unittest.TestCase):
    def test_cdf_center_and_symmetry(self):
        for df in (1, 2, 5, 10, 30):
            self.assertAlmostEqual(ht.t_cdf(0.0, df), 0.5)
            for t in (0.5, 1.5, 3.0):
                self.assertAlmostEqual(ht.t_cdf(-t, df), 1 - ht.t_cdf(t, df))

    def test_large_df_approaches_normal(self):
        self.assertAlmostEqual(ht.t_cdf(1.959963984540054, 1e6),
                               ht.normal_cdf(1.959963984540054), places=6)

    def test_sf_and_cdf_sum_to_one(self):
        for df in (1, 4, 20):
            for t in (-2.0, 0.0, 2.0):
                self.assertAlmostEqual(ht.t_cdf(t, df) + ht.t_sf(t, df), 1.0)

    def test_ppf_round_trip(self):
        for df in (1, 3, 9, 40):
            for p in (0.01, 0.05, 0.5, 0.95, 0.99):
                x = ht.t_ppf(p, df)
                self.assertAlmostEqual(ht.t_cdf(x, df), p, places=9)

    def test_ppf_known_tails(self):
        self.assertAlmostEqual(ht.t_ppf(0.975, 1), 12.706204736431895, places=6)
        self.assertAlmostEqual(ht.t_ppf(0.975, 10), 2.2281388519649385, places=8)
        self.assertAlmostEqual(ht.t_ppf(0.95, 5), 2.015048372669157, places=8)

    @_requires_scipy
    def test_cdf_against_scipy_grid(self):
        for df in (1, 2, 5, 10, 30, 100):
            for t in [t / 100.0 for t in range(-800, 801)]:
                self.assertAlmostEqual(ht.t_cdf(t, df), sp_stats.t.cdf(t, df), places=10)

    @_requires_scipy
    def test_ppf_against_scipy(self):
        for df in (1, 4, 12, 50):
            for p in (0.005, 0.025, 0.5, 0.975, 0.995):
                self.assertAlmostEqual(ht.t_ppf(p, df), sp_stats.t.ppf(p, df),
                                       places=9)

    def test_bad_df(self):
        with self.assertRaises(ValueError):
            ht.t_cdf(1.0, 0.0)


class TestChiSquared(unittest.TestCase):
    def test_cdf_known_values(self):
        self.assertAlmostEqual(ht.chi2_cdf(3.841458820694124, 1), 0.95, places=9)
        self.assertAlmostEqual(ht.chi2_cdf(5.991464547107979, 2), 0.95, places=9)
        self.assertAlmostEqual(ht.chi2_cdf(0.0, 3), 0.0)

    def test_sf_complement(self):
        for df in (1, 3, 10):
            for x in (0.5, 3.0, 20.0):
                self.assertAlmostEqual(ht.chi2_cdf(x, df) + ht.chi2_sf(x, df), 1.0)

    def test_ppf_known_values(self):
        self.assertAlmostEqual(ht.chi2_ppf(0.95, 1), 3.841458820694124, places=8)
        self.assertAlmostEqual(ht.chi2_ppf(0.05, 10), 3.940299138323863, places=6)

    def test_ppf_round_trip(self):
        for df in (1, 5, 20):
            for p in (0.01, 0.1, 0.5, 0.9, 0.99):
                x = ht.chi2_ppf(p, df)
                self.assertAlmostEqual(ht.chi2_cdf(x, df), p, places=8)

    @_requires_scipy
    def test_cdf_against_scipy_grid(self):
        for df in (1, 3, 10, 50):
            for x in [x / 100.0 for x in range(1, 10001)]:
                self.assertAlmostEqual(ht.chi2_cdf(x, df), sp_stats.chi2.cdf(x, df),
                                       places=10)


class TestZTest(unittest.TestCase):
    def test_known_z(self):
        result = ht.z_test(0.5, 0.0, 1.0, 4)
        self.assertAlmostEqual(result["z"], 1.0)
        self.assertAlmostEqual(result["p"], 2 * (1 - ht.normal_cdf(1.0)), places=12)

    def test_one_sided(self):
        r_gt = ht.z_test(0.5, 0.0, 1.0, 4, "greater")
        r_lt = ht.z_test(0.5, 0.0, 1.0, 4, "less")
        self.assertAlmostEqual(r_gt["p"], 1 - ht.normal_cdf(1.0), places=12)
        self.assertAlmostEqual(r_lt["p"], ht.normal_cdf(1.0), places=12)

    def test_two_sided_symmetric(self):
        a = ht.z_test(0.5, 0.0, 1.0, 4)
        b = ht.z_test(-0.5, 0.0, 1.0, 4)
        self.assertAlmostEqual(a["p"], b["p"])

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            ht.z_test(1.0, 0.0, -1.0, 10)
        with self.assertRaises(ValueError):
            ht.z_test(1.0, 0.0, 1.0, 0)
        with self.assertRaises(ValueError):
            ht.z_test(1.0, 0.0, 1.0, 10, "sideways")


class TestTTestOneSample(unittest.TestCase):
    def test_hand_computed(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        m = 3.0
        s = math.sqrt(2.5)
        result = ht.t_test_one_sample(data, 3.0)
        self.assertAlmostEqual(result["t"], (m - 3.0) / (s / math.sqrt(5)))
        self.assertAlmostEqual(result["df"], 4)

    def test_constant_equal_to_mu0(self):
        result = ht.t_test_one_sample([2.0, 2.0, 2.0], 2.0)
        self.assertEqual(result["t"], 0.0)
        self.assertEqual(result["p"], 1.0)

    def test_constant_away_from_mu0(self):
        result = ht.t_test_one_sample([2.0, 2.0, 2.0], 0.0)
        self.assertEqual(result["p"], 0.0)
        self.assertEqual(result["t"], float("inf"))

    def test_too_few(self):
        with self.assertRaises(ValueError):
            ht.t_test_one_sample([1.0])

    @_requires_scipy
    def test_against_scipy(self):
        random.seed(4)
        for _ in range(20):
            data = [random.gauss(1.0, 2.0) for _ in range(random.randint(3, 80))]
            mu0 = random.uniform(-1, 1)
            for alt in ("two-sided", "greater", "less"):
                ours = ht.t_test_one_sample(data, mu0, alt)
                theirs = sp_stats.ttest_1samp(data, mu0, alternative=alt)
                self.assertAlmostEqual(ours["t"], theirs.statistic, places=10)
                self.assertAlmostEqual(ours["p"], theirs.pvalue, places=10)


class TestTTestTwoSample(unittest.TestCase):
    def test_hand_computed_pooled(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        m1, m2 = 2.0, 5.0
        sp = 1.0
        t = (m1 - m2) / (sp * math.sqrt(2.0 / 3.0))
        result = ht.t_test_two_sample(a, b, equal_var=True)
        self.assertAlmostEqual(result["t"], t)
        self.assertAlmostEqual(result["df"], 4)

    def test_welch_df_boundary(self):
        a = [0.0, 1.0, 2.0]
        b = [10.0, 10.0, 10.0]
        result = ht.t_test_two_sample(a, b, equal_var=False)
        self.assertAlmostEqual(result["df"], 2.0, places=9)

    def test_both_constant_raises(self):
        with self.assertRaises(ValueError):
            ht.t_test_two_sample([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])

    def test_too_few(self):
        with self.assertRaises(ValueError):
            ht.t_test_two_sample([1.0], [2.0, 3.0, 4.0])

    @_requires_scipy
    def test_against_scipy(self):
        random.seed(9)
        for _ in range(20):
            a = [random.gauss(0.0, 1.0) for _ in range(random.randint(3, 60))]
            b = [random.gauss(0.3, 2.0) for _ in range(random.randint(3, 60))]
            for equal_var in (True, False):
                for alt in ("two-sided", "greater", "less"):
                    ours = ht.t_test_two_sample(a, b, equal_var=equal_var,
                                                alternative=alt)
                    theirs = sp_stats.ttest_ind(a, b, equal_var=equal_var,
                                                alternative=alt)
                    self.assertAlmostEqual(ours["t"], theirs.statistic, places=10)
                    self.assertAlmostEqual(ours["p"], theirs.pvalue, places=10)
                    if not equal_var:
                        self.assertAlmostEqual(ours["df"], theirs.df, places=9)


class TestTTestPaired(unittest.TestCase):
    def test_equals_one_sample_on_differences(self):
        random.seed(13)
        a = [random.gauss(2.0, 1.0) for _ in range(20)]
        b = [x + random.gauss(0.5, 0.5) for x in a]
        paired = ht.t_test_paired(a, b)
        diffs = [x - y for x, y in zip(a, b)]
        one = ht.t_test_one_sample(diffs, 0.0)
        self.assertAlmostEqual(paired["t"], one["t"])
        self.assertAlmostEqual(paired["p"], one["p"])

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            ht.t_test_paired([1.0, 2.0], [1.0, 2.0, 3.0])

    @_requires_scipy
    def test_against_scipy(self):
        random.seed(17)
        for _ in range(10):
            a = [random.gauss(0.0, 1.0) for _ in range(15)]
            b = [random.gauss(0.2, 1.0) for _ in range(15)]
            for alt in ("two-sided", "greater", "less"):
                ours = ht.t_test_paired(a, b, alt)
                theirs = sp_stats.ttest_rel(a, b, alternative=alt)
                self.assertAlmostEqual(ours["t"], theirs.statistic, places=10)
                self.assertAlmostEqual(ours["p"], theirs.pvalue, places=10)


class TestChiSquaredFit(unittest.TestCase):
    def test_mendel_peas(self):
        observed = [315, 108, 101, 32]
        expected = [312.75, 104.25, 104.25, 34.75]
        result = ht.chi2_goodness_of_fit(observed, expected)
        self.assertAlmostEqual(result["chi2"], 0.4700, places=3)
        self.assertEqual(result["df"], 3)
        self.assertAlmostEqual(result["p"], 0.9254, places=3)

    def test_fair_die(self):
        observed = [100, 95, 105, 98, 102, 100]
        expected = [100.0] * 6
        result = ht.chi2_goodness_of_fit(observed, expected)
        chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
        self.assertAlmostEqual(result["chi2"], chi2)
        self.assertEqual(result["df"], 5)

    def test_contributions_sum(self):
        observed = [10, 20, 30]
        expected = [15.0, 15.0, 30.0]
        result = ht.chi2_goodness_of_fit(observed, expected)
        self.assertAlmostEqual(sum(result["contributions"]), result["chi2"])

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            ht.chi2_goodness_of_fit([1, 2], [1, 2, 3])
        with self.assertRaises(ValueError):
            ht.chi2_goodness_of_fit([1, 2], [0, 2])
        with self.assertRaises(ValueError):
            ht.chi2_goodness_of_fit([1], [1])

    @_requires_scipy
    def test_against_scipy(self):
        random.seed(21)
        for _ in range(15):
            observed = [random.randint(0, 100) for _ in range(random.randint(2, 8))]
            total = sum(observed)
            expected = [total / len(observed)] * len(observed)
            ours = ht.chi2_goodness_of_fit(observed, expected)
            theirs = sp_stats.chisquare(observed, expected)
            self.assertAlmostEqual(ours["chi2"], theirs.statistic, places=10)
            self.assertAlmostEqual(ours["p"], theirs.pvalue, places=10)


class TestChiSquaredIndependence(unittest.TestCase):
    def test_hand_computed_2x2(self):
        table = [[20, 10], [10, 20]]
        result = ht.chi2_independence(table)
        expected = [[15.0, 15.0], [15.0, 15.0]]
        chi2 = sum((table[i][j] - expected[i][j]) ** 2 / expected[i][j]
                   for i in range(2) for j in range(2))
        self.assertAlmostEqual(result["chi2"], chi2)
        self.assertEqual(result["df"], 1)

    def test_independent_table_gives_small_chi2(self):
        table = [[10, 10], [10, 10]]
        result = ht.chi2_independence(table)
        self.assertAlmostEqual(result["chi2"], 0.0)
        self.assertAlmostEqual(result["p"], 1.0)

    def test_ragged_rows_rejected(self):
        with self.assertRaises(ValueError):
            ht.chi2_independence([[1, 2], [3]])

    def test_empty_table_rejected(self):
        with self.assertRaises(ValueError):
            ht.chi2_independence([[0, 0], [0, 0]])

    @_requires_scipy
    def test_against_scipy(self):
        random.seed(27)
        for _ in range(15):
            rows = random.randint(2, 4)
            cols = random.randint(2, 4)
            table = [[random.randint(0, 60) for _ in range(cols)]
                     for _ in range(rows)]
            ours = ht.chi2_independence(table)
            theirs = sp_stats.chi2_contingency(table, correction=False)
            self.assertAlmostEqual(ours["chi2"], theirs.statistic, places=9)
            self.assertAlmostEqual(ours["p"], theirs.pvalue, places=9)


class TestBinomialProportion(unittest.TestCase):
    def test_known_z(self):
        result = ht.binomial_proportion_test(60, 100, 0.5)
        z = (0.6 - 0.5) / math.sqrt(0.25 / 100)
        self.assertAlmostEqual(result["z"], z)
        self.assertAlmostEqual(result["p"], 2 * (1 - ht.normal_cdf(z)), places=10)

    def test_continuity_correction_shifts_p(self):
        result = ht.binomial_proportion_test(60, 100, 0.5,
                                             continuity_correction=True)
        p_adj = 0.6 - 1.0 / 200.0
        z = (p_adj - 0.5) / math.sqrt(0.25 / 100)
        self.assertAlmostEqual(result["z"], z)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            ht.binomial_proportion_test(5, 0, 0.5)
        with self.assertRaises(ValueError):
            ht.binomial_proportion_test(5, 10, 1.5)
        with self.assertRaises(ValueError):
            ht.binomial_proportion_test(11, 10, 0.5)


class TestConfidenceIntervalMean(unittest.TestCase):
    def test_hand_computed(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = ht.confidence_interval_mean(data)
        m = 3.0
        s = math.sqrt(2.5)
        tcrit = ht.t_ppf(0.975, 4)
        self.assertAlmostEqual(result["lower"], m - tcrit * s / math.sqrt(5))
        self.assertAlmostEqual(result["upper"], m + tcrit * s / math.sqrt(5))

    def test_too_few(self):
        with self.assertRaises(ValueError):
            ht.confidence_interval_mean([1.0])

    def test_coverage_rate(self):
        random.seed(33)
        hits = 0
        trials = 1000
        for _ in range(trials):
            data = [random.gauss(5.0, 2.0) for _ in range(30)]
            ci = ht.confidence_interval_mean(data)
            if ci["lower"] <= 5.0 <= ci["upper"]:
                hits += 1
        self.assertGreater(hits / trials, 0.93)
        self.assertLess(hits / trials, 0.97)

    @_requires_scipy
    def test_against_scipy(self):
        random.seed(37)
        data = [random.gauss(1.0, 3.0) for _ in range(50)]
        ours = ht.confidence_interval_mean(data)
        theirs = sp_stats.t.interval(0.95, 49, loc=sum(data) / 50,
                                     scale=ht.confidence_interval_mean(data)["se"])
        self.assertAlmostEqual(ours["lower"], theirs[0], places=10)
        self.assertAlmostEqual(ours["upper"], theirs[1], places=10)


class TestConfidenceIntervalProportion(unittest.TestCase):
    def test_known_zero_case(self):
        result = ht.confidence_interval_proportion(0, 10, 0.05)
        self.assertEqual(result["lower"], 0.0)
        self.assertAlmostEqual(result["upper"], 0.27753279986288915, places=10)

    def test_known_all_case(self):
        result = ht.confidence_interval_proportion(10, 10, 0.05)
        self.assertEqual(result["upper"], 1.0)
        self.assertAlmostEqual(result["lower"], 0.7224672001371108, places=10)

    def test_wilson_is_the_inverted_z_test(self):
        random.seed(41)
        for _ in range(30):
            n = random.randint(2, 200)
            k = random.randint(0, n)
            ci = ht.confidence_interval_proportion(k, n, 0.05)
            for bound in (ci["lower"], ci["upper"]):
                if ci["lower"] == 0.0 and bound == ci["lower"]:
                    continue
                if ci["upper"] == 1.0 and bound == ci["upper"]:
                    continue
                res = ht.binomial_proportion_test(k, n, bound)
                self.assertAlmostEqual(res["p"], 0.05, places=6)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            ht.confidence_interval_proportion(1, 0)
        with self.assertRaises(ValueError):
            ht.confidence_interval_proportion(-1, 10)
        with self.assertRaises(ValueError):
            ht.confidence_interval_proportion(11, 10)


class TestCohensD(unittest.TestCase):
    def test_hand_computed(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = ht.cohens_d(a, b)
        sp = 1.0
        self.assertAlmostEqual(result, (2.0 - 5.0) / sp)

    def test_no_difference_gives_zero(self):
        self.assertAlmostEqual(ht.cohens_d([1, 2, 3], [1, 2, 3]), 0.0)

    def test_too_few(self):
        with self.assertRaises(ValueError):
            ht.cohens_d([1.0], [2.0, 3.0, 4.0])


class TestPowerAndSampleSize(unittest.TestCase):
    def test_power_known_value(self):
        result = ht.power_z_test(0.5, 1.0, 25, 0.05, "greater")
        self.assertAlmostEqual(result, 0.8038, places=3)

    def test_power_increases_with_n(self):
        p_small = ht.power_z_test(0.3, 1.0, 10, 0.05, "greater")
        p_large = ht.power_z_test(0.3, 1.0, 100, 0.05, "greater")
        self.assertGreater(p_large, p_small)

    def test_sample_size_known_value(self):
        result = ht.sample_size_z_test(0.5, 1.0, 0.05, 0.8, "greater")
        self.assertEqual(result, 25)

    def test_sample_size_rounds_up(self):
        result = ht.sample_size_z_test(0.5, 1.0, 0.05, 0.8, "greater")
        self.assertGreaterEqual(ht.power_z_test(0.5, 1.0, result, 0.05, "greater"), 0.8)
        if result > 1:
            self.assertLess(ht.power_z_test(0.5, 1.0, result - 1, 0.05, "greater"), 0.8)

    def test_zero_effect_rejected(self):
        with self.assertRaises(ValueError):
            ht.sample_size_z_test(0.0, 1.0)


class TestDecision(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(ht.decision(0.04, 0.05), "reject the null hypothesis")
        self.assertEqual(ht.decision(0.05, 0.05), "reject the null hypothesis")
        self.assertEqual(ht.decision(0.06, 0.05), "fail to reject the null hypothesis")

    def test_bad_alpha(self):
        with self.assertRaises(ValueError):
            ht.decision(0.05, -0.1)


class TestRealDataIntegration(unittest.TestCase):
    def test_eurusd_returns_have_no_detectable_mean(self):
        import csv
        import os
        base = os.path.dirname(os.path.abspath(__file__))
        data = []
        with open(os.path.join(base, "data/DEXUSEU_returns.csv"), newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["return"] != "":
                    data.append(float(row["return"]))
        self.assertEqual(len(data), 6930)
        result = ht.t_test_one_sample(data, 0.0)
        self.assertGreater(result["p"], 0.05)


if __name__ == "__main__":
    unittest.main()
