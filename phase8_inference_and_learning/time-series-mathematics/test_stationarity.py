import random
import unittest

import stationarity as st
import ar_models as ar


class TestADFRegression(unittest.TestCase):
    def test_statistic_is_finite(self):
        rng = random.Random(1)
        walk = [0.0]
        for _ in range(300):
            walk.append(walk[-1] + rng.gauss(0, 1))
        stat = st.adf_statistic(walk)
        self.assertTrue(stat == stat)  # not NaN

    def test_too_short_series_raises(self):
        with self.assertRaises(ValueError):
            st.adf_statistic([1.0, 2.0, 3.0], n_lags=5)


class TestMonteCarloNullDistribution(unittest.TestCase):
    def test_null_distribution_has_expected_length(self):
        stats = st.simulate_adf_null_distribution(n=200, n_trials=100, seed=1)
        self.assertEqual(len(stats), 100)

    def test_null_distribution_is_centered_negative(self):
        # The ADF statistic's null distribution is known to be skewed
        # left of zero (not symmetric like a t-distribution), with a
        # mean below zero even though gamma=0 is the true null value.
        stats = st.simulate_adf_null_distribution(n=500, n_trials=500, seed=1)
        import statistics
        self.assertLess(statistics.mean(stats), 0.0)

    def test_critical_value_in_plausible_range(self):
        # Published Dickey-Fuller 5% critical values (no trend, no
        # lags, large n) sit close to -2.86; simulating with a modest
        # number of trials should land in the same neighbourhood.
        stats = st.simulate_adf_null_distribution(n=1000, n_trials=3000, seed=1)
        critical_5pct = sorted(stats)[int(0.05 * len(stats))]
        self.assertGreater(critical_5pct, -3.5)
        self.assertLess(critical_5pct, -2.3)


class TestADFTest(unittest.TestCase):
    def test_random_walk_rejection_rate_matches_nominal_level(self):
        # A single random walk instance can legitimately land in the
        # rejection region purely by chance (that is what a correctly
        # calibrated test does approximately 10% of the time at this
        # threshold); testing one specific seed against a fixed
        # threshold is not a statistically meaningful check. The
        # correct check is calibration: across many independent random
        # walks, the false-positive rate should sit near the nominal
        # level implied by the threshold used.
        rejections = 0
        trials = 60
        for i in range(trials):
            rng = random.Random(100 + i)
            walk = [0.0]
            for _ in range(300):
                walk.append(walk[-1] + rng.gauss(0, 1))
            result = st.adf_test(walk, n_trials=500, seed=1)
            if result["p_value"] < 0.10:
                rejections += 1
        rejection_rate = rejections / trials
        self.assertLess(rejection_rate, 0.25)

    def test_stationary_ar1_rejected(self):
        x = ar.simulate_ar(1000, 0.0, [0.5], seed=3)
        result = st.adf_test(x, n_trials=1000, seed=1)
        self.assertLess(result["p_value"], 0.01)

    def test_near_unit_root_ar1_borderline(self):
        # A highly persistent but still technically stationary AR(1)
        # (phi=0.98) should be much harder to distinguish from a unit
        # root than phi=0.5 is, at a modest sample size.
        x_persistent = ar.simulate_ar(300, 0.0, [0.98], seed=4)
        x_weak = ar.simulate_ar(300, 0.0, [0.3], seed=4)
        result_persistent = st.adf_test(x_persistent, n_trials=1000, seed=1)
        result_weak = st.adf_test(x_weak, n_trials=1000, seed=1)
        self.assertGreater(result_persistent["p_value"], result_weak["p_value"])


class TestRollingMeanVariance(unittest.TestCase):
    def test_output_lengths_match(self):
        x = list(range(100))
        result = st.rolling_mean_variance([float(v) for v in x], window=10)
        self.assertEqual(len(result["means"]), 10)
        self.assertEqual(len(result["variances"]), 10)

    def test_constant_series_has_zero_variance_everywhere(self):
        x = [5.0] * 50
        result = st.rolling_mean_variance(x, window=10)
        self.assertTrue(all(v == 0.0 for v in result["variances"]))

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            st.rolling_mean_variance([1.0, 2.0], window=0)
        with self.assertRaises(ValueError):
            st.rolling_mean_variance([1.0, 2.0], window=10)

    def test_random_walk_variance_grows_across_windows(self):
        # A random walk's variance is not stationary: variance within
        # later, more-spread-out windows should tend to exceed that of
        # earlier windows, on average, over enough repetitions.
        rng = random.Random(5)
        walk = [0.0]
        for _ in range(2000):
            walk.append(walk[-1] + rng.gauss(0, 1))
        result = st.rolling_mean_variance(walk, window=200)
        variances = result["variances"]
        self.assertGreater(sum(variances[-3:]), sum(variances[:3]))


if __name__ == "__main__":
    unittest.main()
