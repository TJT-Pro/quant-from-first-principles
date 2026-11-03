import math
import os
import statistics
import unittest

import poisson_processes as pp

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DEXUSEU_returns.csv")


class TestExactDistribution(unittest.TestCase):
    def test_pmf_sums_to_one(self):
        rate, t = 3.0, 2.0
        total = sum(pp.pmf(rate, t, k) for k in range(200))
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_pmf_zero_events_matches_exp(self):
        rate, t = 2.0, 1.5
        self.assertAlmostEqual(pp.pmf(rate, t, 0), math.exp(-rate * t))

    def test_pmf_zero_rate_or_time_gives_certain_zero(self):
        self.assertEqual(pp.pmf(0.0, 5.0, 0), 1.0)
        self.assertEqual(pp.pmf(5.0, 0.0, 0), 1.0)
        self.assertEqual(pp.pmf(0.0, 5.0, 3), 0.0)

    def test_pmf_negative_k_is_zero(self):
        self.assertEqual(pp.pmf(2.0, 1.0, -1), 0.0)

    def test_pmf_matches_factorial_definition_small_k(self):
        rate, t, k = 4.0, 0.5, 3
        mean = rate * t
        expected = (mean ** k) * math.exp(-mean) / math.factorial(k)
        self.assertAlmostEqual(pp.pmf(rate, t, k), expected, places=10)

    def test_cdf_is_cumulative_of_pmf(self):
        rate, t = 2.5, 3.0
        direct = sum(pp.pmf(rate, t, j) for j in range(6))
        self.assertAlmostEqual(pp.cdf(rate, t, 5), direct, places=10)

    def test_mean_and_variance_equal(self):
        rate, t = 1.7, 4.0
        self.assertEqual(pp.mean_count(rate, t), pp.variance_count(rate, t))
        self.assertAlmostEqual(pp.mean_count(rate, t), rate * t)

    def test_negative_rate_raises(self):
        with self.assertRaises(ValueError):
            pp.pmf(-1.0, 1.0, 0)


class TestSimulateArrivals(unittest.TestCase):
    def test_arrivals_within_horizon(self):
        arrivals = pp.simulate_arrivals(2.0, 10.0, seed=1)
        self.assertTrue(all(0 <= a <= 10.0 for a in arrivals))

    def test_arrivals_are_sorted(self):
        arrivals = pp.simulate_arrivals(3.0, 20.0, seed=2)
        self.assertEqual(arrivals, sorted(arrivals))

    def test_deterministic_with_seed(self):
        a = pp.simulate_arrivals(2.0, 15.0, seed=42)
        b = pp.simulate_arrivals(2.0, 15.0, seed=42)
        self.assertEqual(a, b)

    def test_count_mean_matches_theory(self):
        rate, t_max, trials = 5.0, 10.0, 5000
        counts = [len(pp.simulate_arrivals(rate, t_max, seed=i)) for i in range(trials)]
        self.assertAlmostEqual(statistics.mean(counts), rate * t_max, delta=0.5)
        self.assertAlmostEqual(statistics.variance(counts), rate * t_max, delta=3.0)

    def test_nonpositive_rate_raises(self):
        with self.assertRaises(ValueError):
            pp.simulate_arrivals(0.0, 10.0)


class TestSimulateCountDirect(unittest.TestCase):
    def test_nonnegative_integer(self):
        count = pp.simulate_count_direct(3.0, 2.0, seed=1)
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_matches_arrival_based_method_in_distribution(self):
        rate, t, trials = 4.0, 3.0, 6000
        direct_counts = [pp.simulate_count_direct(rate, t, seed=i) for i in range(trials)]
        arrival_counts = [len(pp.simulate_arrivals(rate, t, seed=trials + i)) for i in range(trials)]
        self.assertAlmostEqual(statistics.mean(direct_counts), statistics.mean(arrival_counts), delta=0.3)
        self.assertAlmostEqual(statistics.mean(direct_counts), rate * t, delta=0.3)


class TestInterarrivalAndBins(unittest.TestCase):
    def test_interarrival_times_empty(self):
        self.assertEqual(pp.interarrival_times([]), [])

    def test_interarrival_times_known_sequence(self):
        arrivals = [1.0, 3.0, 4.5]
        self.assertEqual(pp.interarrival_times(arrivals), [1.0, 2.0, 1.5])

    def test_count_in_window(self):
        arrivals = [0.5, 1.5, 2.5, 3.5]
        self.assertEqual(pp.count_in_window(arrivals, 1.0, 3.0), 2)

    def test_counts_in_bins_total_matches_arrivals(self):
        arrivals = [0.1, 1.2, 2.3, 3.9, 4.4, 7.8]
        counts = pp.counts_in_bins(arrivals, t_max=8.0, n_bins=4)
        self.assertEqual(sum(counts), len(arrivals))
        self.assertEqual(len(counts), 4)

    def test_counts_in_bins_matches_manual_partition(self):
        arrivals = [0.5, 2.5, 4.5, 6.5]
        counts = pp.counts_in_bins(arrivals, t_max=8.0, n_bins=4)
        self.assertEqual(counts, [1, 1, 1, 1])


class TestSuperposition(unittest.TestCase):
    def test_superpose_merges_and_sorts(self):
        a = [1.0, 3.0, 5.0]
        b = [2.0, 4.0]
        self.assertEqual(pp.superpose(a, b), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_superposed_rate_matches_sum_of_rates(self):
        rate_a, rate_b, t_max, trials = 2.0, 3.0, 20.0, 3000
        counts = []
        for i in range(trials):
            a = pp.simulate_arrivals(rate_a, t_max, seed=2 * i)
            b = pp.simulate_arrivals(rate_b, t_max, seed=2 * i + 1)
            counts.append(len(pp.superpose(a, b)))
        self.assertAlmostEqual(statistics.mean(counts), (rate_a + rate_b) * t_max, delta=1.5)

    def test_superpose_single_list(self):
        a = [1.0, 2.0, 3.0]
        self.assertEqual(pp.superpose(a), a)


class TestThinning(unittest.TestCase):
    def test_thin_partitions_all_arrivals(self):
        arrivals = list(range(100))
        kept, discarded = pp.thin([float(x) for x in arrivals], 0.3, seed=1)
        self.assertEqual(len(kept) + len(discarded), len(arrivals))
        self.assertEqual(sorted(kept + discarded), sorted(float(x) for x in arrivals))

    def test_thin_fraction_matches_probability(self):
        rate, t_max, keep_prob = 10.0, 200.0, 0.4
        arrivals = pp.simulate_arrivals(rate, t_max, seed=1)
        kept, discarded = pp.thin(arrivals, keep_prob, seed=2)
        fraction = len(kept) / len(arrivals)
        self.assertAlmostEqual(fraction, keep_prob, delta=0.05)

    def test_thinned_streams_match_theoretical_rates(self):
        rate, t_max, keep_prob, trials = 5.0, 50.0, 0.3, 800
        kept_counts, discarded_counts = [], []
        for i in range(trials):
            arrivals = pp.simulate_arrivals(rate, t_max, seed=i)
            kept, discarded = pp.thin(arrivals, keep_prob, seed=trials + i)
            kept_counts.append(len(kept))
            discarded_counts.append(len(discarded))
        self.assertAlmostEqual(statistics.mean(kept_counts), rate * keep_prob * t_max, delta=3.0)
        self.assertAlmostEqual(statistics.mean(discarded_counts), rate * (1 - keep_prob) * t_max, delta=3.0)

    def test_invalid_keep_prob_raises(self):
        with self.assertRaises(ValueError):
            pp.thin([1.0, 2.0], 1.5)


class TestInhomogeneous(unittest.TestCase):
    def test_expected_count_constant_rate_matches_homogeneous(self):
        rate = 4.0
        expected = pp.expected_inhomogeneous_count(lambda t: rate, 10.0)
        self.assertAlmostEqual(expected, rate * 10.0, places=3)

    def test_expected_count_linear_rate(self):
        # integral of t from 0 to 10 is 50
        expected = pp.expected_inhomogeneous_count(lambda t: t, 10.0, n_grid=20000)
        self.assertAlmostEqual(expected, 50.0, places=2)

    def test_simulated_count_matches_expected_integral(self):
        rate_func = lambda t: 1.0 + 0.5 * t
        t_max = 10.0
        rate_max = rate_func(t_max)
        expected = pp.expected_inhomogeneous_count(rate_func, t_max)
        trials = 3000
        counts = [len(pp.simulate_inhomogeneous(rate_func, t_max, rate_max, seed=i)) for i in range(trials)]
        self.assertAlmostEqual(statistics.mean(counts), expected, delta=0.5)

    def test_rate_exceeding_max_raises(self):
        with self.assertRaises(ValueError):
            pp.simulate_inhomogeneous(lambda t: 100.0, 5.0, rate_max=1.0, seed=1)

    def test_nonpositive_rate_max_raises(self):
        with self.assertRaises(ValueError):
            pp.simulate_inhomogeneous(lambda t: 1.0, 5.0, rate_max=0.0)


class TestChiSquare(unittest.TestCase):
    def test_sf_known_value_df_2(self):
        for x in (1.0, 3.0, 6.0):
            self.assertAlmostEqual(pp.chi2_sf(x, 2), math.exp(-x / 2), places=8)

    def test_sf_at_zero_is_one(self):
        self.assertEqual(pp.chi2_sf(0.0, 4), 1.0)

    def test_sf_decreasing(self):
        vals = [pp.chi2_sf(x, 5) for x in (1.0, 3.0, 8.0, 15.0)]
        for earlier, later in zip(vals, vals[1:]):
            self.assertGreater(earlier, later)


class TestDispersionTest(unittest.TestCase):
    def test_equal_counts_give_zero_statistic(self):
        result = pp.fisher_dispersion_test([5, 5, 5, 5, 5])
        self.assertAlmostEqual(result["statistic"], 0.0)
        self.assertAlmostEqual(result["p_value"], 1.0)

    def test_poisson_simulated_counts_not_rejected(self):
        rate, t_max, n_bins = 8.0, 400.0, 40
        arrivals = pp.simulate_arrivals(rate, t_max, seed=1)
        counts = pp.counts_in_bins(arrivals, t_max, n_bins)
        result = pp.fisher_dispersion_test(counts)
        self.assertGreater(result["p_value"], 0.01)
        self.assertAlmostEqual(result["index_of_dispersion"], 1.0, delta=0.6)

    def test_overdispersed_counts_rejected(self):
        # Two widely different constant rates in alternating blocks
        # produces far more bin-to-bin variance than a Poisson process
        # of the same overall mean would.
        counts = ([0] * 20 + [40] * 20)
        result = pp.fisher_dispersion_test(counts)
        self.assertLess(result["p_value"], 0.001)
        self.assertGreater(result["index_of_dispersion"], 5.0)

    def test_too_few_bins_raises(self):
        with self.assertRaises(ValueError):
            pp.fisher_dispersion_test([5])

    def test_zero_mean_raises(self):
        with self.assertRaises(ValueError):
            pp.fisher_dispersion_test([0, 0, 0])


class TestExponentialMomentsCheck(unittest.TestCase):
    def test_exponential_samples_have_cv_near_one(self):
        rate = 2.0
        arrivals = pp.simulate_arrivals(rate, 5000.0, seed=1)
        gaps = pp.interarrival_times(arrivals)
        result = pp.exponential_moments_check(gaps)
        self.assertAlmostEqual(result["coefficient_of_variation"], 1.0, delta=0.05)
        self.assertAlmostEqual(result["mean"], 1.0 / rate, delta=0.02)

    def test_too_few_gaps_raises(self):
        with self.assertRaises(ValueError):
            pp.exponential_moments_check([1.0])


class TestFXApplication(unittest.TestCase):
    def test_load_returns_reads_real_data(self):
        returns = pp.load_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)

    def test_percentile_matches_known_points(self):
        data = [float(x) for x in range(101)]
        self.assertAlmostEqual(pp.percentile(data, 0.5), 50.0)
        self.assertAlmostEqual(pp.percentile(data, 0.0), 0.0)
        self.assertAlmostEqual(pp.percentile(data, 1.0), 100.0)

    def test_large_move_arrival_times_respect_quantile(self):
        returns = pp.load_returns(DATA_PATH)
        arrivals = pp.large_move_arrival_times(returns, quantile=0.95)
        expected_count = round(len(returns) * 0.05)
        self.assertAlmostEqual(len(arrivals), expected_count, delta=len(returns) * 0.01)

    def test_analyze_fx_poisson_process_structure(self):
        returns = pp.load_returns(DATA_PATH)
        result = pp.analyze_fx_poisson_process(returns, quantile=0.95, n_bins=20)
        self.assertEqual(result["n_observations"], 6930)
        self.assertGreater(result["n_arrivals"], 0)
        self.assertGreater(result["rate_hat"], 0.0)
        self.assertEqual(len(result["counts"]), 20)
        self.assertIn("p_value", result["dispersion_test"])

    def test_large_moves_are_overdispersed_relative_to_poisson(self):
        # The central empirical claim: real large FX moves cluster in
        # time far more than a homogeneous Poisson arrival process
        # would produce.
        returns = pp.load_returns(DATA_PATH)
        result = pp.analyze_fx_poisson_process(returns, quantile=0.95, n_bins=30)
        self.assertGreater(result["dispersion_test"]["index_of_dispersion"], 1.3)
        self.assertLess(result["dispersion_test"]["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
