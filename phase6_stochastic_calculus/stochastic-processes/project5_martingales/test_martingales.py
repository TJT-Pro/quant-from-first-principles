import itertools
import math
import os
import statistics
import unittest

import martingales as mg

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DEXUSEU_returns.csv")


class TestStandardNormalCDF(unittest.TestCase):
    def test_at_zero(self):
        self.assertAlmostEqual(mg.standard_normal_cdf(0.0), 0.5)

    def test_symmetry(self):
        for x in (0.4, 1.3, 2.1):
            self.assertAlmostEqual(mg.standard_normal_cdf(-x), 1.0 - mg.standard_normal_cdf(x), places=10)


class TestSimulateWalk(unittest.TestCase):
    def test_length_and_start(self):
        path = mg.simulate_walk(50, p=0.5, seed=1)
        self.assertEqual(len(path), 51)
        self.assertEqual(path[0], 0)

    def test_deterministic_with_seed(self):
        a = mg.simulate_walk(30, 0.5, seed=5)
        b = mg.simulate_walk(30, 0.5, seed=5)
        self.assertEqual(a, b)

    def test_invalid_p_raises(self):
        with self.assertRaises(ValueError):
            mg.simulate_walk(10, p=1.5)


class TestEmpiricalConditionalDrift(unittest.TestCase):
    def test_martingale_shows_near_zero_drift(self):
        xs, dxs = [], []
        for i in range(4000):
            path = mg.simulate_walk(1, p=0.5, seed=i)
            xs.append(0)
            dxs.append(path[1] - path[0])
        buckets = mg.empirical_conditional_drift(xs, dxs, n_buckets=1)
        self.assertAlmostEqual(buckets[0]["mean_dx"], 0.0, delta=0.1)

    def test_biased_walk_shows_nonzero_drift(self):
        xs, dxs = [], []
        for i in range(4000):
            path = mg.simulate_walk(1, p=0.7, seed=i)
            xs.append(0)
            dxs.append(path[1] - path[0])
        buckets = mg.empirical_conditional_drift(xs, dxs, n_buckets=1)
        self.assertAlmostEqual(buckets[0]["mean_dx"], 0.4, delta=0.1)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            mg.empirical_conditional_drift([1, 2], [1], n_buckets=1)

    def test_bucket_counts_partition_data(self):
        xs = list(range(100))
        dxs = [0.0] * 100
        buckets = mg.empirical_conditional_drift(xs, dxs, n_buckets=4)
        self.assertEqual(sum(b["count"] for b in buckets), 100)


class TestCompensatedWalk(unittest.TestCase):
    def test_symmetric_case_matches_raw_walk(self):
        path = mg.compensated_walk(20, 0.5, seed=1)
        raw = mg.simulate_walk(20, 0.5, seed=1)
        for a, b in zip(path, raw):
            self.assertAlmostEqual(a, b)

    def test_restores_martingale_property_for_biased_walk(self):
        p = 0.65
        xs, dxs = [], []
        for i in range(6000):
            m = mg.compensated_walk(1, p, seed=i)
            xs.append(0)
            dxs.append(m[1] - m[0])
        buckets = mg.empirical_conditional_drift(xs, dxs, n_buckets=1)
        self.assertAlmostEqual(buckets[0]["mean_dx"], 0.0, delta=0.1)


class TestExponentialMartingale(unittest.TestCase):
    def test_degenerate_at_half_raises(self):
        with self.assertRaises(ValueError):
            mg.exponential_martingale_path(10, 0.5, seed=1)

    def test_starts_at_one(self):
        path = mg.exponential_martingale_path(10, 0.6, seed=1)
        self.assertAlmostEqual(path[0], 1.0)

    def test_is_a_martingale_by_simulation(self):
        p = 0.7
        theta = (1 - p) / p
        xs, dxs = [], []
        for i in range(8000):
            path = mg.exponential_martingale_path(1, p, seed=i)
            xs.append(0)
            dxs.append(path[1] - path[0])
        buckets = mg.empirical_conditional_drift(xs, dxs, n_buckets=1)
        self.assertAlmostEqual(buckets[0]["mean_dx"], 0.0, delta=theta * 0.1)


class TestGamblersRuin(unittest.TestCase):
    def test_symmetric_ruin_probability_closed_form(self):
        self.assertAlmostEqual(mg.ruin_probability(10, 15, 0.5), 15 / 25)

    def test_symmetric_expected_duration_closed_form(self):
        self.assertAlmostEqual(mg.expected_duration(10, 15, 0.5), 150.0)

    def test_ruin_probability_bounds(self):
        for a, b, p in [(3, 5, 0.4), (7, 2, 0.6), (10, 10, 0.5)]:
            prob = mg.ruin_probability(a, b, p)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_biased_ruin_probability_matches_simulation(self):
        a, b, p, trials = 5, 8, 0.55, 20000
        theoretical = mg.ruin_probability(a, b, p)
        outcomes = [mg.simulate_ruin(a, b, p, seed=i)[0] for i in range(trials)]
        empirical = sum(outcomes) / trials
        self.assertAlmostEqual(empirical, theoretical, delta=0.02)

    def test_biased_expected_duration_matches_simulation(self):
        a, b, p, trials = 5, 8, 0.55, 20000
        theoretical = mg.expected_duration(a, b, p)
        durations = [mg.simulate_ruin(a, b, p, seed=i)[1] for i in range(trials)]
        empirical = statistics.mean(durations)
        self.assertAlmostEqual(empirical, theoretical, delta=1.5)

    def test_nonpositive_boundaries_raise(self):
        with self.assertRaises(ValueError):
            mg.ruin_probability(0, 5, 0.5)
        with self.assertRaises(ValueError):
            mg.expected_duration(5, -1, 0.5)


class TestDoobMaximalInequality(unittest.TestCase):
    def test_bound_formula(self):
        self.assertAlmostEqual(mg.doob_maximal_bound(100, 50), 2.0)

    def test_empirical_probability_never_exceeds_bound(self):
        result = mg.doob_maximal_inequality_check(30, level=100, trials=8000, seed=1)
        self.assertTrue(result["bound_holds"])
        self.assertLessEqual(result["empirical_probability"], result["doob_bound"] + 1e-9)

    def test_nonpositive_level_raises(self):
        with self.assertRaises(ValueError):
            mg.doob_maximal_bound(10, 0.0)


class TestPolyaUrn(unittest.TestCase):
    def test_starting_fraction(self):
        path = mg.polya_urn_path(3, 5, 10, seed=1)
        self.assertAlmostEqual(path[0], 3 / 8)

    def test_length(self):
        path = mg.polya_urn_path(2, 2, 40, seed=1)
        self.assertEqual(len(path), 41)

    def test_terminal_moments_match_beta_formula(self):
        r0, b0, trials = 3, 5, 6000
        finals = [mg.polya_urn_path(r0, b0, 2000, seed=i)[-1] for i in range(trials)]
        theory = mg.polya_urn_terminal_moments(r0, b0)
        self.assertAlmostEqual(statistics.mean(finals), theory["mean"], delta=0.02)
        self.assertAlmostEqual(statistics.variance(finals), theory["variance"], delta=0.01)

    def test_martingale_property_by_simulation(self):
        r0, b0, trials = 3, 5, 4000
        xs, dxs = [], []
        for i in range(trials):
            path = mg.polya_urn_path(r0, b0, 20, seed=i)
            xs.append(path[10])
            dxs.append(path[11] - path[10])
        buckets = mg.empirical_conditional_drift(xs, dxs, n_buckets=4)
        for b in buckets:
            self.assertAlmostEqual(b["mean_dx"], 0.0, delta=0.02)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            mg.polya_urn_path(0, 5, 10)
        with self.assertRaises(ValueError):
            mg.polya_urn_path(5, 0, 10)


class TestOLSFit(unittest.TestCase):
    def test_recovers_known_linear_relationship(self):
        x = list(range(100))
        y = [2.0 + 3.0 * xi for xi in x]
        result = mg.ols_fit([float(v) for v in x], y)
        self.assertAlmostEqual(result["beta0"], 2.0, places=6)
        self.assertAlmostEqual(result["beta1"], 3.0, places=6)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=6)
        # A perfect fit has zero residual variance; the implementation
        # must not raise a ZeroDivisionError computing the z-statistic.
        self.assertEqual(result["p_homoskedastic"], 0.0)

    def test_zero_slope_for_uncorrelated_noise(self):
        import random
        rng = random.Random(1)
        x = [rng.gauss(0, 1) for _ in range(5000)]
        y = [rng.gauss(0, 1) for _ in range(5000)]
        result = mg.ols_fit(x, y)
        self.assertAlmostEqual(result["beta1"], 0.0, delta=0.1)
        self.assertGreater(result["p_homoskedastic"], 0.01)

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            mg.ols_fit([1.0, 2.0], [1.0, 2.0])

    def test_zero_variance_x_raises(self):
        with self.assertRaises(ValueError):
            mg.ols_fit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


class TestMartingaleDifferenceTest(unittest.TestCase):
    def test_structure(self):
        import random
        rng = random.Random(1)
        returns = [rng.gauss(0, 0.01) for _ in range(2000)]
        result = mg.martingale_difference_test(returns)
        self.assertIn("beta1", result)
        self.assertIn("p_homoskedastic", result)
        self.assertIn("p_robust", result)

    def test_too_few_observations_raises(self):
        with self.assertRaises(ValueError):
            mg.martingale_difference_test([0.1, 0.2])


class TestDoublingStrategy(unittest.TestCase):
    def test_exhaustive_mean_zero_uncapped_small_n(self):
        for n in (3, 5, 8, 10):
            total = 0.0
            count = 0
            for seq in itertools.product([1, -1], repeat=n):
                result = mg.simulate_doubling_strategy(list(seq), 1.0, max_doublings=None)
                total += result["final_pnl"]
                count += 1
            self.assertAlmostEqual(total / count, 0.0, places=9)

    def test_exhaustive_mean_zero_capped_small_n(self):
        for n, cap in [(6, 3), (8, 2), (10, 4)]:
            total = 0.0
            count = 0
            for seq in itertools.product([1, -1], repeat=n):
                result = mg.simulate_doubling_strategy(list(seq), 1.0, max_doublings=cap)
                total += result["final_pnl"]
                count += 1
            self.assertAlmostEqual(total / count, 0.0, places=9)

    def test_completed_win_cycle_nets_one_unit(self):
        result = mg.simulate_doubling_strategy([-1, -1, -1, 1], 1.0, max_doublings=None)
        self.assertAlmostEqual(result["final_pnl"], 1.0)

    def test_forced_stop_counted(self):
        result = mg.simulate_doubling_strategy([-1, -1, -1], 1.0, max_doublings=3)
        self.assertEqual(result["forced_stops"], 1)


class TestExactExpectedPnl(unittest.TestCase):
    def test_matches_exhaustive_enumeration(self):
        for n in (3, 6, 9):
            total = 0.0
            count = 0
            for seq in itertools.product([1, -1], repeat=n):
                total += mg.simulate_doubling_strategy(list(seq), 1.0, max_doublings=None)["final_pnl"]
                count += 1
            exhaustive_mean = total / count
            exact = mg.exact_expected_pnl(n, 0.5, 1.0, None)
            self.assertAlmostEqual(exact, exhaustive_mean, places=9)

    def test_fair_game_zero_for_large_n_uncapped_and_capped(self):
        self.assertAlmostEqual(mg.exact_expected_pnl(500, 0.5, 1.0, None), 0.0, places=6)
        self.assertAlmostEqual(mg.exact_expected_pnl(500, 0.5, 1.0, 10), 0.0, places=6)

    def test_biased_game_has_nonzero_expectation(self):
        value = mg.exact_expected_pnl(50, 0.6, 1.0, 10)
        self.assertGreater(value, 0.0)

    def test_no_overflow_for_large_n(self):
        # This would raise OverflowError under a naive 2**k implementation.
        value = mg.exact_expected_pnl(2000, 0.5, 1.0, None)
        self.assertAlmostEqual(value, 0.0, places=6)


class TestProbabilityMaxStreak(unittest.TestCase):
    def test_probability_increases_with_horizon(self):
        short = mg.probability_max_streak_at_least(100, 0.5, 10)
        long = mg.probability_max_streak_at_least(2000, 0.5, 10)
        self.assertGreater(long, short)

    def test_probability_decreases_with_required_streak_length(self):
        p10 = mg.probability_max_streak_at_least(2000, 0.5, 10)
        p20 = mg.probability_max_streak_at_least(2000, 0.5, 20)
        p30 = mg.probability_max_streak_at_least(2000, 0.5, 30)
        self.assertGreater(p10, p20)
        self.assertGreater(p20, p30)

    def test_matches_simple_closed_form_for_short_horizon(self):
        # P(streak of exactly length k occurring immediately, n=k steps)
        # is just (1-p)^k for a walk with no earlier chance to reset.
        p = 0.5
        k = 5
        exact = mg.probability_max_streak_at_least(k, p, k)
        self.assertAlmostEqual(exact, (1 - p) ** k, places=9)

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            mg.probability_max_streak_at_least(10, 0.5, 0)


class TestBlockBootstrap(unittest.TestCase):
    def test_output_length(self):
        series = list(range(50))
        out = mg.block_bootstrap_resample(series, block_length=5, n_out=200, seed=1)
        self.assertEqual(len(out), 200)

    def test_values_drawn_from_original_series(self):
        series = [1, -1, 1, 1, -1, -1, 1, -1]
        out = mg.block_bootstrap_resample(series, block_length=3, n_out=50, seed=2)
        self.assertTrue(all(v in series for v in out))

    def test_invalid_block_length_raises(self):
        with self.assertRaises(ValueError):
            mg.block_bootstrap_resample([1, 2, 3], block_length=10, n_out=5)


class TestFXApplication(unittest.TestCase):
    def test_load_returns_reads_real_data(self):
        returns = mg.load_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)

    def test_sign_outcomes_are_binary(self):
        returns = mg.load_returns(DATA_PATH)
        outcomes = mg.sign_outcomes(returns, seed=1)
        self.assertTrue(all(o in (1, -1) for o in outcomes))

    def test_martingale_difference_test_on_real_data_not_significant(self):
        returns = mg.load_returns(DATA_PATH)
        result = mg.martingale_difference_test(returns)
        self.assertGreater(result["p_homoskedastic"], 0.05)

    def test_down_streak_lengths_structure(self):
        outcomes = [1, -1, -1, 1, 1, -1, -1, -1, 1]
        streaks = mg.down_streak_lengths(outcomes)
        self.assertEqual(streaks, [2, 3])

    def test_down_streak_lengths_trailing_streak_included(self):
        outcomes = [1, -1, -1]
        self.assertEqual(mg.down_streak_lengths(outcomes), [2])

    def test_streak_tail_probabilities_structure(self):
        returns = mg.load_returns(DATA_PATH)
        outcomes = mg.sign_outcomes(returns, seed=1)
        result = mg.streak_tail_probabilities(outcomes, thresholds=[5, 10, 15])
        self.assertEqual(len(result["empirical"]), 3)
        self.assertEqual(len(result["theoretical"]), 3)
        self.assertEqual(result["empirical"], sorted(result["empirical"], reverse=True))

    def test_capped_doubling_strategy_diverges_from_iid_prediction_under_block_bootstrap(self):
        # The central, honest finding: a block bootstrap that preserves
        # real local dependence gives an average payoff far from what
        # an i.i.d. Bernoulli(p_hat) model predicts, while a resample
        # that destroys that dependence (block_length=1, i.e. genuinely
        # i.i.d. draws from the same pool) lands close to the i.i.d.
        # prediction. This shows the doubling strategy's payoff is
        # sensitive to real run-length structure that a simple lag-1
        # independence test does not probe.
        returns = mg.load_returns(DATA_PATH)
        outcomes = mg.sign_outcomes(returns, seed=1)
        p_hat = sum(1 for o in outcomes if o == 1) / len(outcomes)
        iid_prediction = mg.exact_expected_pnl(1000, p_hat, 1.0, max_doublings=10)

        block_pnls = [
            mg.simulate_doubling_strategy(
                mg.block_bootstrap_resample(outcomes, block_length=20, n_out=1000, seed=2000 + i),
                1.0, max_doublings=10)["final_pnl"]
            for i in range(600)
        ]
        iid_resample_pnls = [
            mg.simulate_doubling_strategy(
                mg.block_bootstrap_resample(outcomes, block_length=1, n_out=1000, seed=3000 + i),
                1.0, max_doublings=10)["final_pnl"]
            for i in range(600)
        ]

        block_mean = statistics.mean(block_pnls)
        block_se = statistics.stdev(block_pnls) / len(block_pnls) ** 0.5
        iid_mean = statistics.mean(iid_resample_pnls)
        iid_se = statistics.stdev(iid_resample_pnls) / len(iid_resample_pnls) ** 0.5

        # Block-preserving resample: far from the i.i.d. prediction.
        self.assertGreater(abs(block_mean - iid_prediction), 6 * block_se)
        # Dependence-destroying resample: close to the i.i.d. prediction.
        self.assertLess(abs(iid_mean - iid_prediction), 6 * iid_se)


if __name__ == "__main__":
    unittest.main()
