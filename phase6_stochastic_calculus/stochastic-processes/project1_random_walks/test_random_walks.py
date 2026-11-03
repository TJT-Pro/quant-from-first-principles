import math
import os
import statistics
import unittest

import random_walks as rw


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DEXUSEU_returns.csv")


class TestSimulateWalk1D(unittest.TestCase):
    def test_length_includes_start(self):
        path = rw.simulate_walk_1d(10, seed=1)
        self.assertEqual(len(path), 11)
        self.assertEqual(path[0], 0)

    def test_zero_steps(self):
        path = rw.simulate_walk_1d(0, seed=1)
        self.assertEqual(path, [0])

    def test_step_sizes_are_unit(self):
        path = rw.simulate_walk_1d(500, seed=7)
        diffs = [path[i + 1] - path[i] for i in range(len(path) - 1)]
        self.assertTrue(all(d in (-1, 1) for d in diffs))

    def test_deterministic_with_seed(self):
        path_a = rw.simulate_walk_1d(200, p=0.6, seed=42)
        path_b = rw.simulate_walk_1d(200, p=0.6, seed=42)
        self.assertEqual(path_a, path_b)

    def test_different_seeds_differ(self):
        path_a = rw.simulate_walk_1d(200, seed=1)
        path_b = rw.simulate_walk_1d(200, seed=2)
        self.assertNotEqual(path_a, path_b)

    def test_p_one_always_up(self):
        path = rw.simulate_walk_1d(50, p=1.0, seed=3)
        self.assertEqual(path[-1], 50)

    def test_p_zero_always_down(self):
        path = rw.simulate_walk_1d(50, p=0.0, seed=3)
        self.assertEqual(path[-1], -50)

    def test_negative_steps_raises(self):
        with self.assertRaises(ValueError):
            rw.simulate_walk_1d(-1)

    def test_invalid_probability_raises(self):
        with self.assertRaises(ValueError):
            rw.simulate_walk_1d(10, p=1.5)


class TestSimulateWalk2D(unittest.TestCase):
    def test_length_includes_origin(self):
        path = rw.simulate_walk_2d(20, seed=1)
        self.assertEqual(len(path), 21)
        self.assertEqual(path[0], (0, 0))

    def test_moves_are_unit_manhattan(self):
        path = rw.simulate_walk_2d(300, seed=5)
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            manhattan = abs(x1 - x0) + abs(y1 - y0)
            self.assertEqual(manhattan, 1)

    def test_deterministic_with_seed(self):
        a = rw.simulate_walk_2d(100, seed=9)
        b = rw.simulate_walk_2d(100, seed=9)
        self.assertEqual(a, b)

    def test_negative_steps_raises(self):
        with self.assertRaises(ValueError):
            rw.simulate_walk_2d(-5)


class TestTheoreticalMoments(unittest.TestCase):
    def test_symmetric_mean_zero(self):
        self.assertEqual(rw.theoretical_mean_1d(1000, 0.5), 0.0)

    def test_biased_mean(self):
        self.assertAlmostEqual(rw.theoretical_mean_1d(100, 0.6), 100 * 0.2)

    def test_symmetric_variance(self):
        self.assertAlmostEqual(rw.theoretical_variance_1d(100, 0.5), 100.0)

    def test_variance_zero_at_extremes(self):
        self.assertEqual(rw.theoretical_variance_1d(100, 1.0), 0.0)
        self.assertEqual(rw.theoretical_variance_1d(100, 0.0), 0.0)

    def test_std_is_sqrt_of_variance(self):
        var = rw.theoretical_variance_1d(64, 0.5)
        self.assertAlmostEqual(rw.theoretical_std_1d(64, 0.5), math.sqrt(var))

    def test_monte_carlo_matches_theory(self):
        n, p, trials = 400, 0.5, 4000
        finals = [rw.simulate_walk_1d(n, p=p, seed=i)[-1] for i in range(trials)]
        empirical_mean = statistics.mean(finals)
        empirical_var = statistics.variance(finals)
        self.assertAlmostEqual(empirical_mean, rw.theoretical_mean_1d(n, p), delta=1.5)
        self.assertAlmostEqual(empirical_var, rw.theoretical_variance_1d(n, p), delta=40)


class TestExactDistribution(unittest.TestCase):
    def test_prob_Sn_equals_wrong_parity_is_zero(self):
        self.assertEqual(rw.prob_Sn_equals(4, 1, 0.5), 0.0)

    def test_prob_Sn_equals_out_of_range_is_zero(self):
        self.assertEqual(rw.prob_Sn_equals(4, 6, 0.5), 0.0)

    def test_prob_Sn_equals_symmetric_case(self):
        # S_2 = 0 requires exactly one up and one down step: C(2,1) * 0.5^2
        self.assertAlmostEqual(rw.prob_Sn_equals(2, 0, 0.5), 0.5)

    def test_prob_Sn_equals_sums_to_one(self):
        n, p = 10, 0.37
        total = sum(rw.prob_Sn_equals(n, m, p) for m in range(-n, n + 1))
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_prob_Sn_at_least_matches_direct_sum(self):
        n, p, m = 15, 0.5, 3
        direct = sum(rw.prob_Sn_equals(n, k, p) for k in range(m, n + 1))
        self.assertAlmostEqual(rw.prob_Sn_at_least(n, m, p), direct, places=9)

    def test_prob_Sn_at_least_zero_is_one_when_m_very_negative(self):
        self.assertAlmostEqual(rw.prob_Sn_at_least(10, -100, 0.5), 1.0)

    def test_prob_Sn_at_most_complements_at_least(self):
        n, p, m = 12, 0.4, 2
        self.assertAlmostEqual(
            rw.prob_Sn_at_most(n, m, p) + rw.prob_Sn_at_least(n, m + 1, p), 1.0, places=9
        )


class TestReflectionPrinciple(unittest.TestCase):
    def test_matches_bruteforce_small_cases(self):
        for n, a in [(4, 1), (6, 2), (8, 3), (10, 1), (12, 4)]:
            exact = rw.reflection_principle_max_at_least(n, a, 0.5)
            brute = rw.reflection_principle_max_at_least_bruteforce(n, a)
            self.assertAlmostEqual(exact, brute, places=9, msg=f"n={n}, a={a}")

    def test_rejects_nonpositive_a(self):
        with self.assertRaises(ValueError):
            rw.reflection_principle_max_at_least(10, 0, 0.5)

    def test_rejects_biased_walk(self):
        with self.assertRaises(ValueError):
            rw.reflection_principle_max_at_least(10, 1, 0.6)

    def test_monotone_in_n(self):
        a = 3
        probs = [rw.reflection_principle_max_at_least(n, a, 0.5) for n in range(a, a + 20)]
        for earlier, later in zip(probs, probs[1:]):
            self.assertLessEqual(earlier, later + 1e-12)

    def test_probability_bounds(self):
        for n, a in [(20, 1), (20, 5), (20, 15)]:
            p = rw.reflection_principle_max_at_least(n, a, 0.5)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


class TestFirstPassage(unittest.TestCase):
    def test_pmf_zero_before_a(self):
        self.assertEqual(rw.first_passage_pmf(2, 5, 0.5), 0.0)

    def test_pmf_wrong_parity_zero(self):
        self.assertEqual(rw.first_passage_pmf(4, 1, 0.5), 0.0)

    def test_pmf_matches_direct_first_step_case(self):
        # T_1 = 1 iff the very first step is up.
        self.assertAlmostEqual(rw.first_passage_pmf(1, 1, 0.7), 0.7)

    def test_pmf_sums_less_than_or_equal_one(self):
        a = 2
        total = sum(rw.first_passage_pmf(n, a, 0.5) for n in range(a, a + 200, 2))
        self.assertLessEqual(total, 1.0 + 1e-9)

    def test_pmf_matches_simulation_frequency(self):
        a, p, trials, max_steps = 2, 0.6, 6000, 200
        times = [rw.simulate_first_passage_time(a, p=p, max_steps=max_steps, seed=i)
                  for i in range(trials)]
        for target_n in (2, 4, 6):
            empirical = sum(1 for t in times if t == target_n) / trials
            theoretical = rw.first_passage_pmf(target_n, a, p)
            self.assertAlmostEqual(empirical, theoretical, delta=0.02)

    def test_negative_a_raises(self):
        with self.assertRaises(ValueError):
            rw.first_passage_pmf(5, -1, 0.5)


class TestHittingProbability(unittest.TestCase):
    def test_certain_hit_when_p_ge_half(self):
        self.assertEqual(rw.hitting_probability_ever(5, 0.5), 1.0)
        self.assertEqual(rw.hitting_probability_ever(5, 0.9), 1.0)

    def test_biased_down_formula(self):
        p, a = 0.3, 4
        expected = (p / (1 - p)) ** a
        self.assertAlmostEqual(rw.hitting_probability_ever(a, p), expected)

    def test_smaller_a_is_more_likely_when_biased_down(self):
        p = 0.4
        self.assertGreater(rw.hitting_probability_ever(1, p), rw.hitting_probability_ever(5, p))

    def test_nonpositive_a_raises(self):
        with self.assertRaises(ValueError):
            rw.hitting_probability_ever(0, 0.5)


class TestExpectedHittingTime(unittest.TestCase):
    def test_symmetric_is_infinite(self):
        self.assertEqual(rw.expected_hitting_time_biased(3, 0.5), math.inf)

    def test_negative_drift_is_infinite(self):
        self.assertEqual(rw.expected_hitting_time_biased(3, 0.3), math.inf)

    def test_positive_drift_formula(self):
        a, p = 5, 0.7
        self.assertAlmostEqual(rw.expected_hitting_time_biased(a, p), a / (2 * p - 1))

    def test_matches_simulation_average(self):
        a, p, trials = 3, 0.75, 3000
        times = [rw.simulate_first_passage_time(a, p=p, max_steps=5000, seed=i) for i in range(trials)]
        times = [t for t in times if t is not None]
        empirical_mean = statistics.mean(times)
        theoretical_mean = rw.expected_hitting_time_biased(a, p)
        self.assertAlmostEqual(empirical_mean, theoretical_mean, delta=0.5)

    def test_nonpositive_a_raises(self):
        with self.assertRaises(ValueError):
            rw.expected_hitting_time_biased(0, 0.5)


class TestSimulateFirstPassageTime(unittest.TestCase):
    def test_returns_none_if_unreachable_in_budget(self):
        result = rw.simulate_first_passage_time(1000, p=0.1, max_steps=50, seed=1)
        self.assertIsNone(result)

    def test_returns_positive_int_when_found(self):
        result = rw.simulate_first_passage_time(1, p=1.0, max_steps=10, seed=1)
        self.assertEqual(result, 1)

    def test_nonpositive_a_raises(self):
        with self.assertRaises(ValueError):
            rw.simulate_first_passage_time(0)


class Test2DRandomWalk(unittest.TestCase):
    def test_return_probability_between_zero_and_one(self):
        prob = rw.estimate_return_probability_2d(200, 500, seed=1)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_return_probability_increases_with_horizon(self):
        short = rw.estimate_return_probability_2d(20, 4000, seed=1)
        long = rw.estimate_return_probability_2d(2000, 4000, seed=1)
        self.assertGreater(long, short)

    def test_displacement_matches_path(self):
        path = [(0, 0), (1, 0), (1, 1), (0, 1)]
        disp = rw.displacement_2d(path)
        self.assertAlmostEqual(disp[0], 0.0)
        self.assertAlmostEqual(disp[1], 1.0)
        self.assertAlmostEqual(disp[2], math.sqrt(2))
        self.assertAlmostEqual(disp[3], 1.0)


class TestFXApplication(unittest.TestCase):
    def test_load_returns_reads_real_data(self):
        returns = rw.load_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)
        self.assertTrue(all(isinstance(r, float) for r in returns[:10]))

    def test_sign_sequence_matches_length(self):
        returns = [0.01, -0.02, 0.0, 0.03]
        signs = rw.sign_sequence(returns, seed=1)
        self.assertEqual(len(signs), 4)
        self.assertEqual(signs[0], 1)
        self.assertEqual(signs[1], -1)
        self.assertIn(signs[2], (1, -1))
        self.assertEqual(signs[3], 1)

    def test_empirical_up_probability_bounds(self):
        returns = rw.load_returns(DATA_PATH)
        p_hat = rw.empirical_up_probability(returns, seed=1)
        self.assertGreater(p_hat, 0.3)
        self.assertLess(p_hat, 0.7)

    def test_cumulative_walk_length_and_start(self):
        signs = [1, 1, -1, 1]
        path = rw.cumulative_walk(signs)
        self.assertEqual(path, [0, 1, 2, 1, 2])

    def test_windowed_analysis_keys_and_sanity(self):
        returns = rw.load_returns(DATA_PATH)
        result = rw.fx_windowed_analysis(returns, window=252, seed=1)
        self.assertEqual(result["n_observations"], 6930)
        self.assertEqual(result["window"], 252)
        self.assertGreater(result["n_windows"], 20)
        self.assertAlmostEqual(result["p_hat"], 0.5, delta=0.05)
        self.assertGreaterEqual(result["theoretical_p_exceed"], 0.0)
        self.assertLessEqual(result["theoretical_p_exceed"], 1.0)
        self.assertGreaterEqual(result["empirical_p_exceed"], 0.0)
        self.assertLessEqual(result["empirical_p_exceed"], 1.0)

    def test_windowed_analysis_rejects_bad_window(self):
        returns = [0.01, -0.01, 0.02]
        with self.assertRaises(ValueError):
            rw.fx_windowed_analysis(returns, window=0)
        with self.assertRaises(ValueError):
            rw.fx_windowed_analysis(returns, window=10)


if __name__ == "__main__":
    unittest.main()
