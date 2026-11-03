import math
import os
import unittest

import markov_chains as mc

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DEXUSEU_returns.csv")


class TestLinearAlgebra(unittest.TestCase):
    def test_matrix_multiply_identity(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        i = mc.identity_matrix(2)
        self.assertEqual(mc.matrix_multiply(a, i), a)

    def test_matrix_multiply_known_result(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        result = mc.matrix_multiply(a, b)
        self.assertEqual(result, [[19.0, 22.0], [43.0, 50.0]])

    def test_vector_matrix_multiply(self):
        v = [1.0, 0.0]
        a = [[0.3, 0.7], [0.4, 0.6]]
        self.assertEqual(mc.vector_matrix_multiply(v, a), [0.3, 0.7])

    def test_transpose(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self.assertEqual(mc.transpose(a), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_solve_linear_system_simple(self):
        a = [[2.0, 1.0], [1.0, 3.0]]
        b = [5.0, 10.0]
        x = mc.solve_linear_system(a, b)
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(x[1], 3.0)

    def test_solve_linear_system_singular_raises(self):
        a = [[1.0, 2.0], [2.0, 4.0]]
        with self.assertRaises(ValueError):
            mc.solve_linear_system(a, [1.0, 2.0])

    def test_matrix_power_zero_is_identity(self):
        p = [[0.5, 0.5], [0.3, 0.7]]
        self.assertEqual(mc.matrix_power(p, 0), mc.identity_matrix(2))

    def test_matrix_power_matches_repeated_multiply(self):
        p = [[0.2, 0.8], [0.6, 0.4]]
        by_squaring = mc.matrix_power(p, 5)
        manual = p
        for _ in range(4):
            manual = mc.matrix_multiply(manual, p)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(by_squaring[i][j], manual[i][j])


class TestValidation(unittest.TestCase):
    def test_valid_matrix_passes(self):
        mc.validate_transition_matrix([[0.5, 0.5], [0.3, 0.7]])

    def test_rows_not_summing_to_one_raises(self):
        with self.assertRaises(ValueError):
            mc.validate_transition_matrix([[0.5, 0.4], [0.3, 0.7]])

    def test_negative_entry_raises(self):
        with self.assertRaises(ValueError):
            mc.validate_transition_matrix([[1.2, -0.2], [0.3, 0.7]])

    def test_non_square_raises(self):
        with self.assertRaises(ValueError):
            mc.validate_transition_matrix([[0.5, 0.3, 0.2], [0.3, 0.7]])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            mc.validate_transition_matrix([])


class TestSimulateChain(unittest.TestCase):
    def test_length_and_start(self):
        p = [[0.5, 0.5], [0.5, 0.5]]
        states = mc.simulate_chain(p, initial_state=0, n_steps=100, seed=1)
        self.assertEqual(len(states), 101)
        self.assertEqual(states[0], 0)

    def test_all_states_valid(self):
        p = [[0.5, 0.5], [0.5, 0.5]]
        states = mc.simulate_chain(p, initial_state=0, n_steps=200, seed=2)
        self.assertTrue(all(s in (0, 1) for s in states))

    def test_deterministic_with_seed(self):
        p = [[0.4, 0.6], [0.3, 0.7]]
        a = mc.simulate_chain(p, 0, 50, seed=7)
        b = mc.simulate_chain(p, 0, 50, seed=7)
        self.assertEqual(a, b)

    def test_absorbing_state_stays(self):
        p = [[1.0, 0.0], [0.5, 0.5]]
        states = mc.simulate_chain(p, initial_state=0, n_steps=30, seed=3)
        self.assertTrue(all(s == 0 for s in states))

    def test_out_of_range_initial_state_raises(self):
        p = [[0.5, 0.5], [0.5, 0.5]]
        with self.assertRaises(ValueError):
            mc.simulate_chain(p, initial_state=5, n_steps=10)


class TestStructuralProperties(unittest.TestCase):
    def test_reachable_states_full_chain(self):
        p = [[0.5, 0.5], [0.5, 0.5]]
        self.assertEqual(mc.reachable_states(p, 0), {0, 1})

    def test_irreducible_true_for_connected_chain(self):
        p = [[0.5, 0.5], [0.5, 0.5]]
        self.assertTrue(mc.is_irreducible(p))

    def test_irreducible_false_for_absorbing_chain(self):
        p = [[1.0, 0.0], [0.5, 0.5]]
        self.assertFalse(mc.is_irreducible(p))

    def test_communicating_classes_absorbing_chain(self):
        p = [[1.0, 0.0, 0.0], [0.2, 0.5, 0.3], [0.0, 0.0, 1.0]]
        classes = mc.communicating_classes(p)
        class_sets = sorted([frozenset(c) for c in classes], key=lambda s: min(s))
        self.assertIn(frozenset({0}), class_sets)
        self.assertIn(frozenset({2}), class_sets)
        self.assertIn(frozenset({1}), class_sets)

    def test_communicating_classes_fully_irreducible(self):
        p = [[0.5, 0.5], [0.5, 0.5]]
        classes = mc.communicating_classes(p)
        self.assertEqual(classes, [{0, 1}])

    def test_period_of_two_cycle(self):
        p = [[0.0, 1.0], [1.0, 0.0]]
        self.assertEqual(mc.period_of_state(p, 0), 2)

    def test_period_of_aperiodic_chain(self):
        p = [[0.5, 0.5], [0.5, 0.5]]
        self.assertEqual(mc.period_of_state(p, 0), 1)

    def test_period_of_three_cycle(self):
        p = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        self.assertEqual(mc.period_of_state(p, 0), 3)

    def test_is_aperiodic_true_and_false(self):
        aperiodic = [[0.6, 0.4], [0.3, 0.7]]
        periodic = [[0.0, 1.0], [1.0, 0.0]]
        self.assertTrue(mc.is_aperiodic(aperiodic))
        self.assertFalse(mc.is_aperiodic(periodic))


class TestStationaryDistribution(unittest.TestCase):
    def test_two_state_closed_form(self):
        # For P = [[1-a, a], [b, 1-b]], pi = (b/(a+b), a/(a+b)).
        a, b = 0.1, 0.3
        p = [[1 - a, a], [b, 1 - b]]
        pi = mc.stationary_distribution_linear(p)
        self.assertAlmostEqual(pi[0], b / (a + b), places=8)
        self.assertAlmostEqual(pi[1], a / (a + b), places=8)

    def test_sums_to_one(self):
        p = [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]]
        pi = mc.stationary_distribution_linear(p)
        self.assertAlmostEqual(sum(pi), 1.0, places=8)

    def test_is_a_fixed_point(self):
        p = [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]]
        pi = mc.stationary_distribution_linear(p)
        pi_next = mc.vector_matrix_multiply(pi, p)
        for x, y in zip(pi, pi_next):
            self.assertAlmostEqual(x, y, places=8)

    def test_power_iteration_matches_linear_solve(self):
        p = [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]]
        exact = mc.stationary_distribution_linear(p)
        iterative = mc.stationary_distribution_power_iteration(p)
        for x, y in zip(exact, iterative):
            self.assertAlmostEqual(x, y, places=6)

    def test_uniform_chain_gives_uniform_stationary(self):
        n = 4
        p = [[1.0 / n] * n for _ in range(n)]
        pi = mc.stationary_distribution_linear(p)
        for x in pi:
            self.assertAlmostEqual(x, 1.0 / n, places=8)


class TestConvergence(unittest.TestCase):
    def test_total_variation_distance_zero_for_identical(self):
        self.assertAlmostEqual(mc.total_variation_distance([0.3, 0.7], [0.3, 0.7]), 0.0)

    def test_total_variation_distance_known_value(self):
        self.assertAlmostEqual(mc.total_variation_distance([1.0, 0.0], [0.0, 1.0]), 1.0)

    def test_convergence_trace_decreases_toward_zero(self):
        p = [[0.6, 0.4], [0.3, 0.7]]
        trace = mc.convergence_trace(p, [1.0, 0.0], n_max=40)
        self.assertLess(trace[-1], trace[0])
        self.assertLess(trace[-1], 1e-4)

    def test_convergence_rate_is_negative_for_ergodic_chain(self):
        p = [[0.6, 0.4], [0.3, 0.7]]
        trace = mc.convergence_trace(p, [1.0, 0.0], n_max=30)
        slope = mc.estimate_convergence_rate(trace)
        self.assertIsNotNone(slope)
        self.assertLess(slope, 0.0)

    def test_periodic_chain_never_converges_pointwise(self):
        # The 2-cycle [[0,1],[1,0]] alternates deterministically between
        # [1,0] and [0,1] and never settles at its stationary [0.5,0.5];
        # the TV distance stays pinned at 0.5 for every n rather than
        # decaying to zero.
        p = [[0.0, 1.0], [1.0, 0.0]]
        trace = mc.convergence_trace(p, [1.0, 0.0], n_max=20)
        for d in trace:
            self.assertAlmostEqual(d, 0.5, places=8)


class TestFirstPassageTimes(unittest.TestCase):
    def test_two_state_known_formula(self):
        # From state 0, E[time to reach 1] = 1/a for P=[[1-a,a],[b,1-b]].
        a, b = 0.25, 0.4
        p = [[1 - a, a], [b, 1 - b]]
        h = mc.mean_first_passage_times(p, target=1)
        self.assertAlmostEqual(h[0], 1.0 / a, places=6)
        self.assertEqual(h[1], 0.0)

    def test_target_has_zero_passage_time(self):
        p = [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]]
        h = mc.mean_first_passage_times(p, target=2)
        self.assertEqual(h[2], 0.0)

    def test_passage_times_are_positive(self):
        p = [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]]
        h = mc.mean_first_passage_times(p, target=0)
        for i, hi in enumerate(h):
            if i != 0:
                self.assertGreater(hi, 0.0)

    def test_expected_return_time_matches_kac_formula(self):
        p = [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]]
        pi = mc.stationary_distribution_linear(p)
        for state in range(3):
            self.assertAlmostEqual(mc.expected_return_time(pi, state), 1.0 / pi[state])

    def test_zero_stationary_probability_raises(self):
        with self.assertRaises(ValueError):
            mc.expected_return_time([0.0, 1.0], 0)


class TestChiSquare(unittest.TestCase):
    def test_sf_at_zero_is_one(self):
        self.assertEqual(mc.chi2_sf(0.0, 3), 1.0)

    def test_sf_decreasing_in_x(self):
        vals = [mc.chi2_sf(x, 4) for x in (0.5, 2.0, 5.0, 10.0)]
        for earlier, later in zip(vals, vals[1:]):
            self.assertGreater(earlier, later)

    def test_sf_known_value_df_2(self):
        # For df=2, chi2 is exponential with rate 1/2: sf(x) = exp(-x/2).
        for x in (1.0, 3.0, 6.0):
            self.assertAlmostEqual(mc.chi2_sf(x, 2), math.exp(-x / 2), places=8)

    def test_median_chi2_df1_close_to_known_value(self):
        # The median of chi-square with 1 df is about 0.4549.
        self.assertAlmostEqual(mc.chi2_sf(0.4549, 1), 0.5, places=3)


class TestFXApplication(unittest.TestCase):
    def test_load_returns_reads_real_data(self):
        returns = mc.load_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)

    def test_sign_states_are_binary(self):
        returns = mc.load_returns(DATA_PATH)
        states = mc.sign_states(returns, seed=1)
        self.assertTrue(all(s in (0, 1) for s in states))
        self.assertEqual(len(states), len(returns))

    def test_quantile_breakpoints_count(self):
        data = list(range(100))
        bp = mc.quantile_breakpoints([float(x) for x in data], n_bins=5)
        self.assertEqual(len(bp), 4)
        self.assertEqual(bp, sorted(bp))

    def test_assign_bins_respects_breakpoints(self):
        breakpoints = [0.0, 10.0]
        data = [-5.0, 0.0, 5.0, 10.0, 15.0]
        bins = mc.assign_bins(data, breakpoints)
        self.assertEqual(bins, [0, 0, 1, 1, 2])

    def test_empirical_transition_matrix_is_stochastic(self):
        states = [0, 1, 1, 0, 1, 0, 0, 1]
        matrix = mc.empirical_transition_matrix(states, 2)
        mc.validate_transition_matrix(matrix)

    def test_independence_test_detects_strong_dependence(self):
        # Alternating sequence is perfectly predictable: strongly dependent.
        states = [0, 1] * 500
        result = mc.markov_independence_test(states, 2)
        self.assertLess(result["p_value"], 1e-6)

    def test_independence_test_iid_gives_large_p_value(self):
        import random
        rng = random.Random(123)
        states = [rng.randrange(2) for _ in range(4000)]
        result = mc.markov_independence_test(states, 2)
        self.assertGreater(result["p_value"], 0.01)

    def test_analyze_fx_markov_chain_structure(self):
        returns = mc.load_returns(DATA_PATH)
        result = mc.analyze_fx_markov_chain(returns, n_bins=5, seed=1)
        self.assertEqual(result["n_bins"], 5)
        self.assertEqual(len(result["stationary_distribution"]), 5)
        self.assertAlmostEqual(sum(result["stationary_distribution"]), 1.0, places=6)
        self.assertGreater(result["mean_passage_low_to_high"], 0.0)
        self.assertGreater(result["mean_passage_high_to_low"], 0.0)

    def test_fx_sign_chain_lag_one_dependence_is_not_significant(self):
        # A day-to-day (lag-1) independence test on the real sign
        # process does not reject independence at conventional levels.
        # This matters for Project 1's windowed-variance observation:
        # whatever produced the elevated 252-day variance there is not
        # showing up as detectable one-step transition dependence here.
        returns = mc.load_returns(DATA_PATH)
        states = mc.sign_states(returns, seed=1)
        result = mc.markov_independence_test(states, 2)
        self.assertGreater(result["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
