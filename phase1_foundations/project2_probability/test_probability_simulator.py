"""Tests for Project 2 - Probability Simulator & Bayes Engine.

Run with:  python3 -m unittest -v
"""

import os
import random
import unittest

import probability_simulator as ps


# ---------------------------------------------------------------------------
# coins, dice, empirical probability
# ---------------------------------------------------------------------------

class TestBasicSimulation(unittest.TestCase):

    def test_flip_coin(self):
        flips = ps.flip_coin(100, random.Random(1))
        self.assertEqual(len(flips), 100)
        self.assertTrue(all(f in ('H', 'T') for f in flips))

    def test_roll_die_bounds(self):
        rolls = ps.roll_die(6, 1000, random.Random(2))
        self.assertEqual(len(rolls), 1000)
        self.assertTrue(all(1 <= r <= 6 for r in rolls))

    def test_roll_die_default_sides(self):
        rolls = ps.roll_die(n=1, rng=random.Random(3))
        self.assertTrue(all(1 <= r <= 6 for r in rolls))

    def test_theoretical_die_probability(self):
        self.assertEqual(ps.theoretical_die_probability(6), 1.0 / 6.0)
        self.assertEqual(ps.theoretical_die_probability(20), 1.0 / 20.0)
        with self.assertRaises(ValueError):
            ps.theoretical_die_probability(0)

    def test_empirical_probability_value_and_predicate(self):
        outcomes = ['H', 'T', 'H', 'H']
        self.assertEqual(ps.empirical_probability(outcomes, 'H'), 0.75)
        self.assertEqual(ps.empirical_probability(outcomes, lambda x: x == 'T'), 0.25)

    def test_empirical_probability_empty_raises(self):
        with self.assertRaises(ValueError):
            ps.empirical_probability([], 'H')

    def test_empirical_probability_approximately_converges(self):
        rng = random.Random(7)
        flips = ps.flip_coin(50000, rng)
        p_heads = ps.empirical_probability(flips, 'H')
        self.assertLess(abs(p_heads - 0.5), 0.01)

    def test_running_mean(self):
        self.assertEqual(ps.running_mean([1, 2, 3]), [1.0, 1.5, 2.0])
        self.assertEqual(ps.running_mean([5]), [5.0])

    def test_deterministic_with_seed(self):
        a = ps.flip_coin(50, random.Random(42))
        b = ps.flip_coin(50, random.Random(42))
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# joint tables, marginals, conditional probability
# ---------------------------------------------------------------------------

class TestConditionalProbability(unittest.TestCase):

    TABLE = {('rain', 'yes'): 0.36, ('rain', 'no'): 0.04,
             ('dry', 'yes'): 0.24, ('dry', 'no'): 0.36}

    def test_marginals_first_variable(self):
        m = ps.marginals(self.TABLE, 0)
        self.assertAlmostEqual(m['rain'], 0.40, places=12)
        self.assertAlmostEqual(m['dry'], 0.60, places=12)

    def test_marginals_second_variable(self):
        m = ps.marginals(self.TABLE, 1)
        self.assertAlmostEqual(m['yes'], 0.60, places=12)
        self.assertAlmostEqual(m['no'], 0.40, places=12)

    def test_conditional_known_value(self):
        self.assertAlmostEqual(
            ps.conditional_probability(self.TABLE, 'rain', 'no'), 0.1, places=12)
        self.assertAlmostEqual(
            ps.conditional_probability(self.TABLE, 'rain', 'yes'), 0.6, places=12)

    def test_conditional_zero_given_raises(self):
        table = {('a', 'x'): 1.0}
        with self.assertRaises(ValueError):
            ps.conditional_probability(table, 'a', 'y')

    def test_joint_table_not_normalized_raises(self):
        with self.assertRaises(ValueError):
            ps.marginals({('a', 'b'): 0.5}, 0)

    def test_sample_joint_respects_distribution(self):
        rng = random.Random(5)
        samples = ps.sample_joint(self.TABLE, 50000, rng)
        self.assertEqual(len(samples), 50000)
        p_rain_no = ps.empirical_conditional(samples, 'rain', 'no')
        self.assertLess(abs(p_rain_no - 0.1), 0.02)

    def test_empirical_conditional_matches_computed(self):
        rng = random.Random(6)
        samples = ps.sample_joint(self.TABLE, 50000, rng)
        computed = ps.conditional_probability(self.TABLE, 'dry', 'yes')
        simulated = ps.empirical_conditional(samples, 'dry', 'yes')
        self.assertLess(abs(computed - simulated), 0.02)


# ---------------------------------------------------------------------------
# Bayes engine
# ---------------------------------------------------------------------------

class TestBayes(unittest.TestCase):

    def test_medical_test(self):
        result = ps.medical_test_example()
        posterior = result['posterior']
        self.assertAlmostEqual(posterior['disease'], 1.0 / 6.0, places=12)
        self.assertAlmostEqual(sum(posterior.values()), 1.0, places=12)

    def test_medical_numbers_explicit(self):
        result = ps.medical_test_example()
        self.assertEqual(result['prior']['disease'], 0.01)
        self.assertEqual(result['likelihood']['disease'], 0.99)
        self.assertEqual(result['likelihood']['healthy'], 0.05)

    def test_certain_evidence(self):
        posterior = ps.bayes({'A': 0.5, 'B': 0.5}, {'A': 1.0, 'B': 0.0})
        self.assertEqual(posterior['A'], 1.0)
        self.assertEqual(posterior['B'], 0.0)

    def test_uniform_prior_returns_normalized_likelihood(self):
        posterior = ps.bayes({'A': 0.5, 'B': 0.5}, {'A': 0.25, 'B': 0.75})
        self.assertAlmostEqual(posterior['A'], 0.25, places=12)
        self.assertAlmostEqual(posterior['B'], 0.75, places=12)

    def test_prior_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            ps.bayes({'A': 0.2}, {'A': 0.5})

    def test_missing_likelihood_raises(self):
        with self.assertRaises(ValueError):
            ps.bayes({'A': 0.5, 'B': 0.5}, {'A': 0.5})

    def test_impossible_evidence_raises(self):
        with self.assertRaises(ValueError):
            ps.bayes({'A': 1.0}, {'A': 0.0})

    def test_empty_prior_raises(self):
        with self.assertRaises(ValueError):
            ps.bayes({}, {})


# ---------------------------------------------------------------------------
# samplers and PMFs
# ---------------------------------------------------------------------------

class TestDistributions(unittest.TestCase):

    def test_binomial_mean(self):
        rng = random.Random(8)
        samples = ps.binomial_sample(10, 0.3, 50000, rng)
        mean = sum(samples) / len(samples)
        self.assertLess(abs(mean - 3.0), 0.05)

    def test_binomial_support_within_bounds(self):
        samples = ps.binomial_sample(5, 0.5, 200, random.Random(9))
        self.assertTrue(all(0 <= s <= 5 for s in samples))

    def test_binomial_invalid_params(self):
        with self.assertRaises(ValueError):
            ps.binomial_sample(-1, 0.5)
        with self.assertRaises(ValueError):
            ps.binomial_sample(5, 1.5)

    def test_geometric_mean(self):
        rng = random.Random(10)
        samples = ps.geometric_sample(0.25, 50000, rng)
        mean = sum(samples) / len(samples)
        self.assertLess(abs(mean - 4.0), 0.08)
        self.assertTrue(all(s >= 1 for s in samples))

    def test_geometric_invalid_p(self):
        with self.assertRaises(ValueError):
            ps.geometric_sample(0.0)
        with self.assertRaises(ValueError):
            ps.geometric_sample(1.5)

    def test_poisson_mean(self):
        rng = random.Random(11)
        samples = ps.poisson_sample(2.5, 50000, rng)
        mean = sum(samples) / len(samples)
        self.assertLess(abs(mean - 2.5), 0.05)
        self.assertTrue(all(s >= 0 for s in samples))

    def test_poisson_zero_rate(self):
        rng = random.Random(12)
        samples = ps.poisson_sample(0.0001, 100, rng)
        self.assertTrue(all(s == 0 for s in samples))

    def test_poisson_invalid_lam(self):
        with self.assertRaises(ValueError):
            ps.poisson_sample(0.0)
        with self.assertRaises(ValueError):
            ps.poisson_sample(-1.0)

    def test_binomial_pmf_sum_to_one(self):
        total = sum(ps.binomial_pmf(10, 0.3, k) for k in range(0, 11))
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_geometric_pmf_sum_to_one(self):
        total = sum(ps.geometric_pmf(0.25, k) for k in range(1, 500))
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_poisson_pmf_sum_to_one(self):
        total = sum(ps.poisson_pmf(2.5, k) for k in range(0, 60))
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_pmf_out_of_support_is_zero(self):
        self.assertEqual(ps.binomial_pmf(5, 0.5, 6), 0.0)
        self.assertEqual(ps.binomial_pmf(5, 0.5, -1), 0.0)
        self.assertEqual(ps.geometric_pmf(0.5, 0), 0.0)
        self.assertEqual(ps.poisson_pmf(2.0, -1), 0.0)

    def test_empirical_matches_pmf_closely(self):
        rng = random.Random(13)
        samples = ps.poisson_sample(4.0, 30000, rng)
        counts = ps.empirical_counts(samples)
        for k in range(0, 10):
            emp = counts.get(k, 0) / len(samples)
            theo = ps.poisson_pmf(4.0, k)
            self.assertLess(abs(emp - theo), 0.01)


if __name__ == '__main__':
    unittest.main()
