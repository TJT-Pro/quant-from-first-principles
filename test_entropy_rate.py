import unittest

import entropy_rate as er


class TestStationaryDistribution(unittest.TestCase):
    def test_sums_to_one(self):
        p = [[0.9, 0.1], [0.3, 0.7]]
        pi = er.stationary_distribution(p)
        self.assertAlmostEqual(sum(pi), 1.0, places=8)

    def test_is_a_fixed_point(self):
        p = [[0.9, 0.1], [0.3, 0.7]]
        pi = er.stationary_distribution(p)
        next_pi = [sum(pi[i] * p[i][j] for i in range(2)) for j in range(2)]
        for a, b in zip(pi, next_pi):
            self.assertAlmostEqual(a, b, places=6)

    def test_matches_known_closed_form_for_two_states(self):
        # For P=[[1-a,a],[b,1-b]], stationary pi = (b/(a+b), a/(a+b)).
        a, b = 0.1, 0.3
        p = [[1 - a, a], [b, 1 - b]]
        pi = er.stationary_distribution(p)
        self.assertAlmostEqual(pi[0], b / (a + b), places=6)
        self.assertAlmostEqual(pi[1], a / (a + b), places=6)


class TestExactMarkovEntropyRate(unittest.TestCase):
    def test_deterministic_chain_has_zero_entropy_rate(self):
        # A chain that always transitions deterministically has no
        # remaining uncertainty at all: entropy rate exactly 0.
        p = [[1.0, 0.0], [0.0, 1.0]]
        self.assertAlmostEqual(er.exact_markov_entropy_rate(p), 0.0, places=10)

    def test_iid_uniform_chain_has_entropy_rate_of_one_bit(self):
        # If every row is uniform (next state independent of current),
        # this degenerates to an i.i.d. fair-coin-like source: entropy
        # rate exactly 1 bit for a 2-state chain.
        p = [[0.5, 0.5], [0.5, 0.5]]
        self.assertAlmostEqual(er.exact_markov_entropy_rate(p), 1.0, places=10)

    def test_matches_manual_weighted_row_entropy_computation(self):
        import entropy as ent
        p = [[0.9, 0.1], [0.3, 0.7]]
        pi = er.stationary_distribution(p)
        expected = pi[0] * ent.entropy({0: 0.9, 1: 0.1}) + pi[1] * ent.entropy({0: 0.3, 1: 0.7})
        self.assertAlmostEqual(er.exact_markov_entropy_rate(p), expected, places=10)


class TestSimulateMarkovChain(unittest.TestCase):
    def test_correct_length(self):
        p = [[0.9, 0.1], [0.3, 0.7]]
        seq = er.simulate_markov_chain(p, n_steps=100, seed=1)
        self.assertEqual(len(seq), 101)

    def test_deterministic_given_seed(self):
        p = [[0.9, 0.1], [0.3, 0.7]]
        seq1 = er.simulate_markov_chain(p, n_steps=100, seed=42)
        seq2 = er.simulate_markov_chain(p, n_steps=100, seed=42)
        self.assertEqual(seq1, seq2)

    def test_only_visits_valid_states(self):
        p = [[0.9, 0.1], [0.3, 0.7]]
        seq = er.simulate_markov_chain(p, n_steps=1000, seed=1)
        self.assertTrue(all(s in (0, 1) for s in seq))


class TestBlockEntropyRateEstimate(unittest.TestCase):
    def test_converges_toward_exact_rate_as_block_length_increases(self):
        p = [[0.9, 0.1], [0.3, 0.7]]
        exact = er.exact_markov_entropy_rate(p)
        seq = er.simulate_markov_chain(p, n_steps=100000, seed=1)

        estimates = [er.block_entropy_rate_estimate(seq, k) for k in (1, 2, 4, 6)]
        errors = [abs(e - exact) for e in estimates]
        for earlier, later in zip(errors, errors[1:]):
            self.assertGreaterEqual(earlier, later - 1e-6)
        self.assertLess(errors[-1], 0.05)

    def test_block_length_one_equals_marginal_entropy(self):
        p = [[0.9, 0.1], [0.3, 0.7]]
        seq = er.simulate_markov_chain(p, n_steps=5000, seed=1)
        import entropy as ent
        marginal_entropy = ent.entropy_from_counts(seq)
        block_1_estimate = er.block_entropy_rate_estimate(seq, 1)
        self.assertAlmostEqual(block_1_estimate, marginal_entropy, places=10)


class TestConditionalEntropyRateEstimate(unittest.TestCase):
    def test_order_one_matches_exact_rate_for_first_order_markov_chain(self):
        p = [[0.9, 0.1], [0.3, 0.7]]
        exact = er.exact_markov_entropy_rate(p)
        seq = er.simulate_markov_chain(p, n_steps=200000, seed=1)
        estimate = er.conditional_entropy_rate_estimate(seq, order=1)
        self.assertAlmostEqual(estimate, exact, delta=0.01)

    def test_converges_faster_than_block_entropy_at_matched_sample_size(self):
        # The central claim: for a genuinely first-order Markov
        # process, order-1 conditioning gets close to the true rate
        # with far less data than growing the block length does.
        p = [[0.9, 0.1], [0.3, 0.7]]
        exact = er.exact_markov_entropy_rate(p)
        seq = er.simulate_markov_chain(p, n_steps=20000, seed=2)

        cond_error = abs(er.conditional_entropy_rate_estimate(seq, order=1) - exact)
        block_error = abs(er.block_entropy_rate_estimate(seq, 4) - exact)
        self.assertLess(cond_error, block_error)

    def test_higher_order_conditioning_gives_upper_bound_property(self):
        # Conditioning on the true Markov order already captures
        # everything relevant; conditioning on MORE than that (a
        # higher order than the chain's real memory) should not make
        # the estimate meaningfully worse, since it is still
        # conditioning on at least the truly relevant information.
        p = [[0.9, 0.1], [0.3, 0.7]]
        exact = er.exact_markov_entropy_rate(p)
        seq = er.simulate_markov_chain(p, n_steps=200000, seed=3)
        order_1 = er.conditional_entropy_rate_estimate(seq, order=1)
        order_2 = er.conditional_entropy_rate_estimate(seq, order=2)
        self.assertAlmostEqual(order_1, exact, delta=0.01)
        self.assertAlmostEqual(order_2, exact, delta=0.02)


if __name__ == "__main__":
    unittest.main()
