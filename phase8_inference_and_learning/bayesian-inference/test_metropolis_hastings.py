import math
import statistics
import unittest

import metropolis_hastings as mh


class TestMetropolisHastingsOnKnownNormal(unittest.TestCase):
    def test_recovers_normal_mean_and_std(self):
        true_mean, true_sd = 3.0, 2.0
        log_posterior = lambda theta: -0.5 * ((theta - true_mean) / true_sd) ** 2
        result = mh.metropolis_hastings(log_posterior, initial=0.0, proposal_sd=2.0,
                                         n_samples=20000, burn_in=2000, seed=1)
        samples = result["samples"]
        self.assertAlmostEqual(statistics.mean(samples), true_mean, delta=0.1)
        self.assertAlmostEqual(statistics.stdev(samples), true_sd, delta=0.1)

    def test_sample_count_matches_request(self):
        log_posterior = lambda theta: -0.5 * theta ** 2
        result = mh.metropolis_hastings(log_posterior, 0.0, 1.0, n_samples=500, burn_in=100, seed=1)
        self.assertEqual(len(result["samples"]), 500)

    def test_acceptance_rate_in_valid_range(self):
        log_posterior = lambda theta: -0.5 * theta ** 2
        result = mh.metropolis_hastings(log_posterior, 0.0, 1.0, n_samples=5000, burn_in=500, seed=1)
        self.assertGreaterEqual(result["acceptance_rate"], 0.0)
        self.assertLessEqual(result["acceptance_rate"], 1.0)

    def test_converges_regardless_of_starting_point(self):
        log_posterior = lambda theta: -0.5 * ((theta - 5.0) / 1.0) ** 2
        result_far = mh.metropolis_hastings(log_posterior, initial=100.0, proposal_sd=1.0,
                                             n_samples=10000, burn_in=3000, seed=1)
        self.assertAlmostEqual(statistics.mean(result_far["samples"]), 5.0, delta=0.1)

    def test_bounded_support_via_negative_infinity(self):
        # A target confined to theta > 0 via -inf outside the support.
        log_posterior = lambda theta: -theta if theta > 0 else float("-inf")
        result = mh.metropolis_hastings(log_posterior, initial=1.0, proposal_sd=1.0,
                                         n_samples=10000, burn_in=2000, seed=1)
        self.assertTrue(all(s > 0 for s in result["samples"]))
        # An Exponential(1) distribution has mean 1.
        self.assertAlmostEqual(statistics.mean(result["samples"]), 1.0, delta=0.1)


class TestEffectiveSampleSize(unittest.TestCase):
    def test_low_proposal_sd_gives_low_ess(self):
        # A tiny proposal step size makes consecutive samples highly
        # correlated (the chain barely moves each step), which should
        # show up as a much smaller ESS than the raw chain length.
        log_posterior = lambda theta: -0.5 * theta ** 2
        result = mh.metropolis_hastings(log_posterior, 0.0, proposal_sd=0.05,
                                         n_samples=5000, burn_in=1000, seed=1)
        ess = mh.effective_sample_size(result["samples"])
        self.assertLess(ess, 5000 * 0.3)

    def test_well_tuned_proposal_gives_higher_ess(self):
        log_posterior = lambda theta: -0.5 * theta ** 2
        result_bad = mh.metropolis_hastings(log_posterior, 0.0, proposal_sd=0.02,
                                             n_samples=5000, burn_in=1000, seed=1)
        result_good = mh.metropolis_hastings(log_posterior, 0.0, proposal_sd=2.0,
                                              n_samples=5000, burn_in=1000, seed=1)
        ess_bad = mh.effective_sample_size(result_bad["samples"])
        ess_good = mh.effective_sample_size(result_good["samples"])
        self.assertGreater(ess_good, ess_bad)

    def test_ess_never_exceeds_chain_length(self):
        log_posterior = lambda theta: -0.5 * theta ** 2
        result = mh.metropolis_hastings(log_posterior, 0.0, proposal_sd=2.0,
                                         n_samples=3000, burn_in=500, seed=1)
        ess = mh.effective_sample_size(result["samples"])
        self.assertLessEqual(ess, 3000 * 1.1)  # small slack for finite-sample noise


class TestGelmanRubin(unittest.TestCase):
    def test_converged_chains_give_rhat_near_one(self):
        log_posterior = lambda theta: -0.5 * theta ** 2
        chains = [mh.metropolis_hastings(log_posterior, start, 1.5, n_samples=5000,
                                          burn_in=2000, seed=seed)["samples"]
                  for seed, start in enumerate([-5.0, 0.0, 5.0, 10.0])]
        rhat = mh.gelman_rubin(chains)
        self.assertLess(rhat, 1.05)

    def test_nonconverged_chains_give_high_rhat(self):
        # Simulate chains that have clearly NOT mixed: constant offsets
        # with only tiny internal noise, standing in for chains stuck
        # in different regions.
        import random
        rng = random.Random(1)
        chains = [[offset + rng.gauss(0, 0.01) for _ in range(500)] for offset in (0.0, 10.0, 20.0)]
        rhat = mh.gelman_rubin(chains)
        self.assertGreater(rhat, 2.0)

    def test_too_few_chains_raises(self):
        with self.assertRaises(ValueError):
            mh.gelman_rubin([[1.0, 2.0, 3.0]])

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            mh.gelman_rubin([[1.0, 2.0], [1.0, 2.0, 3.0]])


class TestMetropolisHastingsVector(unittest.TestCase):
    def test_recovers_independent_2d_normal_marginals(self):
        def log_posterior(v):
            x, y = v
            return -0.5 * ((x - 2.0) / 1.0) ** 2 - 0.5 * ((y + 1.0) / 3.0) ** 2

        result = mh.metropolis_hastings_vector(log_posterior, [0.0, 0.0], [1.0, 3.0],
                                                n_samples=20000, burn_in=2000, seed=1)
        samples = result["samples"]
        xs = [s[0] for s in samples]
        ys = [s[1] for s in samples]
        self.assertAlmostEqual(statistics.mean(xs), 2.0, delta=0.1)
        self.assertAlmostEqual(statistics.stdev(xs), 1.0, delta=0.1)
        self.assertAlmostEqual(statistics.mean(ys), -1.0, delta=0.2)
        self.assertAlmostEqual(statistics.stdev(ys), 3.0, delta=0.2)

    def test_sample_shape(self):
        log_posterior = lambda v: -0.5 * sum(vi ** 2 for vi in v)
        result = mh.metropolis_hastings_vector(log_posterior, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                                                n_samples=500, burn_in=100, seed=1)
        self.assertEqual(len(result["samples"]), 500)
        self.assertEqual(len(result["samples"][0]), 3)

    def test_bounded_support_via_negative_infinity(self):
        # Both coordinates confined to positive values.
        log_posterior = lambda v: (-v[0] - v[1]) if (v[0] > 0 and v[1] > 0) else float("-inf")
        result = mh.metropolis_hastings_vector(log_posterior, [1.0, 1.0], [1.0, 1.0],
                                                n_samples=5000, burn_in=1000, seed=1)
        self.assertTrue(all(s[0] > 0 and s[1] > 0 for s in result["samples"]))


if __name__ == "__main__":
    unittest.main()
