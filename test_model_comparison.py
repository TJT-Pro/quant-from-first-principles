import math
import random
import statistics
import unittest

import conjugate_models as cm
import model_comparison as mcomp


class TestBayesFactor(unittest.TestCase):
    def test_equal_evidence_gives_factor_one(self):
        self.assertAlmostEqual(mcomp.bayes_factor(-10.0, -10.0), 1.0)

    def test_higher_evidence_gives_factor_above_one(self):
        bf = mcomp.bayes_factor(-5.0, -10.0)
        self.assertGreater(bf, 1.0)

    def test_log_bayes_factor_matches_direct_subtraction(self):
        self.assertAlmostEqual(mcomp.log_bayes_factor(-3.0, -7.0), 4.0)

    def test_bayes_factor_is_exponential_of_log_version(self):
        le1, le2 = -12.3, -15.7
        self.assertAlmostEqual(mcomp.bayes_factor(le1, le2), math.exp(le1 - le2))


class TestInterpretBayesFactor(unittest.TestCase):
    def test_boundaries(self):
        self.assertIn("favours model 2", mcomp.interpret_bayes_factor(0.5))
        self.assertIn("barely", mcomp.interpret_bayes_factor(2.0))
        self.assertIn("substantial", mcomp.interpret_bayes_factor(5.0))
        self.assertIn("strong", mcomp.interpret_bayes_factor(20.0))
        self.assertIn("decisive", mcomp.interpret_bayes_factor(500.0))


class TestConjugateModelComparisonRealExample(unittest.TestCase):
    def test_beta_binomial_prior_choice_affects_evidence(self):
        # A prior sharply concentrated away from the data's implied
        # rate should be penalised relative to one that isn't.
        k, n = 80, 100
        confident_wrong_prior = cm.BetaBinomial(50.0, 1.0)  # strongly favours theta near 1
        uninformative_prior = cm.BetaBinomial(1.0, 1.0)
        le_wrong = confident_wrong_prior.log_evidence(k, n)
        le_flat = uninformative_prior.log_evidence(k, n)
        self.assertLess(le_wrong, le_flat)


class TestHarmonicMeanEstimator(unittest.TestCase):
    def _log_likelihood_fn(self, k, n):
        log_binom = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        return lambda theta: log_binom + k * math.log(theta) + (n - k) * math.log(1 - theta)

    def test_matches_manual_formula(self):
        log_likelihoods = [-1.0, -2.0, -0.5, -3.0]
        n = len(log_likelihoods)
        from grid_approximation import log_sum_exp
        expected = math.log(n) - log_sum_exp([-ll for ll in log_likelihoods])
        self.assertAlmostEqual(mcomp.harmonic_mean_log_evidence(log_likelihoods), expected)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            mcomp.harmonic_mean_log_evidence([])

    def test_is_in_the_right_ballpark_of_the_true_evidence(self):
        # It should not be wildly, order-of-magnitude wrong, only
        # unreliable in the specific way the next two tests document.
        a, b, k, n = 2.0, 2.0, 15, 20
        posterior = cm.BetaBinomial(a, b).update(k, n)
        true_log_evidence = cm.BetaBinomial(a, b).log_evidence(k, n)
        log_likelihood = self._log_likelihood_fn(k, n)
        rng = random.Random(0)
        samples = [rng.betavariate(posterior.a, posterior.b) for _ in range(4000)]
        estimate = mcomp.harmonic_mean_log_evidence([log_likelihood(s) for s in samples])
        self.assertAlmostEqual(estimate, true_log_evidence, delta=1.0)

    def test_systematically_overestimates_the_true_evidence(self):
        # A real, documented pathology, verified here against a known
        # closed-form answer rather than only asserted: the harmonic
        # mean estimator's known finite-sample bias is not a
        # zero-mean fluctuation, it systematically overstates the
        # true marginal likelihood across independent replications.
        a, b, k, n = 2.0, 2.0, 15, 20
        posterior = cm.BetaBinomial(a, b).update(k, n)
        true_log_evidence = cm.BetaBinomial(a, b).log_evidence(k, n)
        log_likelihood = self._log_likelihood_fn(k, n)

        estimates = []
        for trial in range(20):
            rng = random.Random(trial)
            samples = [rng.betavariate(posterior.a, posterior.b) for _ in range(2000)]
            estimates.append(mcomp.harmonic_mean_log_evidence([log_likelihood(s) for s in samples]))

        self.assertGreater(statistics.mean(estimates), true_log_evidence)

    def test_standard_error_does_not_shrink_reliably_with_more_samples(self):
        # The other real, documented pathology: unlike a well-behaved
        # Monte Carlo estimator, whose standard error should shrink
        # roughly as O(1/sqrt(N)), the harmonic mean estimator's error
        # is dominated by rare low-likelihood posterior draws and does
        # not shrink reliably as N grows. This test checks that the
        # variance at a much larger N is not dramatically smaller than
        # at a modest N (the opposite of what a healthy estimator
        # would guarantee), rather than asserting any specific ordering,
        # since the whole point is that the ordering is unreliable.
        a, b, k, n = 2.0, 2.0, 15, 20
        posterior = cm.BetaBinomial(a, b).update(k, n)
        log_likelihood = self._log_likelihood_fn(k, n)

        def stdev_at(sample_size, n_replications=15):
            estimates = []
            for trial in range(n_replications):
                rng = random.Random(5000 + trial)
                samples = [rng.betavariate(posterior.a, posterior.b) for _ in range(sample_size)]
                estimates.append(mcomp.harmonic_mean_log_evidence([log_likelihood(s) for s in samples]))
            return statistics.stdev(estimates)

        small_n_stdev = stdev_at(500)
        large_n_stdev = stdev_at(16000)
        # A well-behaved O(1/sqrt(N)) estimator going from N=500 to
        # N=16000 (32x) would shrink its standard error by about
        # sqrt(32) ~ 5.7x. The harmonic mean estimator is not expected
        # to deliver anywhere near that, which is the point.
        shrinkage_factor = small_n_stdev / large_n_stdev if large_n_stdev > 0 else float("inf")
        self.assertLess(shrinkage_factor, 5.7)


if __name__ == "__main__":
    unittest.main()
