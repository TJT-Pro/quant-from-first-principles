import math
import statistics
import unittest

import hmm


def two_state_model(p_stay_0=0.9, p_stay_1=0.9, mean0=0.0, mean1=5.0, var0=1.0, var1=1.0):
    return hmm.GaussianHMM(
        means=[mean0, mean1], variances=[var0, var1],
        transition=[[p_stay_0, 1 - p_stay_0], [1 - p_stay_1, p_stay_1]],
        initial=[0.5, 0.5],
    )


class TestGaussianPdf(unittest.TestCase):
    def test_matches_known_value_at_mean(self):
        self.assertAlmostEqual(hmm.gaussian_pdf(0.0, 0.0, 1.0), 1.0 / math.sqrt(2 * math.pi))

    def test_log_pdf_matches_log_of_pdf(self):
        x, mean, var = 1.3, 0.5, 2.0
        self.assertAlmostEqual(hmm.gaussian_log_pdf(x, mean, var), math.log(hmm.gaussian_pdf(x, mean, var)))

    def test_nonpositive_variance_raises(self):
        with self.assertRaises(ValueError):
            hmm.gaussian_pdf(0.0, 0.0, 0.0)


class TestModelConstruction(unittest.TestCase):
    def test_valid_construction(self):
        model = two_state_model()
        self.assertEqual(model.n_states, 2)

    def test_inconsistent_lengths_raise(self):
        with self.assertRaises(ValueError):
            hmm.GaussianHMM([0.0, 1.0], [1.0], [[0.9, 0.1], [0.1, 0.9]], [0.5, 0.5])

    def test_non_square_transition_raises(self):
        with self.assertRaises(ValueError):
            hmm.GaussianHMM([0.0, 1.0], [1.0, 1.0], [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]], [0.5, 0.5])


class TestForwardBackward(unittest.TestCase):
    def test_alpha_rows_sum_to_one(self):
        model = two_state_model()
        obs, _ = model.simulate(50, seed=1)
        alpha, scales = model.forward(obs)
        for row in alpha:
            self.assertAlmostEqual(sum(row), 1.0, places=8)

    def test_scales_are_positive(self):
        model = two_state_model()
        obs, _ = model.simulate(50, seed=1)
        _, scales = model.forward(obs)
        self.assertTrue(all(s > 0 for s in scales))

    def test_gamma_rows_sum_to_one(self):
        model = two_state_model()
        obs, _ = model.simulate(50, seed=1)
        alpha, scales = model.forward(obs)
        beta = model.backward(obs, scales)
        gamma = model.gamma(alpha, beta)
        for row in gamma:
            self.assertAlmostEqual(sum(row), 1.0, places=8)

    def test_xi_entries_sum_to_one_at_each_t(self):
        model = two_state_model()
        obs, _ = model.simulate(50, seed=1)
        alpha, scales = model.forward(obs)
        beta = model.backward(obs, scales)
        xi = model.xi(obs, alpha, beta)
        for slice_ in xi:
            total = sum(sum(row) for row in slice_)
            self.assertAlmostEqual(total, 1.0, places=8)

    def test_gamma_marginal_matches_xi_marginal(self):
        # sum_j xi_t(i,j) should equal gamma_t(i) for t < T-1: a
        # consistency identity between the two posteriors that must
        # hold regardless of the specific model parameters.
        model = two_state_model()
        obs, _ = model.simulate(80, seed=2)
        alpha, scales = model.forward(obs)
        beta = model.backward(obs, scales)
        gamma = model.gamma(alpha, beta)
        xi = model.xi(obs, alpha, beta)
        for t in range(len(xi)):
            for i in range(model.n_states):
                self.assertAlmostEqual(sum(xi[t][i]), gamma[t][i], places=6)

    def test_no_underflow_on_long_sequence(self):
        # A direct (unscaled) forward pass would underflow to exactly
        # zero well before this length; the scaled version must not.
        model = two_state_model()
        obs, _ = model.simulate(5000, seed=1)
        ll = model.log_likelihood(obs)
        self.assertTrue(math.isfinite(ll))


class TestViterbi(unittest.TestCase):
    def test_path_length_matches_observations(self):
        model = two_state_model()
        obs, _ = model.simulate(100, seed=1)
        path = model.viterbi(obs)
        self.assertEqual(len(path), 100)

    def test_recovers_true_states_on_well_separated_model(self):
        # With states far apart relative to their variance, Viterbi
        # should recover the true generating states almost perfectly.
        model = two_state_model(mean0=0.0, mean1=10.0, var0=1.0, var1=1.0)
        obs, true_states = model.simulate(2000, seed=3)
        path = model.viterbi(obs)
        accuracy = sum(1 for a, b in zip(path, true_states) if a == b) / len(true_states)
        self.assertGreater(accuracy, 0.98)

    def test_states_are_valid_indices(self):
        model = two_state_model()
        obs, _ = model.simulate(50, seed=1)
        path = model.viterbi(obs)
        self.assertTrue(all(0 <= s < model.n_states for s in path))


class TestBaumWelchEMGuarantee(unittest.TestCase):
    def test_log_likelihood_never_decreases(self):
        # The defining theoretical guarantee of EM, checked directly
        # rather than only cited: each Baum-Welch step must not lower
        # the observed-data log-likelihood.
        true_model = two_state_model(mean0=0.0, mean1=4.0)
        obs, _ = true_model.simulate(1500, seed=1)
        model = hmm.random_init(obs, n_states=2, seed=5)
        result = model.fit(obs, n_iter=50)
        lls = result["log_likelihoods"]
        for earlier, later in zip(lls, lls[1:]):
            self.assertGreaterEqual(later, earlier - 1e-8)

    def test_recovers_known_parameters_from_synthetic_data(self):
        true_means = [0.0, 6.0]
        true_vars = [1.0, 2.0]
        true_model = hmm.GaussianHMM(true_means, true_vars,
                                      [[0.95, 0.05], [0.1, 0.9]], [0.5, 0.5])
        obs, _ = true_model.simulate(4000, seed=1)

        init_model = hmm.random_init(obs, n_states=2, seed=2)
        result = init_model.fit(obs, n_iter=200)
        fitted = result["model"]

        fitted_means_sorted = sorted(fitted.means)
        self.assertAlmostEqual(fitted_means_sorted[0], true_means[0], delta=0.3)
        self.assertAlmostEqual(fitted_means_sorted[1], true_means[1], delta=0.3)

    def test_viterbi_recovers_states_after_fitting(self):
        true_model = two_state_model(mean0=0.0, mean1=8.0)
        obs, true_states = true_model.simulate(3000, seed=1)
        init_model = hmm.random_init(obs, n_states=2, seed=3)
        fitted = init_model.fit(obs, n_iter=200)["model"]

        path = fitted.viterbi(obs)
        order = sorted(range(2), key=lambda i: fitted.means[i])
        label_map = {order[0]: 0, order[1]: 1}
        mapped_path = [label_map[s] for s in path]
        accuracy = sum(1 for a, b in zip(mapped_path, true_states) if a == b) / len(true_states)
        self.assertGreater(accuracy, 0.9)

    def test_transition_matrix_rows_sum_to_one_after_fitting(self):
        model = two_state_model()
        obs, _ = model.simulate(1000, seed=1)
        fitted = hmm.random_init(obs, 2, seed=1).fit(obs, n_iter=50)["model"]
        for row in fitted.transition:
            self.assertAlmostEqual(sum(row), 1.0, places=6)


class TestRandomInit(unittest.TestCase):
    def test_means_are_distinct_observations(self):
        obs = [float(i) for i in range(100)]
        model = hmm.random_init(obs, n_states=3, seed=1)
        self.assertEqual(len(set(model.means)), 3)

    def test_transition_rows_sum_to_one(self):
        obs = [float(i) for i in range(100)]
        model = hmm.random_init(obs, n_states=3, seed=1)
        for row in model.transition:
            self.assertAlmostEqual(sum(row), 1.0, places=8)

    def test_self_transition_favoured(self):
        obs = [float(i) for i in range(100)]
        model = hmm.random_init(obs, n_states=2, seed=1)
        for i in range(2):
            self.assertEqual(max(range(2), key=lambda j: model.transition[i][j]), i)


if __name__ == "__main__":
    unittest.main()
