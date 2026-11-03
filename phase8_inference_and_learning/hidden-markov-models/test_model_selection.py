import math
import unittest

import hmm
import model_selection as ms


class TestNFreeParameters(unittest.TestCase):
    def test_two_states(self):
        # (2-1) + 2*1 + 2 + 2 = 1 + 2 + 4 = 7
        self.assertEqual(ms.n_free_parameters(2), 7)

    def test_three_states(self):
        # (3-1) + 3*2 + 3 + 3 = 2 + 6 + 6 = 14
        self.assertEqual(ms.n_free_parameters(3), 14)

    def test_formula_matches_closed_form(self):
        for k in range(2, 8):
            self.assertEqual(ms.n_free_parameters(k), k ** 2 + 2 * k - 1)


class TestFitBestOfNRestarts(unittest.TestCase):
    def test_multiple_restarts_at_least_as_good_as_one(self):
        true_model = hmm.GaussianHMM([0.0, 5.0], [1.0, 1.0], [[0.9, 0.1], [0.1, 0.9]], [0.5, 0.5])
        obs, _ = true_model.simulate(800, seed=1)

        single = ms.fit_best_of_n_restarts(obs, n_states=2, n_restarts=1, n_iter=50, seed=1)
        multi = ms.fit_best_of_n_restarts(obs, n_states=2, n_restarts=5, n_iter=50, seed=1)
        self.assertGreaterEqual(multi["log_likelihoods"][-1], single["log_likelihoods"][-1] - 1e-6)

    def test_returns_expected_keys(self):
        true_model = hmm.GaussianHMM([0.0, 5.0], [1.0, 1.0], [[0.9, 0.1], [0.1, 0.9]], [0.5, 0.5])
        obs, _ = true_model.simulate(500, seed=1)
        result = ms.fit_best_of_n_restarts(obs, n_states=2, n_restarts=2, n_iter=30, seed=1)
        self.assertIn("model", result)
        self.assertIn("log_likelihoods", result)


class TestAICBIC(unittest.TestCase):
    def test_aic_formula(self):
        self.assertAlmostEqual(ms.aic(log_likelihood=-100.0, n_states=2),
                                2 * ms.n_free_parameters(2) - 2 * (-100.0))

    def test_bic_formula(self):
        self.assertAlmostEqual(ms.bic(log_likelihood=-100.0, n_states=2, n_obs=500),
                                ms.n_free_parameters(2) * math.log(500) - 2 * (-100.0))

    def test_bic_penalizes_more_for_large_n(self):
        ll = -500.0
        gap_small_n = ms.bic(ll, 3, 20) - ms.aic(ll, 3)
        gap_large_n = ms.bic(ll, 3, 100000) - ms.aic(ll, 3)
        self.assertGreater(gap_large_n, gap_small_n)


class TestSelectNStates(unittest.TestCase):
    def test_recovers_true_number_of_states(self):
        true_model = hmm.GaussianHMM([0.0, 6.0], [1.0, 1.0],
                                      [[0.95, 0.05], [0.08, 0.92]], [0.5, 0.5])
        obs, _ = true_model.simulate(1500, seed=1)
        result = ms.select_n_states(obs, k_range=[1, 2, 3], n_restarts=2, n_iter=80, seed=1)
        self.assertEqual(result["best_bic_k"], 2)

    def test_results_length_matches_k_range(self):
        true_model = hmm.GaussianHMM([0.0, 5.0], [1.0, 1.0], [[0.9, 0.1], [0.1, 0.9]], [0.5, 0.5])
        obs, _ = true_model.simulate(400, seed=1)
        result = ms.select_n_states(obs, k_range=[1, 2, 3], n_restarts=2, n_iter=50, seed=1)
        self.assertEqual(len(result["results"]), 3)

    def test_log_likelihood_is_nondecreasing_in_k(self):
        # A K-state HMM can always mimic a (K-1)-state one, so the
        # best achievable log-likelihood should never decrease as K
        # grows, even though AIC/BIC may prefer a smaller K.
        true_model = hmm.GaussianHMM([0.0, 5.0], [1.0, 1.0], [[0.9, 0.1], [0.1, 0.9]], [0.5, 0.5])
        obs, _ = true_model.simulate(500, seed=1)
        result = ms.select_n_states(obs, k_range=[1, 2, 3], n_restarts=2, n_iter=50, seed=1)
        lls = [r["log_likelihood"] for r in result["results"]]
        for earlier, later in zip(lls, lls[1:]):
            self.assertGreaterEqual(later, earlier - 1.0)  # small slack for imperfect EM convergence


if __name__ == "__main__":
    unittest.main()
