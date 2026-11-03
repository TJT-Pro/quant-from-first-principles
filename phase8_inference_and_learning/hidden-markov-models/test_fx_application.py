import os
import statistics
import unittest

import hmm
import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestDataLoading(unittest.TestCase):
    def test_loads_expected_count(self):
        returns = fx.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)


class TestFitRegimeModel(unittest.TestCase):
    def test_state_order_matches_ascending_variance(self):
        true_model = hmm.GaussianHMM([0.0, 0.0], [1.0, 9.0],
                                      [[0.9, 0.1], [0.1, 0.9]], [0.5, 0.5])
        obs, _ = true_model.simulate(1500, seed=1)
        result = fx.fit_regime_model(obs, n_states=2, n_restarts=3, seed=1)
        order = result["state_order_low_to_high_vol"]
        model = result["model"]
        self.assertLess(model.variances[order[0]], model.variances[order[1]])

    def test_returns_expected_keys(self):
        true_model = hmm.GaussianHMM([0.0, 5.0], [1.0, 1.0],
                                      [[0.9, 0.1], [0.1, 0.9]], [0.5, 0.5])
        obs, _ = true_model.simulate(500, seed=1)
        result = fx.fit_regime_model(obs, n_states=2, n_restarts=2, seed=1)
        self.assertIn("model", result)
        self.assertIn("log_likelihoods", result)
        self.assertIn("state_order_low_to_high_vol", result)


class TestStationaryDistribution(unittest.TestCase):
    def test_sums_to_one(self):
        transition = [[0.9, 0.1], [0.2, 0.8]]
        pi = fx._stationary_distribution(transition)
        self.assertAlmostEqual(sum(pi), 1.0, places=8)

    def test_is_a_fixed_point(self):
        transition = [[0.9, 0.1], [0.2, 0.8]]
        pi = fx._stationary_distribution(transition)
        next_pi = [sum(pi[i] * transition[i][j] for i in range(2)) for j in range(2)]
        for a, b in zip(pi, next_pi):
            self.assertAlmostEqual(a, b, places=6)

    def test_matches_known_two_state_closed_form(self):
        # For P=[[1-a,a],[b,1-b]], stationary pi = (b/(a+b), a/(a+b)).
        a, b = 0.1, 0.3
        transition = [[1 - a, a], [b, 1 - b]]
        pi = fx._stationary_distribution(transition)
        self.assertAlmostEqual(pi[0], b / (a + b), places=6)
        self.assertAlmostEqual(pi[1], a / (a + b), places=6)


class TestRegimeSummary(unittest.TestCase):
    def test_fractions_sum_to_one(self):
        true_model = hmm.GaussianHMM([0.0, 5.0], [1.0, 4.0],
                                      [[0.9, 0.1], [0.1, 0.9]], [0.5, 0.5])
        obs, _ = true_model.simulate(1000, seed=1)
        result = fx.fit_regime_model(obs, n_states=2, n_restarts=3, seed=1)
        model = result["model"]
        order = result["state_order_low_to_high_vol"]
        path = model.viterbi(obs)
        summary = fx.regime_summary(model, path, order)
        total_fraction = sum(s["viterbi_fraction"] for s in summary)
        self.assertAlmostEqual(total_fraction, 1.0, places=8)

    def test_ranks_are_ordered_by_increasing_sd(self):
        true_model = hmm.GaussianHMM([0.0, 5.0], [1.0, 9.0],
                                      [[0.9, 0.1], [0.1, 0.9]], [0.5, 0.5])
        obs, _ = true_model.simulate(1000, seed=1)
        result = fx.fit_regime_model(obs, n_states=2, n_restarts=3, seed=1)
        model = result["model"]
        order = result["state_order_low_to_high_vol"]
        path = model.viterbi(obs)
        summary = fx.regime_summary(model, path, order)
        sds = [s["sd"] for s in summary]
        self.assertEqual(sds, sorted(sds))


class TestRegimeDurationStatistics(unittest.TestCase):
    def test_episode_lengths_sum_to_path_length(self):
        path = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
        order = [0, 1]
        durations = fx.regime_duration_statistics(path, order)
        total = sum(d["n_episodes"] * d["mean_duration"] for d in durations.values())
        self.assertAlmostEqual(total, len(path))

    def test_single_regime_path(self):
        path = [0] * 20
        order = [0, 1]
        durations = fx.regime_duration_statistics(path, order)
        self.assertEqual(durations[0]["n_episodes"], 1)
        self.assertEqual(durations[0]["max_duration"], 20)


class TestRealDataApplication(unittest.TestCase):
    def test_two_state_fit_separates_volatility(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.fit_regime_model(returns, n_states=2, n_restarts=3, seed=1)
        model = result["model"]
        order = result["state_order_low_to_high_vol"]
        low_var, high_var = model.variances[order[0]], model.variances[order[1]]
        # A real, substantial volatility separation, not two nearly
        # identical states EM failed to distinguish.
        self.assertGreater(high_var / low_var, 1.3)

    def test_regimes_are_persistent_not_noisy_flickering(self):
        # Both regimes should have a strong self-transition
        # probability (long expected duration), the defining feature
        # that makes them genuine "regimes" rather than day-to-day
        # noise reclassification.
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.fit_regime_model(returns, n_states=2, n_restarts=3, seed=1)
        model = result["model"]
        for i in range(2):
            self.assertGreater(model.transition[i][i], 0.95)


if __name__ == "__main__":
    unittest.main()
