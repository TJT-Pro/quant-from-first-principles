import math
import os
import statistics
import unittest

import first_passage as fp

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DEXUSEU_returns.csv")


class TestAbsorbingChainRuin(unittest.TestCase):
    def test_symmetric_matches_closed_form(self):
        a, b = 8, 12
        self.assertAlmostEqual(fp.absorbing_chain_ruin_probability(a, b, 0.5), b / (a + b), places=8)

    def test_symmetric_duration_matches_closed_form(self):
        a, b = 8, 12
        self.assertAlmostEqual(fp.absorbing_chain_expected_duration(a, b, 0.5), a * b, places=6)

    def test_biased_matches_simulation(self):
        a, b, p, trials = 6, 9, 0.55, 20000
        theoretical = fp.absorbing_chain_ruin_probability(a, b, p)
        outcomes = [fp.simulate_ruin(a, b, p, seed=i)[0] for i in range(trials)]
        empirical = sum(outcomes) / trials
        self.assertAlmostEqual(empirical, theoretical, delta=0.02)

    def test_duration_matches_simulation(self):
        a, b, p, trials = 6, 9, 0.55, 20000
        theoretical = fp.absorbing_chain_expected_duration(a, b, p)
        durations = [fp.simulate_ruin(a, b, p, seed=i)[1] for i in range(trials)]
        empirical = statistics.mean(durations)
        self.assertAlmostEqual(empirical, theoretical, delta=theoretical * 0.05)

    def test_probability_bounds(self):
        for a, b, p in [(3, 5, 0.4), (10, 2, 0.6)]:
            prob = fp.absorbing_chain_ruin_probability(a, b, p)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_nonpositive_boundaries_raise(self):
        with self.assertRaises(ValueError):
            fp.absorbing_chain_ruin_probability(0, 5, 0.5)
        with self.assertRaises(ValueError):
            fp.absorbing_chain_expected_duration(5, -1, 0.5)


class TestSurvivalCurve(unittest.TestCase):
    def test_starts_at_one(self):
        curve = fp.absorbing_chain_survival_curve(5, 5, 0.5, n_steps=10)
        self.assertAlmostEqual(curve[0], 1.0)

    def test_monotone_nonincreasing(self):
        curve = fp.absorbing_chain_survival_curve(5, 7, 0.5, n_steps=40)
        for a, b in zip(curve, curve[1:]):
            self.assertGreaterEqual(a, b - 1e-12)

    def test_eventually_near_zero(self):
        curve = fp.absorbing_chain_survival_curve(4, 4, 0.5, n_steps=200)
        self.assertLess(curve[-1], 0.01)


class TestSolveLinearSystem(unittest.TestCase):
    def test_simple_system(self):
        a = [[2.0, 1.0], [1.0, 3.0]]
        b = [5.0, 10.0]
        x = fp.solve_linear_system(a, b)
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(x[1], 3.0)

    def test_singular_raises(self):
        with self.assertRaises(ValueError):
            fp.solve_linear_system([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


class TestBrownianMotionRuin(unittest.TestCase):
    def test_driftless_matches_closed_form(self):
        a, b, sigma = 1.5, 2.5, 1.0
        self.assertAlmostEqual(fp.bm_ruin_probability(a, b, 0.0, sigma), b / (a + b), places=8)

    def test_driftless_duration_matches_closed_form(self):
        a, b, sigma = 1.5, 2.5, 1.0
        alpha = b / (a + b)
        expected = (alpha * a ** 2 + (1 - alpha) * b ** 2) / sigma ** 2
        self.assertAlmostEqual(fp.bm_expected_duration(a, b, 0.0, sigma), expected, places=8)

    def test_drifted_matches_monte_carlo(self):
        a, b, mu, sigma, dt, trials = 1.0, 1.5, 0.3, 1.0, 0.001, 4000
        theoretical = fp.bm_ruin_probability(a, b, mu, sigma)
        outcomes = [fp.simulate_bm_ruin(a, b, mu, sigma, dt, seed=i)[0] for i in range(trials)]
        empirical = sum(outcomes) / trials
        self.assertAlmostEqual(empirical, theoretical, delta=0.03)

    def test_drifted_duration_matches_monte_carlo(self):
        a, b, mu, sigma, dt, trials = 1.0, 1.5, 0.3, 1.0, 0.001, 4000
        theoretical = fp.bm_expected_duration(a, b, mu, sigma)
        durations = [fp.simulate_bm_ruin(a, b, mu, sigma, dt, seed=i)[1] for i in range(trials)]
        empirical = statistics.mean(durations)
        self.assertAlmostEqual(empirical, theoretical, delta=theoretical * 0.1)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            fp.bm_ruin_probability(0.0, 1.0, 0.1, 1.0)
        with self.assertRaises(ValueError):
            fp.bm_expected_duration(1.0, 1.0, 0.1, 0.0)


class TestDriftAdjustedReflection(unittest.TestCase):
    def test_zero_drift_matches_project4_style_formula(self):
        a, t, sigma = 1.2, 2.0, 1.0
        ours = fp.bm_max_exceeds(a, t, 0.0, sigma)
        expected = 2.0 * (1.0 - fp.standard_normal_cdf(a / (sigma * math.sqrt(t))))
        self.assertAlmostEqual(ours, expected, places=10)

    def test_probability_bounds(self):
        for mu in (-0.5, 0.0, 0.5):
            p = fp.bm_max_exceeds(1.0, 2.0, mu, 1.0)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_positive_drift_increases_exceedance_probability(self):
        a, t, sigma = 1.0, 2.0, 1.0
        p_no_drift = fp.bm_max_exceeds(a, t, 0.0, sigma)
        p_pos_drift = fp.bm_max_exceeds(a, t, 0.3, sigma)
        self.assertGreater(p_pos_drift, p_no_drift)

    def test_matches_monte_carlo(self):
        a, t, mu, sigma, n_steps, trials = 1.0, 1.0, 0.2, 1.0, 500, 20000
        theoretical = fp.bm_max_exceeds(a, t, mu, sigma)
        dt = t / n_steps
        hits = 0
        for i in range(trials):
            rng = __import__("random").Random(i)
            x = 0.0
            hit = False
            for _ in range(n_steps):
                x += mu * dt + rng.gauss(0.0, 1.0) * sigma * math.sqrt(dt)
                if x >= a:
                    hit = True
                    break
            if hit:
                hits += 1
        empirical = hits / trials
        self.assertAlmostEqual(empirical, theoretical, delta=0.02)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            fp.bm_max_exceeds(0.0, 1.0, 0.0, 1.0)


class TestFXApplication(unittest.TestCase):
    def test_load_log_returns_reads_real_data(self):
        returns = fp.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)

    def test_calibration_reasonable(self):
        returns = fp.load_log_returns(DATA_PATH)
        cal = fp.calibrate_gbm(returns)
        self.assertAlmostEqual(cal["mu"], 0.0, delta=0.001)
        self.assertGreater(cal["sigma"], 0.0)
        self.assertLess(cal["sigma"], 0.05)

    def test_historical_backtest_structure(self):
        returns = fp.load_log_returns(DATA_PATH)
        cal = fp.calibrate_gbm(returns)
        target = stop = 10 * cal["sigma"]
        result = fp.historical_stop_target_backtest(returns, target, stop, max_horizon=100)
        self.assertGreater(result["n_trials"], 0)
        total = result["p_hit_target"] + result["p_hit_stop"] + result["p_unresolved"]
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_model_prediction_structure(self):
        returns = fp.load_log_returns(DATA_PATH)
        cal = fp.calibrate_gbm(returns)
        target = stop = 10 * cal["sigma"]
        result = fp.model_stop_target_prediction(target, stop, cal["mu"], cal["sigma"])
        self.assertAlmostEqual(result["p_hit_target"] + result["p_hit_stop"], 1.0, places=6)
        self.assertGreater(result["expected_duration"], 0.0)

    def test_symmetric_stop_target_near_symmetric_model_probabilities(self):
        # With near-zero drift and equal target/stop distances, the
        # model should predict close to a 50/50 split.
        returns = fp.load_log_returns(DATA_PATH)
        cal = fp.calibrate_gbm(returns)
        target = stop = 8 * cal["sigma"]
        result = fp.model_stop_target_prediction(target, stop, cal["mu"], cal["sigma"])
        self.assertAlmostEqual(result["p_hit_target"], 0.5, delta=0.02)

    def test_historical_unresolved_fraction_exceeds_model(self):
        # The central empirical finding: real price paths leave a
        # symmetric stop/target bracket unresolved within a finite
        # horizon more often than the calibrated constant-volatility
        # model predicts.
        returns = fp.load_log_returns(DATA_PATH)
        cal = fp.calibrate_gbm(returns)
        target = stop = 10 * cal["sigma"]
        max_horizon = 200
        historical = fp.historical_stop_target_backtest(returns, target, stop, max_horizon)

        rng = __import__("random").Random(1)
        trials = 4000
        unresolved = 0
        for i in range(trials):
            r = __import__("random").Random(i)
            x = 0.0
            resolved = False
            for _ in range(max_horizon):
                x += cal["mu"] + r.gauss(0.0, 1.0) * cal["sigma"]
                if x >= target or x <= -stop:
                    resolved = True
                    break
            if not resolved:
                unresolved += 1
        model_unresolved = unresolved / trials

        self.assertGreater(historical["p_unresolved"], model_unresolved)


if __name__ == "__main__":
    unittest.main()
