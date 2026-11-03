import math
import os
import random
import unittest

import fx_application as fx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestDataLoading(unittest.TestCase):
    def test_loads_expected_count(self):
        returns = fx.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)


class TestAutocorrelation(unittest.TestCase):
    def test_lag_zero_is_one(self):
        x = [1.0, 3.0, 2.0, 5.0, 4.0]
        self.assertAlmostEqual(fx.autocorrelation(x, 0), 1.0)

    def test_known_hand_calculation(self):
        x = [1.0, 2.0, 3.0, 4.0]
        mean = 2.5
        gamma0 = sum((v - mean) ** 2 for v in x)
        gamma1 = sum((x[t] - mean) * (x[t + 1] - mean) for t in range(3))
        self.assertAlmostEqual(fx.autocorrelation(x, 1), gamma1 / gamma0)


class TestCalibrateMeanReversion(unittest.TestCase):
    def test_recovers_known_kappa_from_synthetic_exponential_decay(self):
        # Construct a synthetic squared-return-like series whose true
        # autocorrelation decays exactly exponentially with a known
        # kappa, and confirm the calibration recovers it, before
        # trusting the same procedure on real, ambiguous data. Built
        # by simulating a discretised Ornstein-Uhlenbeck-type AR(1)
        # process, whose lag-k autocorrelation is exactly phi^k.
        rng = random.Random(1)
        true_kappa = 0.05
        phi = math.exp(-true_kappa)
        theta = 2.0e-5
        x = [theta]
        for _ in range(15000):
            x.append(theta + phi * (x[-1] - theta) + rng.gauss(0, theta * 0.3))
        x = [max(v, 1e-8) for v in x]

        result = fx.calibrate_mean_reversion(x, max_lag=40)
        self.assertAlmostEqual(result["kappa"], true_kappa, delta=0.01)
        self.assertGreater(result["r_squared"], 0.9)

    def test_theta_matches_sample_mean(self):
        import statistics
        import random
        rng = random.Random(0)
        # Genuinely autocorrelated (AR(1)-type) data, so a handful of
        # lags reliably show positive autocorrelation; this test only
        # cares about the theta calculation, not the decay fit itself.
        x = [0.02]
        for _ in range(200):
            x.append(0.02 + 0.7 * (x[-1] - 0.02) + rng.gauss(0, 0.002))
        x = [max(v, 1e-6) for v in x]
        result = fx.calibrate_mean_reversion(x, max_lag=5)
        self.assertAlmostEqual(result["theta"], statistics.mean(x))

    def test_too_few_positive_lags_raises(self):
        x = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0] * 10  # alternating, mostly negative ACF
        with self.assertRaises(ValueError):
            fx.calibrate_mean_reversion(x, max_lag=2)


class TestSimulateVolatilityShock(unittest.TestCase):
    def test_all_methods_agree_with_exact_solution(self):
        theta, kappa, v0 = 3e-5, 0.01, 1.2e-4
        result = fx.simulate_volatility_shock(theta, kappa, v0, t_end=50, h=0.5)
        exact = result["exact_final_value"]
        for method in ("euler", "improved_euler", "midpoint", "rk4", "adaptive_rk4"):
            self.assertAlmostEqual(result[method]["final_value"], exact, delta=1e-6)

    def test_higher_order_methods_more_accurate(self):
        theta, kappa, v0 = 3e-5, 0.01, 1.2e-4
        result = fx.simulate_volatility_shock(theta, kappa, v0, t_end=50, h=2.0)
        self.assertLess(result["rk4"]["error"], result["euler"]["error"])
        self.assertLess(result["rk4"]["error"], result["improved_euler"]["error"])

    def test_shock_decays_toward_theta(self):
        theta, kappa, v0 = 3e-5, 0.05, 1.5e-4
        result = fx.simulate_volatility_shock(theta, kappa, v0, t_end=200, h=1.0)
        self.assertAlmostEqual(result["rk4"]["final_value"], theta, delta=1e-6)

    def test_days_to_within_5pct_matches_analytic_inversion(self):
        theta, kappa, v0 = 1.0, 0.1, 2.0
        result = fx.simulate_volatility_shock(theta, kappa, v0, t_end=100, h=0.5)
        t_star = result["days_to_within_5pct"]
        exact_at_tstar = theta + (v0 - theta) * math.exp(-kappa * t_star)
        self.assertAlmostEqual(abs(exact_at_tstar - theta), 0.05 * theta, places=6)


class TestTwoStateGenerator(unittest.TestCase):
    def test_rows_sum_to_zero(self):
        q = fx.two_state_generator_from_persistence(0.95, 0.9)
        for row in q:
            self.assertAlmostEqual(sum(row), 0.0, places=10)

    def test_symmetric_persistence_gives_symmetric_generator(self):
        q = fx.two_state_generator_from_persistence(0.98, 0.98)
        self.assertAlmostEqual(q[0][0], q[1][1])

    def test_diagonal_entries_are_negative(self):
        q = fx.two_state_generator_from_persistence(0.9, 0.95)
        self.assertLess(q[0][0], 0.0)
        self.assertLess(q[1][1], 0.0)


class TestKolmogorovForward(unittest.TestCase):
    def test_probability_conserved_over_time(self):
        import ode_solvers as ode
        q = fx.two_state_generator_from_persistence(0.95, 0.9)
        rhs = fx.kolmogorov_forward_rhs(q)
        sol = ode.solve_ivp(rhs, 0.0, [1.0, 0.0], 500, h=1.0, method="rk4")
        for p in sol["y"]:
            self.assertAlmostEqual(sum(p), 1.0, places=6)

    def test_converges_to_stationary_distribution(self):
        import ode_solvers as ode
        q = fx.two_state_generator_from_persistence(0.95, 0.9)
        pi = fx.stationary_distribution_generator(q)
        rhs = fx.kolmogorov_forward_rhs(q)
        sol = ode.solve_ivp(rhs, 0.0, [1.0, 0.0], 2000, h=1.0, method="rk4")
        for a, b in zip(sol["y"][-1], pi):
            self.assertAlmostEqual(a, b, places=5)

    def test_stationary_distribution_is_a_fixed_point(self):
        q = fx.two_state_generator_from_persistence(0.93, 0.97)
        pi = fx.stationary_distribution_generator(q)
        rhs = fx.kolmogorov_forward_rhs(q)
        derivative = rhs(0.0, pi)
        for d in derivative:
            self.assertAlmostEqual(d, 0.0, places=8)

    def test_stationary_distribution_sums_to_one(self):
        q = fx.two_state_generator_from_persistence(0.9, 0.8)
        pi = fx.stationary_distribution_generator(q)
        self.assertAlmostEqual(sum(pi), 1.0, places=8)


class TestRealDataApplication(unittest.TestCase):
    def test_short_window_calibration_is_essentially_uninformative(self):
        # The central, honest finding: fitting the simple exponential
        # decay model over just 30 lags gives a very poor fit, because
        # the true squared-return ACF does not decay meaningfully over
        # that short a window.
        returns = fx.load_log_returns(DATA_PATH)
        squared = [r ** 2 for r in returns]
        result = fx.calibrate_mean_reversion(squared, max_lag=30)
        self.assertLess(result["r_squared"], 0.1)

    def test_long_window_calibration_shows_real_but_partial_decay(self):
        returns = fx.load_log_returns(DATA_PATH)
        squared = [r ** 2 for r in returns]
        result = fx.calibrate_mean_reversion(squared, max_lag=300)
        self.assertGreater(result["kappa"], 0.0)
        self.assertGreater(result["r_squared"], 0.5)
        self.assertLess(result["r_squared"], 0.9)

    def test_long_window_half_life_is_a_plausible_number_of_months(self):
        returns = fx.load_log_returns(DATA_PATH)
        squared = [r ** 2 for r in returns]
        result = fx.calibrate_mean_reversion(squared, max_lag=300)
        self.assertGreater(result["half_life_days"], 20)
        self.assertLess(result["half_life_days"], 200)


if __name__ == "__main__":
    unittest.main()
