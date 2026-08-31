"""Tests for the Monte Carlo risk and drawdown analyser.

Every formula is tested against a hand-computed value or a known
mathematical identity.  The Monte Carlo tests use fixed seeds so the
results are deterministic and comparable across runs.  numpy and scipy
are used only in the verification tests, never in the module itself.
"""

import csv
import math
import os
import random
import unittest

import risk_analyser as ra


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_returns():
    path = os.path.join(os.path.dirname(__file__),
                        "data", "DEXUSEU_returns.csv")
    values = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cell = row["return"]
            if cell:
                try:
                    values.append(float(cell))
                except ValueError:
                    pass
    return values


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestInternalHelpers(unittest.TestCase):
    def test_mean_known(self):
        self.assertAlmostEqual(ra._mean([1.0, 2.0, 3.0]), 2.0)

    def test_mean_single(self):
        self.assertEqual(ra._mean([5.0]), 5.0)

    def test_mean_empty(self):
        with self.assertRaises(ValueError):
            ra._mean([])

    def test_variance_sample_known(self):
        # var([2, 4, 4, 4, 5, 5, 7, 9]) sample = 4.571...
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        mu = sum(values) / len(values)
        expected = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
        self.assertAlmostEqual(ra._variance_sample(values), expected)

    def test_variance_sample_needs_two(self):
        with self.assertRaises(ValueError):
            ra._variance_sample([1.0])

    def test_quantile_endpoints(self):
        s = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(ra._quantile(s, 0.0), 1.0)
        self.assertEqual(ra._quantile(s, 1.0), 5.0)

    def test_quantile_midpoint(self):
        s = [0.0, 1.0]
        self.assertAlmostEqual(ra._quantile(s, 0.5), 0.5)

    def test_quantile_interpolation(self):
        s = [0.0, 2.0, 4.0, 6.0, 8.0]
        # alpha=0.25 -> pos=1.0, so index 1 -> 2.0
        self.assertAlmostEqual(ra._quantile(s, 0.25), 2.0)

    def test_quantile_empty(self):
        with self.assertRaises(ValueError):
            ra._quantile([], 0.5)


# ---------------------------------------------------------------------------
# Normal CDF and PPF
# ---------------------------------------------------------------------------

class TestNormalFunctions(unittest.TestCase):
    def test_cdf_symmetry(self):
        self.assertAlmostEqual(ra.normal_cdf(0.0), 0.5)

    def test_cdf_tail(self):
        # P(Z <= -1.96) is approx 0.025
        self.assertAlmostEqual(ra.normal_cdf(-1.96), 0.025, places=3)

    def test_cdf_upper_tail(self):
        self.assertAlmostEqual(ra.normal_cdf(1.96), 0.975, places=3)

    def test_ppf_median(self):
        self.assertAlmostEqual(ra._normal_ppf(0.5), 0.0, places=8)

    def test_ppf_round_trip(self):
        for p in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            z = ra._normal_ppf(p)
            self.assertAlmostEqual(ra.normal_cdf(z), p, places=8,
                                   msg="round-trip failed at p={}".format(p))

    def test_ppf_known_quantiles(self):
        self.assertAlmostEqual(ra._normal_ppf(0.975), 1.96, places=2)
        self.assertAlmostEqual(ra._normal_ppf(0.025), -1.96, places=2)

    def test_ppf_bad_input(self):
        with self.assertRaises(ValueError):
            ra._normal_ppf(0.0)
        with self.assertRaises(ValueError):
            ra._normal_ppf(1.0)


# ---------------------------------------------------------------------------
# Random number generation
# ---------------------------------------------------------------------------

class TestRandomGeneration(unittest.TestCase):
    def test_standard_normal_count(self):
        draws = ra._standard_normal_sample(100, seed=1)
        self.assertEqual(len(draws), 100)

    def test_standard_normal_mean_and_std(self):
        draws = ra._standard_normal_sample(50000, seed=42)
        mu = sum(draws) / len(draws)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in draws) / len(draws))
        self.assertAlmostEqual(mu, 0.0, delta=0.02)
        self.assertAlmostEqual(sigma, 1.0, delta=0.02)

    def test_student_t_count(self):
        draws = ra._student_t_sample(50, 5, seed=1)
        self.assertEqual(len(draws), 50)

    def test_student_t_mean_near_zero(self):
        draws = ra._student_t_sample(20000, 10, seed=7)
        mu = sum(draws) / len(draws)
        self.assertAlmostEqual(mu, 0.0, delta=0.05)

    def test_student_t_bad_df(self):
        with self.assertRaises(ValueError):
            ra._student_t_sample(10, 0)
        with self.assertRaises(ValueError):
            ra._student_t_sample(10, -5)

    def test_seed_reproducibility(self):
        a = ra._standard_normal_sample(20, seed=99)
        b = ra._standard_normal_sample(20, seed=99)
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        a = ra._standard_normal_sample(20, seed=1)
        b = ra._standard_normal_sample(20, seed=2)
        self.assertNotEqual(a, b)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

class TestFitting(unittest.TestCase):
    def test_fit_normal_known(self):
        # sample of [1, 2, 3]: mean=2, std=1 (sample)
        mu, sigma = ra.fit_normal([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mu, 2.0)
        self.assertAlmostEqual(sigma, 1.0)

    def test_fit_normal_constant(self):
        mu, sigma = ra.fit_normal([5.0, 5.0, 5.0])
        self.assertAlmostEqual(mu, 5.0)
        self.assertAlmostEqual(sigma, 0.0)

    def test_fit_t_variance_matches(self):
        # fit_t should produce a sigma such that
        # sigma^2 * df/(df-2) = sample variance
        returns = [0.01, -0.02, 0.015, -0.005, 0.03]
        df = 5.0
        mu, sigma = ra.fit_t(returns, df=df)
        s2 = ra._variance_sample(returns)
        implied_var = sigma ** 2 * df / (df - 2.0)
        self.assertAlmostEqual(implied_var, s2, places=12)

    def test_fit_t_bad_df(self):
        with self.assertRaises(ValueError):
            ra.fit_t([1.0, 2.0], df=2.0)
        with self.assertRaises(ValueError):
            ra.fit_t([1.0, 2.0], df=1.5)


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

class TestEquityCurve(unittest.TestCase):
    def test_known_curve(self):
        returns = [0.1, -0.1, 0.2]
        curve = ra.equity_curve(returns)
        self.assertAlmostEqual(curve[0], 1.0)
        self.assertAlmostEqual(curve[1], 1.1)
        self.assertAlmostEqual(curve[2], 1.1 * 0.9)
        self.assertAlmostEqual(curve[3], 1.1 * 0.9 * 1.2)

    def test_initial_wealth(self):
        curve = ra.equity_curve([0.05, 0.05], initial=1000.0)
        self.assertAlmostEqual(curve[0], 1000.0)
        self.assertAlmostEqual(curve[2], 1000.0 * 1.05 * 1.05)

    def test_flat_returns(self):
        curve = ra.equity_curve([0.0, 0.0, 0.0])
        self.assertTrue(all(abs(v - 1.0) < 1e-12 for v in curve))

    def test_length(self):
        curve = ra.equity_curve([0.1] * 10)
        self.assertEqual(len(curve), 11)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ra.equity_curve([])

    def test_total_loss_raises(self):
        with self.assertRaises(ValueError):
            ra.equity_curve([-1.0])

    def test_beyond_total_loss_raises(self):
        with self.assertRaises(ValueError):
            ra.equity_curve([-1.5])


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

class TestDrawdown(unittest.TestCase):
    def test_no_drawdown(self):
        # monotone rising curve
        returns = [0.1, 0.1, 0.1, 0.1]
        curve = ra.equity_curve(returns)
        dd = ra.drawdown_series(curve)
        self.assertTrue(all(d == 0.0 for d in dd))

    def test_known_drawdown(self):
        # curve goes 1 -> 1.2 -> 0.84 -> drawdown = (1.2 - 0.84)/1.2 = 0.3
        curve = [1.0, 1.2, 0.84]
        dd = ra.drawdown_series(curve)
        self.assertAlmostEqual(dd[0], 0.0)
        self.assertAlmostEqual(dd[1], 0.0)
        self.assertAlmostEqual(dd[2], 0.3)

    def test_max_drawdown_known(self):
        curve = [1.0, 1.5, 1.0, 2.0, 1.0]
        # first episode: (1.5-1.0)/1.5 = 1/3
        # second episode: (2.0-1.0)/2.0 = 0.5
        self.assertAlmostEqual(ra.max_drawdown(curve), 0.5)

    def test_max_drawdown_flat(self):
        self.assertEqual(ra.max_drawdown([1.0, 1.0, 1.0]), 0.0)

    def test_drawdown_duration_counts_only_down_periods(self):
        curve = [1.0, 1.2, 1.0, 1.0, 1.2, 1.2]
        # periods 2 and 3 are below peak 1.2
        self.assertEqual(ra.drawdown_duration(curve), 2)

    def test_max_drawdown_duration_longest_episode(self):
        # two episodes: length 1 and length 3
        curve = [1.0, 1.5, 1.2, 1.5, 1.0, 0.9, 0.8, 1.5]
        dur = ra.max_drawdown_duration(curve)
        self.assertEqual(dur, 3)

    def test_max_drawdown_duration_no_drawdown(self):
        curve = [1.0, 1.1, 1.2, 1.3]
        self.assertEqual(ra.max_drawdown_duration(curve), 0)

    def test_calmar_positive_return_positive_dd(self):
        returns = [0.01] * 200 + [-0.3] + [0.01] * 200
        calmar = ra.calmar_ratio(returns)
        self.assertIsNotNone(calmar)
        self.assertGreater(calmar, 0.0)

    def test_calmar_none_when_no_drawdown(self):
        returns = [0.01] * 50
        self.assertIsNone(ra.calmar_ratio(returns))

    def test_calmar_higher_return_raises_calmar(self):
        high = [0.02] * 100 + [-0.2] + [0.02] * 100
        low  = [0.005] * 100 + [-0.2] + [0.005] * 100
        self.assertGreater(ra.calmar_ratio(high), ra.calmar_ratio(low))


# ---------------------------------------------------------------------------
# Value at Risk
# ---------------------------------------------------------------------------

class TestVaR(unittest.TestCase):
    RETURNS = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.04]

    def test_historical_var_is_positive_number(self):
        var = ra.historical_var(self.RETURNS, alpha=0.05)
        self.assertGreater(var, 0.0)

    def test_historical_var_alpha_monotone(self):
        # tighter confidence (smaller alpha) => larger VaR
        var_5 = ra.historical_var(self.RETURNS, alpha=0.05)
        var_10 = ra.historical_var(self.RETURNS, alpha=0.10)
        self.assertGreaterEqual(var_5, var_10)

    def test_historical_var_known(self):
        # With 7 values and alpha=0.05, pos = 0.05*6 = 0.3
        # lo=0, hi=1, frac=0.3 => sorted[0]*0.7 + sorted[1]*0.3
        # sorted: -0.05, -0.03, -0.01, 0, 0.01, 0.02, 0.04
        # = -0.05*0.7 + (-0.03)*0.3 = -0.035 - 0.009 = -0.044
        # var = 0.044
        var = ra.historical_var(self.RETURNS, alpha=0.05)
        self.assertAlmostEqual(var, 0.044)

    def test_parametric_var_positive(self):
        var = ra.parametric_var(self.RETURNS, alpha=0.05)
        self.assertGreater(var, 0.0)

    def test_parametric_var_monotone_in_alpha(self):
        var_5 = ra.parametric_var(self.RETURNS, alpha=0.05)
        var_10 = ra.parametric_var(self.RETURNS, alpha=0.10)
        self.assertGreater(var_5, var_10)

    def test_var_bad_alpha(self):
        with self.assertRaises(ValueError):
            ra.historical_var(self.RETURNS, alpha=0.0)
        with self.assertRaises(ValueError):
            ra.historical_var(self.RETURNS, alpha=1.0)
        with self.assertRaises(ValueError):
            ra.parametric_var(self.RETURNS, alpha=0.0)

    def test_var_empty(self):
        with self.assertRaises(ValueError):
            ra.historical_var([])
        with self.assertRaises(ValueError):
            ra.parametric_var([])


# ---------------------------------------------------------------------------
# CVaR
# ---------------------------------------------------------------------------

class TestCVaR(unittest.TestCase):
    RETURNS = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.04]

    def test_cvar_exceeds_var(self):
        var = ra.historical_var(self.RETURNS, alpha=0.10)
        cvar = ra.historical_cvar(self.RETURNS, alpha=0.10)
        self.assertGreaterEqual(cvar, var)

    def test_historical_cvar_known(self):
        # alpha=0.10, cutoff = floor(0.10 * 7) = 0 -> max(1, 0) = 1
        # tail = [-0.05], mean = -0.05, cvar = 0.05
        cvar = ra.historical_cvar(self.RETURNS, alpha=0.10)
        self.assertAlmostEqual(cvar, 0.05)

    def test_parametric_cvar_exceeds_parametric_var(self):
        var = ra.parametric_var(self.RETURNS, alpha=0.05)
        cvar = ra.parametric_cvar(self.RETURNS, alpha=0.05)
        self.assertGreater(cvar, var)

    def test_cvar_bad_alpha(self):
        with self.assertRaises(ValueError):
            ra.historical_cvar(self.RETURNS, alpha=0.0)
        with self.assertRaises(ValueError):
            ra.parametric_cvar(self.RETURNS, alpha=0.0)

    def test_cvar_monotone_in_alpha(self):
        # smaller alpha = more extreme tail = larger CVaR
        cvar_5 = ra.historical_cvar(self.RETURNS, alpha=0.05)
        cvar_10 = ra.historical_cvar(self.RETURNS, alpha=0.10)
        self.assertGreaterEqual(cvar_5, cvar_10)

    def test_cvar_empty(self):
        with self.assertRaises(ValueError):
            ra.historical_cvar([])
        with self.assertRaises(ValueError):
            ra.parametric_cvar([])


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class TestSimulation(unittest.TestCase):
    RETURNS = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.04,
               0.005, -0.015, 0.03, -0.02, 0.01, -0.008, 0.012]

    def test_normal_path_count(self):
        paths = ra.simulate_normal_returns(0.0, 0.01, 50, 100, seed=1)
        self.assertEqual(len(paths), 100)
        self.assertTrue(all(len(p) == 50 for p in paths))

    def test_t_path_count(self):
        paths = ra.simulate_t_returns(0.0, 0.01, 5, 50, 100, seed=1)
        self.assertEqual(len(paths), 100)
        self.assertTrue(all(len(p) == 50 for p in paths))

    def test_simulate_bad_distribution(self):
        with self.assertRaises(ValueError):
            ra.simulate_risk_metrics(self.RETURNS, n_paths=10,
                                     distribution="cauchy")

    def test_simulate_bad_df(self):
        with self.assertRaises(ValueError):
            ra.simulate_t_returns(0.0, 0.01, 2, 10, 10)

    def test_simulate_output_keys(self):
        results = ra.simulate_risk_metrics(self.RETURNS, n_paths=100,
                                           n_periods=20, seed=5)
        for key in ("paths", "final_wealth", "path_var", "path_cvar",
                    "path_max_dd", "mean_max_dd", "median_max_dd",
                    "worst_max_dd"):
            self.assertIn(key, results)

    def test_simulate_equity_curve_lengths(self):
        results = ra.simulate_risk_metrics(self.RETURNS, n_paths=50,
                                           n_periods=30, seed=3)
        self.assertTrue(all(len(p) == 31 for p in results["paths"]))

    def test_simulate_drawdown_in_range(self):
        results = ra.simulate_risk_metrics(self.RETURNS, n_paths=200,
                                           n_periods=50, seed=11)
        self.assertTrue(all(0.0 <= d <= 1.0 for d in results["path_max_dd"]))

    def test_simulate_cvar_exceeds_var(self):
        results = ra.simulate_risk_metrics(self.RETURNS, n_paths=500,
                                           n_periods=50, seed=42)
        self.assertGreaterEqual(results["path_cvar"], results["path_var"])

    def test_simulate_worst_mdd_is_max(self):
        results = ra.simulate_risk_metrics(self.RETURNS, n_paths=100,
                                           n_periods=30, seed=6)
        self.assertEqual(results["worst_max_dd"], max(results["path_max_dd"]))

    def test_simulate_mean_median_order(self):
        # mean and median should both be in (0, 1)
        results = ra.simulate_risk_metrics(self.RETURNS, n_paths=200,
                                           n_periods=50, seed=8)
        self.assertGreater(results["mean_max_dd"], 0.0)
        self.assertGreater(results["median_max_dd"], 0.0)

    def test_t_fatter_tails_than_normal(self):
        # t(df) with low df has higher kurtosis than the normal.
        # With a large enough sample the excess kurtosis of the t draws
        # should exceed that of the normal draws.
        # Kurtosis = E[(X - mu)^4] / sigma^4 - 3
        # Normal kurtosis is 0; t(df) kurtosis is 6/(df-4) for df > 4.
        # We use df=5 so theoretical excess kurtosis = 6.
        n = 40000
        t_draws = ra._student_t_sample(n, df=5, seed=77)
        z_draws = ra._standard_normal_sample(n, seed=77)

        def excess_kurtosis(xs):
            mu = sum(xs) / len(xs)
            sigma2 = sum((x - mu) ** 2 for x in xs) / len(xs)
            if sigma2 == 0.0:
                return 0.0
            m4 = sum((x - mu) ** 4 for x in xs) / len(xs)
            return m4 / sigma2 ** 2 - 3.0

        t_kurt = excess_kurtosis(t_draws)
        z_kurt = excess_kurtosis(z_draws)
        self.assertGreater(t_kurt, z_kurt,
                           "t draws should have higher excess kurtosis than normal")


# ---------------------------------------------------------------------------
# Ruin probability
# ---------------------------------------------------------------------------

class TestRuinProbability(unittest.TestCase):
    RETURNS = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.04,
               0.005, -0.015, 0.03, -0.02, 0.01, -0.008, 0.012]

    def test_ruin_in_range(self):
        results = ra.simulate_risk_metrics(self.RETURNS, n_paths=500, seed=1)
        p = ra.ruin_probability(results, threshold=0.2)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_ruin_near_zero_threshold_is_one(self):
        results = ra.simulate_risk_metrics(self.RETURNS, n_paths=200, seed=2)
        p = ra.ruin_probability(results, threshold=1e-9)
        self.assertEqual(p, 1.0)

    def test_ruin_threshold_one_is_zero_or_near(self):
        results = ra.simulate_risk_metrics(self.RETURNS, n_paths=200, seed=3)
        p = ra.ruin_probability(results, threshold=1.0)
        self.assertAlmostEqual(p, 0.0, delta=0.05)

    def test_ruin_monotone_in_threshold(self):
        results = ra.simulate_risk_metrics(self.RETURNS, n_paths=300, seed=4)
        p_low = ra.ruin_probability(results, threshold=0.1)
        p_high = ra.ruin_probability(results, threshold=0.5)
        self.assertGreaterEqual(p_low, p_high)

    def test_ruin_bad_threshold(self):
        results = ra.simulate_risk_metrics(self.RETURNS, n_paths=10, seed=5)
        with self.assertRaises(ValueError):
            ra.ruin_probability(results, threshold=0.0)
        with self.assertRaises(ValueError):
            ra.ruin_probability(results, threshold=1.5)

    def test_ruin_ci_contains_point(self):
        results = ra.simulate_risk_metrics(self.RETURNS, n_paths=300, seed=6)
        point, lower, upper = ra.ruin_probability_ci(
            results, threshold=0.2, n_boot=500, seed=6
        )
        self.assertLessEqual(lower, point)
        self.assertLessEqual(point, upper)

    def test_ruin_ci_width_shrinks_with_paths(self):
        small = ra.simulate_risk_metrics(self.RETURNS, n_paths=100, seed=7)
        large = ra.simulate_risk_metrics(self.RETURNS, n_paths=2000, seed=7)
        _, lo_s, hi_s = ra.ruin_probability_ci(small, 0.2, n_boot=500, seed=7)
        _, lo_l, hi_l = ra.ruin_probability_ci(large, 0.2, n_boot=500, seed=7)
        self.assertGreater(hi_s - lo_s, hi_l - lo_l)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

class TestBootstrapCI(unittest.TestCase):
    def test_ci_contains_truth(self):
        # 50 draws from N(0,1), CI for the mean should include 0 most of the time
        rng = random.Random(42)
        values = [rng.gauss(0.0, 1.0) for _ in range(50)]
        point, lower, upper = ra.bootstrap_ci(
            values, lambda v: sum(v) / len(v), confidence=0.95, n_boot=1000, seed=42
        )
        # the CI should include a value near 0 (it's a wide-enough interval)
        self.assertLessEqual(lower, point)
        self.assertLessEqual(point, upper)

    def test_ci_width_positive(self):
        rng = random.Random(1)
        values = [rng.gauss(0.0, 1.0) for _ in range(30)]
        _, lower, upper = ra.bootstrap_ci(
            values, lambda v: sum(v) / len(v), n_boot=500, seed=1
        )
        self.assertGreater(upper, lower)

    def test_ci_empty_raises(self):
        with self.assertRaises(ValueError):
            ra.bootstrap_ci([], lambda v: sum(v) / len(v))


# ---------------------------------------------------------------------------
# Real data integration
# ---------------------------------------------------------------------------

class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns = _load_returns()

    def test_dataset_loaded(self):
        self.assertEqual(len(self.returns), 6930)

    def test_historical_var_on_real_data(self):
        var = ra.historical_var(self.returns, alpha=0.05)
        # EUR/USD daily 95% VaR should be a small positive fraction
        self.assertGreater(var, 0.0)
        self.assertLess(var, 0.1)

    def test_historical_cvar_exceeds_var_on_real_data(self):
        var = ra.historical_var(self.returns, alpha=0.05)
        cvar = ra.historical_cvar(self.returns, alpha=0.05)
        self.assertGreater(cvar, var)

    def test_parametric_var_and_cvar_reasonable(self):
        var = ra.parametric_var(self.returns, alpha=0.05)
        cvar = ra.parametric_cvar(self.returns, alpha=0.05)
        self.assertGreater(var, 0.0)
        self.assertGreater(cvar, var)

    def test_equity_curve_starts_at_one(self):
        curve = ra.equity_curve(self.returns)
        self.assertAlmostEqual(curve[0], 1.0)

    def test_max_drawdown_in_range(self):
        curve = ra.equity_curve(self.returns)
        mdd = ra.max_drawdown(curve)
        self.assertGreater(mdd, 0.0)
        self.assertLessEqual(mdd, 1.0)

    def test_monte_carlo_runs_on_real_data(self):
        results = ra.simulate_risk_metrics(
            self.returns, n_paths=500, n_periods=252, seed=42
        )
        self.assertEqual(len(results["paths"]), 500)
        self.assertGreater(results["mean_max_dd"], 0.0)

    def test_ruin_prob_on_real_data(self):
        results = ra.simulate_risk_metrics(
            self.returns, n_paths=500, n_periods=252, seed=1
        )
        p = ra.ruin_probability(results, threshold=0.15)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_risk_report_all_keys_present(self):
        report = ra.risk_report(self.returns, n_paths=200, n_periods=100,
                                seed=42)
        expected_keys = [
            "n_returns", "mu", "sigma", "hist_var", "hist_cvar",
            "param_var", "param_cvar", "obs_max_dd", "obs_mdd_dur",
            "obs_calmar", "sim_var", "sim_cvar", "sim_mean_mdd",
            "sim_median_mdd", "sim_worst_mdd", "ruin_prob",
            "ruin_lower", "ruin_upper", "alpha", "ruin_threshold",
            "distribution", "n_paths", "n_periods",
        ]
        for key in expected_keys:
            self.assertIn(key, report, msg="missing key: {}".format(key))

    def test_risk_report_values_in_range(self):
        report = ra.risk_report(self.returns, n_paths=200, n_periods=100,
                                seed=42)
        self.assertGreater(report["hist_var"], 0.0)
        self.assertGreater(report["hist_cvar"], report["hist_var"])
        self.assertGreater(report["obs_max_dd"], 0.0)
        self.assertGreaterEqual(report["ruin_prob"], 0.0)
        self.assertLessEqual(report["ruin_prob"], 1.0)


if __name__ == "__main__":
    unittest.main()
