import math
import os
import random
import statistics
import unittest

import fx_application as fx
import metropolis_hastings as mh

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DEXUSEU_returns.csv")


class TestDataLoading(unittest.TestCase):
    def test_loads_expected_count(self):
        returns = fx.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)


class TestSequentialBetaBinomial(unittest.TestCase):
    def test_checkpoints_returned_in_order(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.sequential_beta_binomial_updating(returns, checkpoints=[100, 1000, 6930])
        ns = [c["n"] for c in result["checkpoints"]]
        self.assertEqual(ns, [100, 1000, 6930])

    def test_credible_interval_narrows_over_time(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.sequential_beta_binomial_updating(returns, checkpoints=[100, 1000, 6930])
        widths = [c["ci_high"] - c["ci_low"] for c in result["checkpoints"]]
        for earlier, later in zip(widths, widths[1:]):
            self.assertGreater(earlier, later)

    def test_final_posterior_mean_close_to_one_half(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.sequential_beta_binomial_updating(returns, checkpoints=[6930])
        self.assertAlmostEqual(result["checkpoints"][0]["mean"], 0.5, delta=0.03)


class TestBayesianDrift(unittest.TestCase):
    def test_posterior_mean_matches_sample_mean_with_uninformative_prior(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.bayesian_drift_estimate(returns, prior_mu0=0.0, prior_tau0_sq=1.0)
        self.assertAlmostEqual(result["posterior_mean"], result["sample_mean"], places=6)

    def test_posterior_sd_is_tiny_relative_to_zero(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.bayesian_drift_estimate(returns)
        # The posterior should be many posterior standard deviations
        # away from ruling out zero drift as implausible, i.e. zero
        # should sit well within a couple of posterior SDs.
        self.assertLess(abs(result["posterior_mean"]) / result["posterior_sd"], 1.0)


class TestLargeMoveIndicator(unittest.TestCase):
    def test_fraction_matches_requested_quantile(self):
        returns = fx.load_log_returns(DATA_PATH)
        indicators = fx.large_move_indicator(returns, quantile=0.95)
        fraction = sum(indicators) / len(indicators)
        self.assertAlmostEqual(fraction, 0.05, delta=0.01)

    def test_output_is_binary(self):
        returns = fx.load_log_returns(DATA_PATH)
        indicators = fx.large_move_indicator(returns, quantile=0.9)
        self.assertTrue(all(v in (0, 1) for v in indicators))


class TestGammaPoissonPosteriorPredictiveCheck(unittest.TestCase):
    def test_structure(self):
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.gamma_poisson_posterior_predictive_check(returns, quantile=0.95, window=20)
        self.assertGreater(result["n_windows"], 100)
        self.assertGreater(result["posterior_rate_mean"], 0.0)

    def test_reveals_overdispersion_relative_to_poisson(self):
        # The central finding: empirical variance across windows
        # should exceed the empirical mean by a clear margin, the
        # Bayesian-application counterpart to the Poisson dispersion
        # test result in stochastic-processes/project3.
        returns = fx.load_log_returns(DATA_PATH)
        result = fx.gamma_poisson_posterior_predictive_check(returns, quantile=0.95, window=20)
        self.assertGreater(result["empirical_variance"], 1.5 * result["empirical_mean"])

    def test_poisson_simulated_data_does_not_show_overdispersion(self):
        # A sanity check that the check itself isn't just always
        # flagging overdispersion regardless of the data: genuinely
        # Poisson-generated indicator data should NOT trigger it.
        # Small jitter is added so no two magnitudes tie exactly; a
        # first version of this test used exactly repeated values,
        # which put the 95th-percentile threshold precisely on the
        # boundary value itself and made large_move_indicator's
        # strict ">" comparison select nothing at all (a test
        # construction artifact, not a production bug: real
        # continuous-valued return data never has this kind of exact
        # tie in the first place).
        rng = random.Random(1)
        n_windows, window, rate = 300, 20, 1.0
        synthetic_returns = []
        for _ in range(n_windows):
            for _ in range(window):
                is_event = rng.random() < rate / window
                base = 0.1 if is_event else 0.001
                synthetic_returns.append(base + rng.uniform(-1e-5, 1e-5))
        result = fx.gamma_poisson_posterior_predictive_check(synthetic_returns, quantile=0.95, window=window)
        # Loose bound: should be much closer to equidispersion than
        # the real EUR/USD data's ~2x ratio.
        self.assertLess(result["empirical_variance"] / result["empirical_mean"], 1.6)


class TestStudentTLogPdf(unittest.TestCase):
    def test_matches_normal_in_high_df_limit(self):
        # As df -> infinity, the Student-t converges to a Normal with
        # the same scale as its standard deviation.
        x, scale = 0.5, 1.0
        t_density = fx.student_t_log_pdf(x, df=100000.0, scale=scale)
        normal_density = -0.5 * math.log(2 * math.pi * scale ** 2) - x ** 2 / (2 * scale ** 2)
        self.assertAlmostEqual(t_density, normal_density, places=3)

    def test_invalid_parameters_give_negative_infinity(self):
        self.assertEqual(fx.student_t_log_pdf(0.0, df=-1.0, scale=1.0), float("-inf"))
        self.assertEqual(fx.student_t_log_pdf(0.0, df=5.0, scale=-1.0), float("-inf"))

    def test_heavier_tails_at_low_df(self):
        # At a point far from zero, a low-df Student-t should assign
        # more density than a high-df one (fatter tails).
        x = 5.0
        low_df_density = fx.student_t_log_pdf(x, df=3.0, scale=1.0)
        high_df_density = fx.student_t_log_pdf(x, df=100.0, scale=1.0)
        self.assertGreater(low_df_density, high_df_density)


class TestFitStudentTViaMCMC(unittest.TestCase):
    def test_recovers_known_df_and_scale_from_synthetic_data(self):
        # Validated on a known ground truth before trusting it on real,
        # ambiguous data: simulate genuine Student-t draws with known
        # parameters (via the standard chi-square-mixture construction,
        # X = Z / sqrt(V/df) * scale for Z~N(0,1), V~chi2(df)), then
        # confirm MCMC recovers them.
        rng = random.Random(0)
        true_df, true_scale = 5.0, 0.01
        synthetic = []
        for _ in range(1500):
            z = rng.gauss(0, 1)
            v = sum(rng.gauss(0, 1) ** 2 for _ in range(int(true_df)))
            synthetic.append(true_scale * z / math.sqrt(v / true_df))

        result = fx.fit_student_t_via_mcmc(synthetic, n_samples=2000, burn_in=800, seed=1)
        self.assertAlmostEqual(result["df_mean"], true_df, delta=2.5)
        self.assertAlmostEqual(result["scale_mean"], true_scale, delta=0.002)

    def test_real_data_smoke_test_small_subsample(self):
        # A fast, small-subsample smoke test of the full real-data
        # pipeline; the actual reported EUR/USD result in the README
        # uses the full 6,930-observation series with many more MCMC
        # steps, which takes several minutes (see the README for why)
        # and is run once offline rather than inside the test suite.
        returns = fx.load_log_returns(DATA_PATH)
        subsample = returns[-300:]
        result = fx.fit_student_t_via_mcmc(subsample, n_samples=600, burn_in=300, seed=1)
        self.assertGreater(result["df_mean"], 0.0)
        self.assertGreater(result["scale_mean"], 0.0)
        self.assertGreaterEqual(result["acceptance_rate"], 0.0)
        self.assertLessEqual(result["acceptance_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
