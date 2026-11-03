import math
import os
import random
import statistics
import unittest

import brownian_motion as bm

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DEXUSEU_returns.csv")


class TestNormalSampling(unittest.TestCase):
    def test_generate_correct_count(self):
        rng = random.Random(1)
        values = bm.generate_standard_normals(37, rng)
        self.assertEqual(len(values), 37)

    def test_zero_count(self):
        rng = random.Random(1)
        self.assertEqual(bm.generate_standard_normals(0, rng), [])

    def test_moments_match_standard_normal(self):
        rng = random.Random(1)
        values = bm.generate_standard_normals(200000, rng)
        self.assertAlmostEqual(statistics.mean(values), 0.0, delta=0.02)
        self.assertAlmostEqual(statistics.variance(values), 1.0, delta=0.02)

    def test_negative_count_raises(self):
        with self.assertRaises(ValueError):
            bm.generate_standard_normals(-1, random.Random(1))


class TestStandardNormalCDF(unittest.TestCase):
    def test_cdf_at_zero_is_half(self):
        self.assertAlmostEqual(bm.standard_normal_cdf(0.0), 0.5)

    def test_cdf_symmetry(self):
        for x in (0.5, 1.0, 2.3):
            self.assertAlmostEqual(bm.standard_normal_cdf(-x), 1.0 - bm.standard_normal_cdf(x), places=10)

    def test_cdf_known_value(self):
        # Phi(1.959964) ~ 0.975, the standard two-sided 95% z-value.
        self.assertAlmostEqual(bm.standard_normal_cdf(1.959964), 0.975, places=4)

    def test_cdf_monotone(self):
        xs = [-3, -1, 0, 1, 3]
        vals = [bm.standard_normal_cdf(x) for x in xs]
        self.assertEqual(vals, sorted(vals))


class TestSimulateBrownianMotion(unittest.TestCase):
    def test_starts_at_zero(self):
        times, values = bm.simulate_brownian_motion(1.0, 100, seed=1)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(times[0], 0.0)

    def test_lengths_match(self):
        times, values = bm.simulate_brownian_motion(2.0, 250, seed=1)
        self.assertEqual(len(times), 251)
        self.assertEqual(len(values), 251)

    def test_final_time_matches_t_max(self):
        times, _ = bm.simulate_brownian_motion(3.5, 500, seed=1)
        self.assertAlmostEqual(times[-1], 3.5, places=8)

    def test_deterministic_with_seed(self):
        a = bm.simulate_brownian_motion(1.0, 100, seed=42)
        b = bm.simulate_brownian_motion(1.0, 100, seed=42)
        self.assertEqual(a, b)

    def test_terminal_variance_matches_t(self):
        t_max, n_steps, trials = 2.0, 200, 5000
        finals = [bm.simulate_brownian_motion(t_max, n_steps, seed=i)[1][-1] for i in range(trials)]
        self.assertAlmostEqual(statistics.mean(finals), 0.0, delta=0.1)
        self.assertAlmostEqual(statistics.variance(finals), t_max, delta=0.2)

    def test_nonpositive_inputs_raise(self):
        with self.assertRaises(ValueError):
            bm.simulate_brownian_motion(0.0, 10)
        with self.assertRaises(ValueError):
            bm.simulate_brownian_motion(1.0, 0)


class TestDrift(unittest.TestCase):
    def test_zero_drift_zero_vol_is_flat(self):
        times, values = bm.simulate_brownian_motion_with_drift(2.0, 50, mu=0.0, sigma=0.0, seed=1)
        self.assertTrue(all(v == 0.0 for v in values))

    def test_drift_matches_mean_path(self):
        t_max, n_steps, mu, sigma, trials = 3.0, 100, 0.5, 0.2, 4000
        finals = [bm.simulate_brownian_motion_with_drift(t_max, n_steps, mu, sigma, seed=i)[1][-1]
                  for i in range(trials)]
        self.assertAlmostEqual(statistics.mean(finals), mu * t_max, delta=0.05)
        self.assertAlmostEqual(statistics.variance(finals), (sigma ** 2) * t_max, delta=0.05)


class TestPathFunctionals(unittest.TestCase):
    def test_quadratic_variation_known_path(self):
        values = [0.0, 1.0, 1.5, 3.0]
        expected = 1.0 ** 2 + 0.5 ** 2 + 1.5 ** 2
        self.assertAlmostEqual(bm.quadratic_variation(values), expected)

    def test_total_variation_known_path(self):
        values = [0.0, 1.0, 0.5, 2.0]
        expected = abs(1.0) + abs(-0.5) + abs(1.5)
        self.assertAlmostEqual(bm.total_variation(values), expected)

    def test_quadratic_variation_converges_to_t(self):
        t_max = 2.0
        means = []
        for n_steps in (50, 500, 5000):
            qvs = [bm.quadratic_variation(bm.simulate_brownian_motion(t_max, n_steps, seed=i)[1])
                   for i in range(300)]
            means.append(statistics.mean(qvs))
        for m in means:
            self.assertAlmostEqual(m, t_max, delta=0.25)

    def test_quadratic_variation_variance_shrinks_with_finer_partition(self):
        t_max = 2.0
        variances = []
        for n_steps in (20, 200, 2000):
            qvs = [bm.quadratic_variation(bm.simulate_brownian_motion(t_max, n_steps, seed=i)[1])
                   for i in range(400)]
            variances.append(statistics.variance(qvs))
        self.assertGreater(variances[0], variances[1])
        self.assertGreater(variances[1], variances[2])

    def test_total_variation_grows_with_finer_partition(self):
        t_max = 2.0
        means = []
        for n_steps in (50, 500, 5000):
            tvs = [bm.total_variation(bm.simulate_brownian_motion(t_max, n_steps, seed=i)[1])
                   for i in range(200)]
            means.append(statistics.mean(tvs))
        self.assertLess(means[0], means[1])
        self.assertLess(means[1], means[2])


class TestReflectionPrinciple(unittest.TestCase):
    def test_probability_bounds(self):
        for a, t in [(0.5, 1.0), (2.0, 3.0), (5.0, 0.1)]:
            p = bm.reflection_principle_bm(a, t)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_larger_a_is_less_likely(self):
        t = 1.0
        p_small = bm.reflection_principle_bm(0.5, t)
        p_large = bm.reflection_principle_bm(2.0, t)
        self.assertGreater(p_small, p_large)

    def test_matches_monte_carlo(self):
        a, t, n_steps, trials = 1.0, 1.0, 500, 20000
        theoretical = bm.reflection_principle_bm(a, t)
        hits = 0
        for i in range(trials):
            _, values = bm.simulate_brownian_motion(t, n_steps, seed=i)
            if max(values) >= a:
                hits += 1
        empirical = hits / trials
        self.assertAlmostEqual(empirical, theoretical, delta=0.02)

    def test_first_passage_cdf_equals_reflection_principle(self):
        a, t = 1.5, 2.0
        self.assertEqual(bm.first_passage_cdf_bm(a, t), bm.reflection_principle_bm(a, t))

    def test_nonpositive_a_raises(self):
        with self.assertRaises(ValueError):
            bm.reflection_principle_bm(0.0, 1.0)


class TestFirstPassageDensity(unittest.TestCase):
    def test_density_is_nonnegative(self):
        for t in (0.1, 1.0, 5.0):
            self.assertGreaterEqual(bm.first_passage_density_bm(1.0, t), 0.0)

    def test_density_integrates_to_cdf(self):
        a = 1.0
        t_end = 3.0
        n = 20000
        h = t_end / n
        total = 0.0
        for i in range(1, n):
            total += bm.first_passage_density_bm(a, i * h)
        total *= h
        self.assertAlmostEqual(total, bm.first_passage_cdf_bm(a, t_end), delta=0.01)

    def test_simulated_first_passage_time_matches_theory(self):
        a, dt, max_steps, trials = 0.5, 0.01, 2000, 4000
        times = [bm.simulate_first_passage_bm(a, dt, max_steps, seed=i) for i in range(trials)]
        hit_within = sum(1 for t in times if t is not None) / trials
        theoretical = bm.first_passage_cdf_bm(a, max_steps * dt)
        self.assertAlmostEqual(hit_within, theoretical, delta=0.03)


class TestDonsker(unittest.TestCase):
    def test_endpoint_variance_matches_t_max(self):
        t_max, n_steps, trials = 2.0, 400, 4000
        finals = [bm.donsker_walk(n_steps, t_max, seed=i)[1][-1] for i in range(trials)]
        self.assertAlmostEqual(statistics.mean(finals), 0.0, delta=0.1)
        self.assertAlmostEqual(statistics.variance(finals), t_max, delta=0.2)

    def test_converges_toward_brownian_motion_as_steps_increase(self):
        # As n_steps grows, the walk's terminal distribution should
        # look more like the exact Brownian terminal distribution,
        # measured here by how close the kurtosis is to 3 (Gaussian).
        t_max, trials = 1.0, 6000

        def excess_kurtosis(sample):
            mean = statistics.mean(sample)
            var = statistics.variance(sample)
            m4 = sum((x - mean) ** 4 for x in sample) / len(sample)
            return m4 / (var ** 2) - 3.0

        coarse = [bm.donsker_walk(4, t_max, seed=i)[1][-1] for i in range(trials)]
        fine = [bm.donsker_walk(400, t_max, seed=i)[1][-1] for i in range(trials)]
        self.assertLess(abs(excess_kurtosis(fine)), abs(excess_kurtosis(coarse)) + 0.3)

    def test_nonpositive_inputs_raise(self):
        with self.assertRaises(ValueError):
            bm.donsker_walk(0, 1.0)
        with self.assertRaises(ValueError):
            bm.donsker_walk(10, 0.0)


class TestBrownianBridge(unittest.TestCase):
    def test_endpoints_pinned_to_zero(self):
        times, bridge = bm.brownian_bridge(2.0, 200, seed=1)
        self.assertAlmostEqual(bridge[0], 0.0, places=8)
        self.assertAlmostEqual(bridge[-1], 0.0, places=8)

    def test_variance_matches_known_formula(self):
        t_max, n_steps, trials = 4.0, 400, 4000
        checkpoint_index = n_steps // 4  # t = t_max/4
        values_at_checkpoint = []
        for i in range(trials):
            times, bridge = bm.brownian_bridge(t_max, n_steps, seed=i)
            values_at_checkpoint.append(bridge[checkpoint_index])
        t = t_max / 4
        theoretical_var = t * (t_max - t) / t_max
        self.assertAlmostEqual(statistics.variance(values_at_checkpoint), theoretical_var, delta=0.15)


class TestFXApplication(unittest.TestCase):
    def test_load_log_returns_reads_real_data(self):
        returns = bm.load_log_returns(DATA_PATH)
        self.assertEqual(len(returns), 6930)

    def test_realized_variance_positive(self):
        returns = bm.load_log_returns(DATA_PATH)
        self.assertGreater(bm.realized_variance(returns), 0.0)

    def test_realized_variance_matches_manual_sum(self):
        returns = [0.01, -0.02, 0.005]
        expected = 0.01 ** 2 + 0.02 ** 2 + 0.005 ** 2
        self.assertAlmostEqual(bm.realized_variance(returns), expected)

    def test_variance_ratio_near_one_under_iid_null(self):
        rng = random.Random(1)
        synthetic = [rng.gauss(0.0, 0.01) for _ in range(6000)]
        result = bm.variance_ratio_test(synthetic, q=5)
        self.assertAlmostEqual(result["vr"], 1.0, delta=0.15)
        self.assertGreater(result["p_value"], 0.01)

    def test_variance_ratio_detects_strong_positive_autocorrelation(self):
        rng = random.Random(2)
        base = [rng.gauss(0.0, 0.01) for _ in range(3000)]
        # Momentum: each return repeats the sign and half the magnitude
        # of the previous one, inducing strong positive autocorrelation.
        momentum = [base[0]]
        for i in range(1, len(base)):
            momentum.append(0.5 * momentum[-1] + base[i])
        result = bm.variance_ratio_test(momentum, q=10)
        self.assertGreater(result["vr"], 1.3)
        self.assertLess(result["p_value"], 0.01)

    def test_variance_ratio_rejects_small_q(self):
        with self.assertRaises(ValueError):
            bm.variance_ratio_test([0.01] * 1000, q=1)

    def test_variance_ratio_on_real_data_close_to_one(self):
        returns = bm.load_log_returns(DATA_PATH)
        for q in (2, 5, 10, 20):
            result = bm.variance_ratio_test(returns, q)
            self.assertAlmostEqual(result["vr"], 1.0, delta=0.15)
            self.assertGreater(result["p_value"], 0.05)

    def test_compare_scaling_to_theory_structure(self):
        returns = bm.load_log_returns(DATA_PATH)
        result = bm.compare_scaling_to_theory(returns, horizons=[1, 5, 10, 20])
        self.assertEqual(len(result["empirical_variance"]), 4)
        self.assertEqual(len(result["theoretical_variance"]), 4)
        # Both series should be increasing in q.
        self.assertEqual(result["theoretical_variance"], sorted(result["theoretical_variance"]))


if __name__ == "__main__":
    unittest.main()
