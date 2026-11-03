import math
import unittest

import brownian_motion as bm


class TestSimulateBrownianPath(unittest.TestCase):
    def test_starts_at_zero(self):
        ts, ws = bm.simulate_brownian_path(1.0, 100, seed=1)
        self.assertEqual(ws[0], 0.0)
        self.assertEqual(ts[0], 0.0)

    def test_correct_length_and_endpoint_time(self):
        ts, ws = bm.simulate_brownian_path(2.0, 50, seed=1)
        self.assertEqual(len(ts), 51)
        self.assertEqual(len(ws), 51)
        self.assertAlmostEqual(ts[-1], 2.0, places=10)

    def test_deterministic_given_seed(self):
        _, ws1 = bm.simulate_brownian_path(1.0, 100, seed=42)
        _, ws2 = bm.simulate_brownian_path(1.0, 100, seed=42)
        self.assertEqual(ws1, ws2)


class TestTerminalDistribution(unittest.TestCase):
    def test_empirical_mean_close_to_zero(self):
        result = bm.verify_terminal_distribution(1.0, 100, 20000, seed=1)
        self.assertAlmostEqual(result["empirical_mean"], 0.0, delta=0.05)

    def test_empirical_variance_close_to_t(self):
        result = bm.verify_terminal_distribution(2.5, 100, 20000, seed=2)
        self.assertAlmostEqual(result["empirical_variance"], 2.5, delta=0.1)


class TestQuadraticVariation(unittest.TestCase):
    def test_known_small_path(self):
        ws = [0.0, 1.0, 0.0, 2.0]
        self.assertAlmostEqual(bm.quadratic_variation(ws), 6.0)

    def test_converges_to_t_as_resolution_increases(self):
        result = bm.verify_quadratic_variation_converges_to_t(1.0, [10, 100, 1000, 20000], seed=1)
        finest_qv = result["results"][-1]["quadratic_variation"]
        coarsest_qv = result["results"][0]["quadratic_variation"]
        finest_error = abs(finest_qv - 1.0)
        coarsest_error = abs(coarsest_qv - 1.0)
        self.assertLess(finest_error, coarsest_error)
        self.assertLess(finest_error, 0.05)

    def test_zero_variation_for_constant_path(self):
        ws = [1.0, 1.0, 1.0, 1.0]
        self.assertEqual(bm.quadratic_variation(ws), 0.0)


class TestScalingProperty(unittest.TestCase):
    def test_scaled_process_variance_matches_t(self):
        result = bm.verify_scaling_property(c=4.0, t_end=1.0, n_steps=100, n_paths=20000, seed=3)
        self.assertAlmostEqual(result["empirical_variance_of_scaled_process"], 1.0, delta=0.1)

    def test_different_scaling_constant(self):
        result = bm.verify_scaling_property(c=0.25, t_end=2.0, n_steps=100, n_paths=20000, seed=4)
        self.assertAlmostEqual(result["empirical_variance_of_scaled_process"], 2.0, delta=0.15)


if __name__ == "__main__":
    unittest.main()
