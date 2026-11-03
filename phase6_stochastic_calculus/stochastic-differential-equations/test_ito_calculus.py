import math
import unittest

import ito_calculus as ito


class TestApplyItoLemma(unittest.TestCase):
    def test_log_transform_of_gbm_gives_constant_drift_and_diffusion(self):
        mu_param, sigma_param = 0.05, 0.3
        mu_fn = lambda x, t: mu_param * x
        sigma_fn = lambda x, t: sigma_param * x
        f, f_x, f_xx = math.log, lambda x: 1.0 / x, lambda x: -1.0 / x ** 2

        mu_y, sigma_y = ito.apply_ito_lemma(f, f_x, f_xx, mu_fn, sigma_fn)
        for x in (0.5, 1.0, 5.0, 100.0):
            self.assertAlmostEqual(mu_y(x, 0.0), mu_param - 0.5 * sigma_param ** 2, places=8)
            self.assertAlmostEqual(sigma_y(x, 0.0), sigma_param, places=8)


class TestVerifyItoTransformMatchesDirectSimulation(unittest.TestCase):
    def test_log_transform_of_gbm_matches_within_discretization_error(self):
        mu_param, sigma_param = 0.05, 0.3
        mu_fn = lambda x, t: mu_param * x
        sigma_fn = lambda x, t: sigma_param * x
        f, f_x, f_xx = math.log, lambda x: 1.0 / x, lambda x: -1.0 / x ** 2

        result = ito.verify_ito_transform_matches_direct_simulation(
            f, f_x, f_xx, mu_fn, sigma_fn, x0=1.0, t_end=1.0, n_steps=5000, n_paths=300, seed=1)
        self.assertLess(result["mean_absolute_difference"], 0.01)

    def test_square_transform_of_ou_process_matches(self):
        theta, mu_ou, sigma_ou = 1.0, 0.05, 0.2
        mu_fn = lambda x, t: theta * (mu_ou - x)
        sigma_fn = lambda x, t: sigma_ou
        f, f_x, f_xx = lambda x: x ** 2, lambda x: 2 * x, lambda x: 2.0

        result = ito.verify_ito_transform_matches_direct_simulation(
            f, f_x, f_xx, mu_fn, sigma_fn, x0=0.1, t_end=1.0, n_steps=5000, n_paths=300, seed=1)
        self.assertLess(result["mean_absolute_difference"], 0.005)

    def test_finer_discretization_gives_smaller_discrepancy(self):
        mu_param, sigma_param = 0.05, 0.3
        mu_fn = lambda x, t: mu_param * x
        sigma_fn = lambda x, t: sigma_param * x
        f, f_x, f_xx = math.log, lambda x: 1.0 / x, lambda x: -1.0 / x ** 2

        coarse = ito.verify_ito_transform_matches_direct_simulation(
            f, f_x, f_xx, mu_fn, sigma_fn, x0=1.0, t_end=1.0, n_steps=200, n_paths=300, seed=2)
        fine = ito.verify_ito_transform_matches_direct_simulation(
            f, f_x, f_xx, mu_fn, sigma_fn, x0=1.0, t_end=1.0, n_steps=20000, n_paths=300, seed=2)
        self.assertLess(fine["mean_absolute_difference"], coarse["mean_absolute_difference"])


class TestGBMLogTransformCheck(unittest.TestCase):
    def test_matches_exact_solution_to_floating_point_precision(self):
        result = ito.gbm_log_transform_check(0.05, 0.3, 1.0, 1.0, 1000, seed=1)
        self.assertAlmostEqual(result["difference"], 0.0, places=10)

    def test_consistent_across_different_parameter_values(self):
        for mu_param, sigma_param, x0 in [(0.1, 0.2, 50.0), (-0.05, 0.5, 1.0), (0.0, 0.15, 100.0)]:
            result = ito.gbm_log_transform_check(mu_param, sigma_param, x0, 2.0, 500, seed=3)
            self.assertAlmostEqual(result["difference"], 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
