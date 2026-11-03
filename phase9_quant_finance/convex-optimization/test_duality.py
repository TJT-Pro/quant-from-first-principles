import unittest

import duality as du
import linear_algebra as la


class TestLagrangian(unittest.TestCase):
    def test_matches_manual_computation(self):
        f = lambda x: x[0] ** 2 + x[1] ** 2
        h_list = [lambda x: x[0] + x[1] - 1]
        g_list = [lambda x: x[0] - 5]
        x = [2.0, 3.0]
        lam = [4.0]
        mu = [2.0]
        expected = f(x) + 4.0 * h_list[0](x) + 2.0 * g_list[0](x)
        self.assertAlmostEqual(du.lagrangian(f, h_list, g_list, x, lam, mu), expected)

    def test_reduces_to_f_when_multipliers_zero(self):
        f = lambda x: x[0] ** 3 - x[1]
        h_list = [lambda x: x[0] - 1]
        g_list = [lambda x: x[1] - 2]
        x = [3.0, -1.0]
        self.assertAlmostEqual(du.lagrangian(f, h_list, g_list, x, [0.0], [0.0]), f(x))


class TestKnownAnalyticalExample(unittest.TestCase):
    def setUp(self):
        self.f = lambda x: x[0] ** 2
        self.g_list = [lambda x: 1.0 - x[0]]

    def test_dual_function_matches_hand_derivation_at_a_fixed_mu(self):
        result = du.dual_function(self.f, [], self.g_list, [], [2.0], [0.5])
        self.assertAlmostEqual(result["dual_value"], 1.0, places=4)
        self.assertAlmostEqual(result["minimizing_x"][0], 1.0, places=4)

    def test_dual_ascent_recovers_known_optimal_multiplier(self):
        result = du.maximize_dual_via_projected_gradient_ascent(
            self.f, [], self.g_list, [], [0.0], [0.5], step=0.5, max_iter=200)
        self.assertAlmostEqual(result["mu"][0], 2.0, places=3)
        self.assertAlmostEqual(result["x"][0], 1.0, places=3)
        self.assertAlmostEqual(result["dual_value"], 1.0, places=3)

    def test_strong_duality_holds(self):
        result = du.maximize_dual_via_projected_gradient_ascent(
            self.f, [], self.g_list, [], [0.0], [0.5], step=0.5, max_iter=200)
        strong = du.verify_strong_duality(primal_optimal_value=1.0, dual_optimal_value=result["dual_value"])
        self.assertTrue(strong["strong_duality_holds"])
        self.assertAlmostEqual(strong["duality_gap"], 0.0, places=3)


class TestWeakDuality(unittest.TestCase):
    def test_holds_for_any_feasible_gap(self):
        result = du.verify_weak_duality(primal_value=5.0, dual_value=3.0)
        self.assertTrue(result["weak_duality_holds"])
        self.assertAlmostEqual(result["duality_gap"], 2.0)

    def test_holds_at_exact_equality(self):
        result = du.verify_weak_duality(primal_value=1.0, dual_value=1.0)
        self.assertTrue(result["weak_duality_holds"])

    def test_fails_if_dual_exceeds_primal(self):
        result = du.verify_weak_duality(primal_value=1.0, dual_value=5.0)
        self.assertFalse(result["weak_duality_holds"])


class TestStrongDualityOnMinVarianceProblem(unittest.TestCase):
    def test_equality_constrained_min_variance_has_zero_duality_gap(self):
        import constrained_optimization as co
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        qp_result = co.solve_equality_constrained_qp(sigma, [0.0, 0.0, 0.0], [[1.0, 1.0, 1.0]], [1.0])
        w_star, lam_star = qp_result["x"], qp_result["lambda"]

        f = lambda w: 0.5 * la.dot(w, la.mat_vec(sigma, w))
        h_list = [lambda w: sum(w) - 1.0]
        primal_optimal_value = f(w_star)

        dual_result = du.dual_function(f, h_list, [], lam_star, [], w_star)
        strong = du.verify_strong_duality(primal_optimal_value, dual_result["dual_value"], tol=1e-6)
        self.assertTrue(strong["strong_duality_holds"])


if __name__ == "__main__":
    unittest.main()
