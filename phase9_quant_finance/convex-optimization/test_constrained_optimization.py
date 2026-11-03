import unittest

import constrained_optimization as co
import linear_algebra as la


class TestSimplexProjection(unittest.TestCase):
    def test_point_already_in_simplex_is_unchanged(self):
        v = [0.5, 0.3, 0.2]
        proj = co.project_onto_simplex(v)
        for a, b in zip(proj, v):
            self.assertAlmostEqual(a, b, places=8)

    def test_result_always_sums_to_one(self):
        for v in ([2.0, 1.0, -1.0], [0.4, 0.4, 0.4], [-5.0, 10.0, 0.0], [1.0, 1.0, 1.0, 1.0]):
            proj = co.project_onto_simplex(v)
            self.assertAlmostEqual(sum(proj), 1.0, places=8)

    def test_result_always_nonnegative(self):
        for v in ([2.0, 1.0, -1.0], [-5.0, 10.0, 0.0], [-1.0, -2.0, -3.0]):
            proj = co.project_onto_simplex(v)
            for x in proj:
                self.assertGreaterEqual(x, -1e-10)

    def test_dominant_coordinate_gets_all_mass(self):
        proj = co.project_onto_simplex([100.0, 0.0, 0.0])
        self.assertAlmostEqual(proj[0], 1.0, places=6)
        self.assertAlmostEqual(proj[1], 0.0, places=6)
        self.assertAlmostEqual(proj[2], 0.0, places=6)

    def test_symmetric_point_projects_to_uniform(self):
        proj = co.project_onto_simplex([0.4, 0.4, 0.4])
        for x in proj:
            self.assertAlmostEqual(x, 1.0 / 3, places=8)

    def test_projection_is_the_closest_simplex_point(self):
        v = [3.0, -1.0, 0.5]
        proj = co.project_onto_simplex(v)
        proj_dist = la.vector_norm(la.vec_sub(proj, v))
        best_alt_dist = float("inf")
        for i in range(0, 11):
            for j in range(0, 11 - i):
                k = 10 - i - j
                candidate = [i / 10, j / 10, k / 10]
                dist = la.vector_norm(la.vec_sub(candidate, v))
                best_alt_dist = min(best_alt_dist, dist)
        self.assertLessEqual(proj_dist, best_alt_dist + 1e-9)


class TestBoxProjection(unittest.TestCase):
    def test_clips_to_bounds(self):
        result = co.project_onto_box([5.0, -5.0, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        self.assertEqual(result, [1.0, 0.0, 0.5])

    def test_interior_point_unchanged(self):
        result = co.project_onto_box([0.3, 0.7], [0.0, 0.0], [1.0, 1.0])
        self.assertEqual(result, [0.3, 0.7])


class TestProjectedGradientDescent(unittest.TestCase):
    def test_unconstrained_optimum_already_feasible(self):
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        f = lambda w: 0.5 * la.dot(w, la.mat_vec(sigma, w))
        grad_fn = lambda w: la.mat_vec(sigma, w)

        result = co.projected_gradient_descent(f, [1 / 3, 1 / 3, 1 / 3], co.project_onto_simplex,
                                                 grad_fn=grad_fn, step=0.5)

        sigma_inv = la.invert(sigma)
        ones = [1.0, 1.0, 1.0]
        numer = la.mat_vec(sigma_inv, ones)
        denom = la.dot(ones, numer)
        closed_form = [x / denom for x in numer]

        for a, b in zip(result["x"], closed_form):
            self.assertAlmostEqual(a, b, places=4)

    def test_result_respects_the_simplex_constraint(self):
        sigma = [[1.0, 0.0], [0.0, 4.0]]
        f = lambda w: 0.5 * la.dot(w, la.mat_vec(sigma, w))
        grad_fn = lambda w: la.mat_vec(sigma, w)
        result = co.projected_gradient_descent(f, [0.5, 0.5], co.project_onto_simplex, grad_fn=grad_fn, step=0.3)
        self.assertAlmostEqual(sum(result["x"]), 1.0, places=6)
        for w in result["x"]:
            self.assertGreaterEqual(w, -1e-8)

    def test_box_constrained_optimum_at_boundary(self):
        f = lambda x: (x[0] - 5) ** 2
        grad_fn = lambda x: [2 * (x[0] - 5)]
        project = lambda x: co.project_onto_box(x, [0.0], [2.0])
        result = co.projected_gradient_descent(f, [0.0], project, grad_fn=grad_fn, step=0.1)
        self.assertAlmostEqual(result["x"][0], 2.0, places=4)


class TestEqualityConstrainedQP(unittest.TestCase):
    def test_matches_closed_form_min_variance_formula(self):
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        result = co.solve_equality_constrained_qp(sigma, [0.0, 0.0, 0.0], [[1.0, 1.0, 1.0]], [1.0])

        sigma_inv = la.invert(sigma)
        ones = [1.0, 1.0, 1.0]
        numer = la.mat_vec(sigma_inv, ones)
        denom = la.dot(ones, numer)
        closed_form = [x / denom for x in numer]

        for a, b in zip(result["x"], closed_form):
            self.assertAlmostEqual(a, b, places=8)

    def test_solution_satisfies_the_constraint(self):
        sigma = [[2.0, 0.5], [0.5, 1.0]]
        result = co.solve_equality_constrained_qp(sigma, [0.0, 0.0], [[1.0, 1.0]], [1.0])
        self.assertAlmostEqual(sum(result["x"]), 1.0, places=8)

    def test_two_constraints_target_return(self):
        sigma = [[0.04, 0.0], [0.0, 0.09]]
        mu = [0.05, 0.10]
        a = [[1.0, 1.0], mu]
        b = [1.0, 0.08]
        result = co.solve_equality_constrained_qp(sigma, [0.0, 0.0], a, b)
        self.assertAlmostEqual(sum(result["x"]), 1.0, places=6)
        self.assertAlmostEqual(la.dot(result["x"], mu), 0.08, places=6)


class TestVerifyKKT(unittest.TestCase):
    def test_kkt_holds_at_the_qp_solution(self):
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        result = co.solve_equality_constrained_qp(sigma, [0.0, 0.0, 0.0], [[1.0, 1.0, 1.0]], [1.0])
        w, lam = result["x"], result["lambda"]

        grad_f = la.mat_vec(sigma, w)
        grad_h = [[1.0, 1.0, 1.0]]
        h_values = [sum(w) - 1.0]

        kkt = co.verify_kkt(grad_f, grad_h, h_values, lam)
        self.assertTrue(kkt["all_conditions_satisfied"])
        self.assertLess(kkt["stationarity_residual"], 1e-8)

    def test_kkt_fails_at_a_non_optimal_point(self):
        sigma = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]
        w = [1 / 3, 1 / 3, 1 / 3]
        grad_f = la.mat_vec(sigma, w)
        grad_h = [[1.0, 1.0, 1.0]]
        h_values = [sum(w) - 1.0]
        kkt = co.verify_kkt(grad_f, grad_h, h_values, [-grad_f[0]])
        self.assertFalse(kkt["stationarity"])

    def test_inequality_constraint_kkt_with_active_bound(self):
        x = 2.0
        grad_f = [2 * (x - 5)]
        grad_g = [[1.0]]
        g_values = [x - 2.0]
        mu = [6.0]
        kkt = co.verify_kkt(grad_f, [], [], [], grad_g=grad_g, g_values=g_values, mu=mu)
        self.assertTrue(kkt["all_conditions_satisfied"])

    def test_inequality_constraint_kkt_fails_with_negative_multiplier(self):
        x = 2.0
        grad_f = [2 * (x - 5)]
        grad_g = [[1.0]]
        g_values = [x - 2.0]
        mu = [-6.0]
        kkt = co.verify_kkt(grad_f, [], [], [], grad_g=grad_g, g_values=g_values, mu=mu)
        self.assertFalse(kkt["dual_feasibility"])
        self.assertFalse(kkt["all_conditions_satisfied"])


if __name__ == "__main__":
    unittest.main()
