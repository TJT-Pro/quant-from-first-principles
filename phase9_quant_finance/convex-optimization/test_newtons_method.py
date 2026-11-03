import math
import unittest

import newtons_method as nm


class TestExactOneStepConvergenceOnQuadratics(unittest.TestCase):
    def test_isotropic_quadratic_with_analytic_derivatives(self):
        f = lambda x: x[0] ** 2 + x[1] ** 2
        grad_fn = lambda x: [2 * x[0], 2 * x[1]]
        hess_fn = lambda x: [[2.0, 0.0], [0.0, 2.0]]
        result = nm.newtons_method(f, [5.0, -3.0], grad_fn=grad_fn, hess_fn=hess_fn)
        self.assertEqual(result["iterations"], 1)
        self.assertAlmostEqual(result["x"][0], 0.0, places=10)
        self.assertAlmostEqual(result["x"][1], 0.0, places=10)

    def test_ill_conditioned_quadratic_still_one_step_with_analytic_derivatives(self):
        f = lambda x: 100 * x[0] ** 2 + x[1] ** 2
        grad_fn = lambda x: [200 * x[0], 2 * x[1]]
        hess_fn = lambda x: [[200.0, 0.0], [0.0, 2.0]]
        result = nm.newtons_method(f, [1.0, 1.0], grad_fn=grad_fn, hess_fn=hess_fn)
        self.assertEqual(result["iterations"], 1)
        self.assertAlmostEqual(result["x"][0], 0.0, places=10)
        self.assertAlmostEqual(result["x"][1], 0.0, places=10)

    def test_far_starting_point_still_one_step(self):
        f = lambda x: (x[0] - 3) ** 2 + (x[1] + 2) ** 2
        grad_fn = lambda x: [2 * (x[0] - 3), 2 * (x[1] + 2)]
        hess_fn = lambda x: [[2.0, 0.0], [0.0, 2.0]]
        result = nm.newtons_method(f, [1000.0, -500.0], grad_fn=grad_fn, hess_fn=hess_fn)
        self.assertEqual(result["iterations"], 1)
        self.assertAlmostEqual(result["x"][0], 3.0, places=8)
        self.assertAlmostEqual(result["x"][1], -2.0, places=8)

    def test_numerical_derivatives_need_one_extra_corrective_step(self):
        f = lambda x: 100 * x[0] ** 2 + x[1] ** 2
        result = nm.newtons_method(f, [1.0, 1.0])
        self.assertEqual(result["iterations"], 2)
        self.assertAlmostEqual(result["x"][0], 0.0, places=6)
        self.assertAlmostEqual(result["x"][1], 0.0, places=6)


class TestQuadraticConvergenceRate(unittest.TestCase):
    def test_convergence_ratio_matches_theoretical_constant(self):
        f = lambda x: math.exp(x[0]) - x[0]
        grad_fn = lambda x: [math.exp(x[0]) - 1]
        hess_fn = lambda x: [[math.exp(x[0])]]
        result = nm.newtons_method(f, [2.0], grad_fn=grad_fn, hess_fn=hess_fn, tol=1e-15, max_iter=20)

        errors = [abs(x[0]) for x in result["x_history"]]
        ratios = []
        for i in range(len(errors) - 1):
            if 1e-8 < errors[i] < 0.01:
                ratios.append(errors[i + 1] / errors[i] ** 2)
        self.assertGreater(len(ratios), 1)
        for r in ratios:
            self.assertAlmostEqual(r, 0.5, delta=0.02)

    def test_error_decreases_overall_despite_possible_early_nonmonotonicity(self):
        f = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        grad_fn = lambda x: [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]

        def hess_fn(x):
            h11 = 2 - 400 * x[1] + 1200 * x[0] ** 2
            h12 = -400 * x[0]
            return [[h11, h12], [h12, 200.0]]

        result = nm.newtons_method(f, [-1.0, 1.0], grad_fn=grad_fn, hess_fn=hess_fn, tol=1e-13, max_iter=50)
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["x"][0], 1.0, places=6)
        self.assertAlmostEqual(result["x"][1], 1.0, places=6)


class TestNewtonVsGradientDescentComparison(unittest.TestCase):
    def test_newton_needs_far_fewer_iterations_on_ill_conditioned_quadratic(self):
        import gradient_methods as gm
        f = lambda x: 100 * x[0] ** 2 + x[1] ** 2
        grad_fn = lambda x: [200 * x[0], 2 * x[1]]
        hess_fn = lambda x: [[200.0, 0.0], [0.0, 2.0]]

        newton_result = nm.newtons_method(f, [1.0, 1.0], grad_fn=grad_fn, hess_fn=hess_fn)
        gd_result = gm.gradient_descent(f, [1.0, 1.0], grad_fn=grad_fn)

        self.assertLess(newton_result["iterations"], gd_result["iterations"] / 50)


if __name__ == "__main__":
    unittest.main()
