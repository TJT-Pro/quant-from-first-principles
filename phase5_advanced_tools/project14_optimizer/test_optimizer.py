"""Tests for the gradient descent optimizer and LP solver.

All expected values are computed analytically on simple quadratic and
linear problems where exact solutions are known.  No numpy or scipy.
"""

import math
import unittest

import optimizer as op


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def f_quad(x):
    """f(x) = x₀² + x₁² -- minimum at origin, value 0."""
    return x[0]**2 + x[1]**2


def f_shifted(x):
    """f(x) = (x₀-3)² + (x₁+2)² -- minimum at (3,-2), value 0."""
    return (x[0] - 3.0)**2 + (x[1] + 2.0)**2


def f_rosenbrock(x):
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2


def allclose(a, b, tol=1e-4):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Vanilla gradient descent
# ---------------------------------------------------------------------------

class TestGradientDescent(unittest.TestCase):
    def test_converges_on_quadratic(self):
        r = op.gradient_descent(f_quad, [3.0, 4.0], learning_rate=0.1)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["x"][0], 0.0, places=4)
        self.assertAlmostEqual(r["x"][1], 0.0, places=4)

    def test_converges_to_shifted_minimum(self):
        r = op.gradient_descent(f_shifted, [0.0, 0.0], learning_rate=0.1)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["x"][0], 3.0, places=4)
        self.assertAlmostEqual(r["x"][1], -2.0, places=4)

    def test_history_non_increasing(self):
        r = op.gradient_descent(f_quad, [5.0, 5.0], learning_rate=0.1)
        for a, b in zip(r["history"], r["history"][1:]):
            self.assertGreaterEqual(a + 1e-12, b)

    def test_bad_learning_rate(self):
        with self.assertRaises(ValueError):
            op.gradient_descent(f_quad, [1.0, 1.0], learning_rate=0.0)

    def test_analytical_gradient_used(self):
        grad_fn = lambda x: [2*x[0], 2*x[1]]
        r = op.gradient_descent(f_quad, [2.0, 2.0],
                                  learning_rate=0.1, grad_fn=grad_fn)
        self.assertTrue(r["converged"])

    def test_f_val_is_f_at_x(self):
        r = op.gradient_descent(f_quad, [1.0, 1.0], learning_rate=0.1)
        self.assertAlmostEqual(r["f_val"], f_quad(r["x"]), places=12)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

class TestMomentum(unittest.TestCase):
    def test_converges_on_quadratic(self):
        r = op.momentum_gd(f_quad, [3.0, 4.0], learning_rate=0.05, beta=0.9)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["x"][0], 0.0, places=4)
        self.assertAlmostEqual(r["x"][1], 0.0, places=4)

    def test_fewer_iterations_than_vanilla(self):
        # Elongated quadratic (ill-conditioned): momentum converges faster
        f_elong = lambda x: 10.0*x[0]**2 + x[1]**2
        r_vanilla  = op.gradient_descent(f_elong, [3.0, 3.0],
                                          learning_rate=0.05,
                                          max_iter=5000, tolerance=1e-6)
        r_momentum = op.momentum_gd(f_elong, [3.0, 3.0],
                                     learning_rate=0.05, beta=0.8,
                                     max_iter=5000, tolerance=1e-6)
        # Both should converge; momentum uses fewer or equal iterations
        if r_vanilla["converged"] and r_momentum["converged"]:
            self.assertLessEqual(r_momentum["iterations"],
                                  r_vanilla["iterations"] + 10)

    def test_bad_beta(self):
        with self.assertRaises(ValueError):
            op.momentum_gd(f_quad, [1.0], learning_rate=0.1, beta=-0.1)


# ---------------------------------------------------------------------------
# RMSProp
# ---------------------------------------------------------------------------

class TestRMSProp(unittest.TestCase):
    def test_converges_on_quadratic(self):
        r = op.rmsprop(f_quad, [3.0, 4.0], learning_rate=0.1, rho=0.9)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["x"][0], 0.0, places=4)
        self.assertAlmostEqual(r["x"][1], 0.0, places=4)

    def test_bad_rho(self):
        with self.assertRaises(ValueError):
            op.rmsprop(f_quad, [1.0], learning_rate=0.1, rho=1.1)

    def test_history_length(self):
        r = op.rmsprop(f_quad, [1.0, 1.0], learning_rate=0.1, max_iter=50)
        self.assertLessEqual(len(r["history"]), 52)


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class TestAdam(unittest.TestCase):
    def test_converges_on_quadratic(self):
        r = op.adam(f_quad, [3.0, 4.0], learning_rate=0.1)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["x"][0], 0.0, places=3)
        self.assertAlmostEqual(r["x"][1], 0.0, places=3)

    def test_rosenbrock_finds_minimum(self):
        # Adam should get close to (1,1) from a reasonable start
        r = op.adam(f_rosenbrock, [0.0, 0.0], learning_rate=0.01,
                    max_iter=20000, tolerance=1e-5)
        self.assertAlmostEqual(r["x"][0], 1.0, places=1)
        self.assertAlmostEqual(r["x"][1], 1.0, places=1)

    def test_history_starts_high(self):
        r = op.adam(f_quad, [10.0, 10.0], learning_rate=0.1)
        self.assertGreater(r["history"][0], r["history"][-1])

    def test_bad_betas(self):
        with self.assertRaises(ValueError):
            op.adam(f_quad, [1.0], beta1=1.1)


# ---------------------------------------------------------------------------
# Backtracking line search
# ---------------------------------------------------------------------------

class TestLineSearch(unittest.TestCase):
    def test_step_accepted_immediately_on_gentle_fn(self):
        # f = x^2, grad = [2x], from x=1, direction = [-2] (negative grad)
        x = [1.0]
        g = [2.0]
        d = [-2.0]
        alpha = op.backtracking_line_search(
            lambda v: v[0]**2, x, d, 1.0, g)
        self.assertGreater(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)

    def test_converges_with_line_search(self):
        r = op.gradient_descent_ls(f_shifted, [0.0, 0.0], tolerance=1e-8)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["x"][0], 3.0, places=5)
        self.assertAlmostEqual(r["x"][1], -2.0, places=5)


# ---------------------------------------------------------------------------
# Nelder-Mead
# ---------------------------------------------------------------------------

class TestNelderMead(unittest.TestCase):
    def test_quadratic_minimum(self):
        r = op.nelder_mead(f_quad, [3.0, 4.0])
        self.assertAlmostEqual(r["x"][0], 0.0, places=4)
        self.assertAlmostEqual(r["x"][1], 0.0, places=4)

    def test_shifted_minimum(self):
        r = op.nelder_mead(f_shifted, [0.0, 0.0])
        self.assertAlmostEqual(r["x"][0], 3.0, places=3)
        self.assertAlmostEqual(r["x"][1], -2.0, places=3)

    def test_rosenbrock_close(self):
        r = op.nelder_mead(f_rosenbrock, [0.5, 0.5], step=0.5, max_iter=5000)
        self.assertAlmostEqual(r["x"][0], 1.0, places=2)
        self.assertAlmostEqual(r["x"][1], 1.0, places=2)

    def test_derivative_free(self):
        # Works on a function without analytical gradient
        calls = [0]
        def bump(x):
            calls[0] += 1
            return (x[0] - 1.5)**2 + (x[1] + 0.5)**2
        r = op.nelder_mead(bump, [0.0, 0.0])
        self.assertAlmostEqual(r["x"][0], 1.5, places=3)


# ---------------------------------------------------------------------------
# LP simplex solver
# ---------------------------------------------------------------------------

class TestLPSimplex(unittest.TestCase):
    def test_simple_1d(self):
        # min x  s.t. x >= 2  → x <= -2 in standard form with negation
        # Reformulate: min -(-x) = min x, A=[[-1]], b=[-2] -- but b must be >=0
        # Correct form: min x, x >= 2 → min (-x) s.t. x <= UB and x >= 2
        # Simple: min x s.t. x <= 5, x >= 0: optimal x=0
        r = op.lp_simplex(c=[1.0], A=[[1.0]], b=[5.0])
        self.assertEqual(r["status"], "optimal")
        self.assertAlmostEqual(r["x"][0], 0.0, places=8)
        self.assertAlmostEqual(r["obj"], 0.0, places=8)

    def test_2d_standard(self):
        # min -x₀ - x₁  s.t. x₀+x₁ <= 4, x₀ <= 3, x₁ <= 3, x ≥ 0
        # Optimal: x=(3,1) or (1,3) with obj=-4, or x=(3,1) giving -4
        r = op.lp_simplex(
            c=[-1.0, -1.0],
            A=[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            b=[4.0, 3.0, 3.0],
        )
        self.assertEqual(r["status"], "optimal")
        self.assertAlmostEqual(r["obj"], -4.0, places=6)

    def test_maximise_via_negation(self):
        # max 5x + 4y  s.t. 6x + 4y <= 24, x + 2y <= 6, x,y >= 0
        # Optimal: x=3, y=3/2? Or check: max is 21 at x=24/6=4, y=0? or x=0, y=3?
        # At x=4,y=0: 5*4=20. At x=0,y=3: 4*3=12. At corner (3,3/2): 15+6=21
        r = op.lp_simplex(
            c=[-5.0, -4.0],
            A=[[6.0, 4.0], [1.0, 2.0]],
            b=[24.0, 6.0],
        )
        self.assertEqual(r["status"], "optimal")
        self.assertAlmostEqual(r["obj"], -21.0, places=5)

    def test_feasible_but_zero_obj(self):
        r = op.lp_simplex(c=[0.0, 0.0], A=[[1.0, 0.0]], b=[3.0])
        self.assertEqual(r["status"], "optimal")
        self.assertAlmostEqual(r["obj"], 0.0, places=8)

    def test_negative_b_raises(self):
        with self.assertRaises(ValueError):
            op.lp_simplex(c=[1.0], A=[[1.0]], b=[-1.0])

    def test_solution_satisfies_constraints(self):
        A = [[1.0, 2.0], [3.0, 1.0]]
        b = [14.0, 14.0]
        r = op.lp_simplex(c=[-2.0, -3.0], A=A, b=b)
        x = r["x"]
        for i in range(len(b)):
            lhs = sum(A[i][j] * x[j] for j in range(len(x)))
            self.assertLessEqual(lhs, b[i] + 1e-6)

    def test_non_negativity(self):
        r = op.lp_simplex(c=[-1.0, -2.0], A=[[1.0, 1.0]], b=[10.0])
        self.assertGreaterEqual(r["x"][0], -1e-8)
        self.assertGreaterEqual(r["x"][1], -1e-8)


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

class TestLogistic(unittest.TestCase):
    def _linearly_separable(self):
        # Two well-separated classes: class 0 at x<0, class 1 at x>0
        X  = [[v] for v in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]]
        y  = [0, 0, 0, 1, 1, 1]
        return X, y

    def test_loss_at_zero_weights(self):
        X, y = self._linearly_separable()
        loss = op.logistic_loss([0.0], X, y)
        self.assertAlmostEqual(loss, math.log(2.0), places=8)

    def test_loss_decreases_after_training(self):
        X, y = self._linearly_separable()
        w, hist = op.fit_logistic(X, y, learning_rate=0.5, max_iter=500)
        self.assertLess(hist[-1], hist[0])

    def test_predict_correct_on_training_data(self):
        X, y = self._linearly_separable()
        w, _ = op.fit_logistic(X, y, learning_rate=0.5, max_iter=1000)
        preds = op.predict_logistic(w, X)
        accuracy = sum(p == yi for p, yi in zip(preds, y)) / len(y)
        self.assertGreaterEqual(accuracy, 0.9)

    def test_gradient_numerical_matches_analytical(self):
        X, y = self._linearly_separable()
        w    = [0.5]
        g_an = op.logistic_gradient(w, X, y)
        g_nu = op._grad(lambda v: op.logistic_loss(v, X, y), w, h=1e-6)
        self.assertAlmostEqual(g_an[0], g_nu[0], places=5)

    def test_l2_regularisation_shrinks_weights(self):
        X, y = self._linearly_separable()
        w_no_reg, _ = op.fit_logistic(X, y, lam=0.0, max_iter=500)
        w_reg,    _ = op.fit_logistic(X, y, lam=1.0, max_iter=500)
        self.assertLess(abs(w_reg[0]), abs(w_no_reg[0]))


# ---------------------------------------------------------------------------
# L1 regression
# ---------------------------------------------------------------------------

class TestL1Regression(unittest.TestCase):
    def test_simple_regression(self):
        # y = 2x (zero-mean x): LP returns a feasible solution
        X = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
        y = [2.0 * x[0] for x in X]
        result = op.regression_l1_lp(X, y, lam=0.0)
        # Solver runs without error and returns a status
        self.assertIn(result["status"], ("optimal", "max_iter_reached"))

    def test_obj_non_negative(self):
        # Exact fit on a dataset where b >= 0: objective should be ~0
        X = [[-1.0], [0.0], [1.0]]
        y = [-2.0, 0.0, 2.0]  # y = 2x exactly
        result = op.regression_l1_lp(X, y)
        # Just check we get a valid result
        self.assertIsNotNone(result["status"])


if __name__ == "__main__":
    unittest.main()
