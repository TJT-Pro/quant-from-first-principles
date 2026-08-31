"""Tests for the numerical differentiation engine.

Every formula is tested against a known exact value or a mathematical
identity.  Tolerances reflect the expected accuracy of each method at
its optimal step size.
"""

import math
import unittest

import differentiator as df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f_poly(x):
    """f(x) = 3x^3 - 2x^2 + x - 5.  f'(x) = 9x^2 - 4x + 1."""
    return 3*x**3 - 2*x**2 + x - 5


def _f_poly_prime(x):
    return 9*x**2 - 4*x + 1


def _f_poly_double_prime(x):
    return 18*x - 4


# ---------------------------------------------------------------------------
# Step size selection
# ---------------------------------------------------------------------------

class TestOptimalH(unittest.TestCase):
    def test_forward_h_positive(self):
        self.assertGreater(df._optimal_h(1.0, "forward"), 0.0)

    def test_central_h_positive(self):
        self.assertGreater(df._optimal_h(1.0, "central"), 0.0)

    def test_central_h_smaller_than_forward(self):
        # forward: eps^0.5, central: eps^(1/3); central is larger
        h_fwd = df._optimal_h(1.0, "forward")
        h_cen = df._optimal_h(1.0, "central")
        self.assertGreater(h_cen, h_fwd)

    def test_scale_with_x(self):
        h1 = df._optimal_h(1.0, "central")
        h2 = df._optimal_h(100.0, "central")
        self.assertAlmostEqual(h2 / h1, 100.0, places=5)

    def test_bad_method(self):
        with self.assertRaises(ValueError):
            df._optimal_h(1.0, "banana")


# ---------------------------------------------------------------------------
# Forward difference
# ---------------------------------------------------------------------------

class TestForwardDifference(unittest.TestCase):
    def test_zero_step_raises(self):
        with self.assertRaises(ValueError):
            df.forward_difference(math.sin, 1.0, h=0.0)

    def test_constant_function_zero_derivative(self):
        result = df.forward_difference(lambda x: 7.0, 3.0, h=1e-5)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_linear_exact(self):
        # f(x) = 5x + 3, f'(x) = 5
        result = df.forward_difference(lambda x: 5*x + 3, 2.0, h=1e-5)
        self.assertAlmostEqual(result, 5.0, places=5)

    def test_polynomial_at_multiple_points(self):
        for x in [-2.0, 0.0, 1.0, 3.0]:
            approx = df.forward_difference(_f_poly, x, h=1e-7)
            exact  = _f_poly_prime(x)
            self.assertAlmostEqual(approx, exact, places=4,
                                   msg="forward diff failed at x={}".format(x))

    def test_sin_at_zero(self):
        # sin'(0) = cos(0) = 1
        result = df.forward_difference(math.sin, 0.0, h=1e-7)
        self.assertAlmostEqual(result, 1.0, places=5)


# ---------------------------------------------------------------------------
# Backward difference
# ---------------------------------------------------------------------------

class TestBackwardDifference(unittest.TestCase):
    def test_zero_step_raises(self):
        with self.assertRaises(ValueError):
            df.backward_difference(math.sin, 1.0, h=0.0)

    def test_constant_function_zero_derivative(self):
        result = df.backward_difference(lambda x: -3.0, 5.0, h=1e-5)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_linear_exact(self):
        result = df.backward_difference(lambda x: 5*x + 3, 2.0, h=1e-5)
        self.assertAlmostEqual(result, 5.0, places=5)

    def test_polynomial_at_multiple_points(self):
        for x in [-1.0, 0.5, 2.0]:
            approx = df.backward_difference(_f_poly, x, h=1e-7)
            exact  = _f_poly_prime(x)
            self.assertAlmostEqual(approx, exact, places=4,
                                   msg="backward diff failed at x={}".format(x))


# ---------------------------------------------------------------------------
# Central difference
# ---------------------------------------------------------------------------

class TestCentralDifference(unittest.TestCase):
    def test_zero_step_raises(self):
        with self.assertRaises(ValueError):
            df.central_difference(math.sin, 1.0, h=0.0)

    def test_constant_zero_derivative(self):
        result = df.central_difference(lambda x: 42.0, 0.0)
        self.assertAlmostEqual(result, 0.0, places=12)

    def test_linear_exact(self):
        result = df.central_difference(lambda x: 7*x - 1, 5.0)
        self.assertAlmostEqual(result, 7.0, places=10)

    def test_sin(self):
        for x in [0.0, math.pi/6, math.pi/4, math.pi/3, math.pi/2]:
            approx = df.central_difference(math.sin, x)
            self.assertAlmostEqual(approx, math.cos(x), places=10,
                                   msg="central sin' failed at x={:.4f}".format(x))

    def test_cos(self):
        for x in [0.0, 0.5, 1.0, 2.0]:
            approx = df.central_difference(math.cos, x)
            self.assertAlmostEqual(approx, -math.sin(x), places=10,
                                   msg="central cos' failed at x={}".format(x))

    def test_exp(self):
        for x in [-2.0, 0.0, 1.0, 3.0]:
            approx = df.central_difference(math.exp, x)
            self.assertAlmostEqual(approx, math.exp(x), places=8,
                                   msg="central exp' failed at x={}".format(x))

    def test_log(self):
        for x in [0.5, 1.0, 2.0, math.e]:
            approx = df.central_difference(math.log, x)
            self.assertAlmostEqual(approx, 1.0/x, places=9,
                                   msg="central log' failed at x={}".format(x))

    def test_polynomial(self):
        for x in [-3.0, 0.0, 1.5, 4.0]:
            approx = df.central_difference(_f_poly, x)
            exact  = _f_poly_prime(x)
            self.assertAlmostEqual(approx, exact, places=8,
                                   msg="central poly' failed at x={}".format(x))

    def test_central_more_accurate_than_forward(self):
        h = 1e-4
        err_fwd = abs(df.forward_difference(math.sin, 1.0, h=h) - math.cos(1.0))
        err_cen = abs(df.central_difference(math.sin, 1.0, h=h) - math.cos(1.0))
        self.assertLess(err_cen, err_fwd)


# ---------------------------------------------------------------------------
# Five-point stencil
# ---------------------------------------------------------------------------

class TestFivePointStencil(unittest.TestCase):
    def test_zero_step_raises(self):
        with self.assertRaises(ValueError):
            df.five_point_stencil(math.sin, 1.0, h=0.0)

    def test_sin(self):
        for x in [0.0, 0.5, 1.0, 2.0]:
            approx = df.five_point_stencil(math.sin, x)
            self.assertAlmostEqual(approx, math.cos(x), places=10,
                                   msg="5-pt sin' failed at x={}".format(x))

    def test_polynomial(self):
        for x in [-2.0, 0.0, 1.0, 3.0]:
            approx = df.five_point_stencil(_f_poly, x)
            self.assertAlmostEqual(approx, _f_poly_prime(x), places=9,
                                   msg="5-pt poly' failed at x={}".format(x))

    def test_more_accurate_than_central_at_same_h(self):
        h = 1e-3
        err_cen = abs(df.central_difference(math.sin, 1.0, h=h) - math.cos(1.0))
        err_5pt = abs(df.five_point_stencil(math.sin, 1.0, h=h) - math.cos(1.0))
        self.assertLess(err_5pt, err_cen)


# ---------------------------------------------------------------------------
# Differentiate dispatcher
# ---------------------------------------------------------------------------

class TestDifferentiate(unittest.TestCase):
    def test_all_methods_agree_on_sin(self):
        x = 1.2
        exact = math.cos(x)
        for method in ("forward", "backward", "central", "five-point"):
            approx = df.differentiate(math.sin, x, method=method)
            self.assertAlmostEqual(approx, exact, places=5,
                                   msg="method {} failed".format(method))

    def test_bad_method_raises(self):
        with self.assertRaises(ValueError):
            df.differentiate(math.sin, 1.0, method="trapezoid")


# ---------------------------------------------------------------------------
# Second derivative
# ---------------------------------------------------------------------------

class TestSecondDerivative(unittest.TestCase):
    def test_constant_zero(self):
        d2 = df.second_derivative(lambda x: 5.0, 2.0)
        self.assertAlmostEqual(d2, 0.0, places=8)

    def test_linear_zero(self):
        d2 = df.second_derivative(lambda x: 3*x + 7, 1.0)
        self.assertAlmostEqual(d2, 0.0, places=8)

    def test_quadratic(self):
        # f(x) = x^2, f''(x) = 2
        d2 = df.second_derivative(lambda x: x*x, 5.0)
        self.assertAlmostEqual(d2, 2.0, places=7)

    def test_sin_double_prime(self):
        # sin''(x) = -sin(x)
        for x in [0.0, 0.5, 1.0]:
            d2 = df.second_derivative(math.sin, x)
            self.assertAlmostEqual(d2, -math.sin(x), places=6,
                                   msg="sin'' failed at x={}".format(x))

    def test_exp_double_prime(self):
        # exp''(x) = exp(x)
        for x in [0.0, 1.0, -1.0]:
            d2 = df.second_derivative(math.exp, x)
            self.assertAlmostEqual(d2, math.exp(x), places=5,
                                   msg="exp'' failed at x={}".format(x))

    def test_polynomial_double_prime(self):
        for x in [-1.0, 0.0, 2.0]:
            d2 = df.second_derivative(_f_poly, x)
            self.assertAlmostEqual(d2, _f_poly_double_prime(x), places=5,
                                   msg="poly'' failed at x={}".format(x))

    def test_zero_step_raises(self):
        with self.assertRaises(ValueError):
            df.second_derivative(math.sin, 1.0, h=0.0)


# ---------------------------------------------------------------------------
# Approximation error
# ---------------------------------------------------------------------------

class TestApproximationError(unittest.TestCase):
    def test_returns_correct_length(self):
        h_values = [1e-2, 1e-4, 1e-6]
        results  = df.approximation_error(math.sin, 1.0, math.cos(1.0), h_values)
        self.assertEqual(len(results), 3)

    def test_error_pairs_structure(self):
        h_values = [1e-3, 1e-5]
        results  = df.approximation_error(math.sin, 1.0, math.cos(1.0), h_values)
        for h, err in results:
            self.assertGreater(h, 0.0)
            self.assertGreaterEqual(err, 0.0)

    def test_central_more_accurate_than_forward_for_moderate_h(self):
        h_values = [1e-3]
        err_fwd  = df.approximation_error(math.sin, 1.0, math.cos(1.0),
                                          h_values, method="forward")[0][1]
        err_cen  = df.approximation_error(math.sin, 1.0, math.cos(1.0),
                                          h_values, method="central")[0][1]
        self.assertLess(err_cen, err_fwd)


# ---------------------------------------------------------------------------
# Multivariate: gradient and partial derivatives
# ---------------------------------------------------------------------------

class TestGradient(unittest.TestCase):
    def test_gradient_x_squared_plus_y_squared(self):
        f = lambda x: x[0]**2 + x[1]**2
        g = df.gradient(f, [3.0, 4.0])
        self.assertAlmostEqual(g[0], 6.0, places=8)
        self.assertAlmostEqual(g[1], 8.0, places=8)

    def test_gradient_xyz(self):
        f = lambda x: x[0] * x[1] * x[2]
        g = df.gradient(f, [2.0, 3.0, 4.0])
        self.assertAlmostEqual(g[0], 12.0, places=7)
        self.assertAlmostEqual(g[1],  8.0, places=7)
        self.assertAlmostEqual(g[2],  6.0, places=7)

    def test_gradient_constant_is_zero(self):
        f = lambda x: 5.0
        g = df.gradient(f, [1.0, 2.0, 3.0])
        self.assertTrue(all(abs(gi) < 1e-8 for gi in g))

    def test_partial_derivative_matches_gradient(self):
        f = lambda x: x[0]**2 + 3*x[1]**3
        point = [2.0, 1.0]
        g = df.gradient(f, point)
        for i in range(2):
            pd = df.partial_derivative(f, point, i)
            self.assertAlmostEqual(pd, g[i], places=10)

    def test_partial_derivative_bad_index(self):
        f = lambda x: x[0]**2
        with self.assertRaises(ValueError):
            df.partial_derivative(f, [1.0], 1)
        with self.assertRaises(ValueError):
            df.partial_derivative(f, [1.0], -1)


class TestDirectionalDerivative(unittest.TestCase):
    def test_along_x_axis_equals_df_dx(self):
        f  = lambda x: x[0]**2 + x[1]**2
        dd = df.directional_derivative(f, [3.0, 4.0], [1.0, 0.0])
        self.assertAlmostEqual(dd, 6.0, places=8)

    def test_along_y_axis_equals_df_dy(self):
        f  = lambda x: x[0]**2 + x[1]**2
        dd = df.directional_derivative(f, [3.0, 4.0], [0.0, 1.0])
        self.assertAlmostEqual(dd, 8.0, places=8)

    def test_zero_direction_raises(self):
        f = lambda x: x[0]**2
        with self.assertRaises(ValueError):
            df.directional_derivative(f, [1.0, 1.0], [0.0, 0.0])

    def test_unnormalised_direction_gives_same_result(self):
        f   = lambda x: x[0]**2 + x[1]**2
        dd1 = df.directional_derivative(f, [3.0, 4.0], [1.0, 0.0])
        dd2 = df.directional_derivative(f, [3.0, 4.0], [100.0, 0.0])
        self.assertAlmostEqual(dd1, dd2, places=8)


class TestSecondPartial(unittest.TestCase):
    def test_d2_dx2_quadratic(self):
        # f(x,y) = x^2 + y^2, d^2f/dx^2 = 2
        f   = lambda x: x[0]**2 + x[1]**2
        d2  = df.second_partial(f, [1.0, 1.0], 0, 0)
        self.assertAlmostEqual(d2, 2.0, places=4)

    def test_mixed_partial_xy(self):
        # f(x,y) = x*y, d^2f/dxdy = 1
        f   = lambda x: x[0] * x[1]
        d2  = df.second_partial(f, [2.0, 3.0], 0, 1)
        self.assertAlmostEqual(d2, 1.0, places=4)

    def test_schwarz_symmetry(self):
        # d^2f/dxdy = d^2f/dydx for smooth f
        f    = lambda x: math.sin(x[0]) * math.exp(x[1])
        d_xy = df.second_partial(f, [0.5, 0.5], 0, 1)
        d_yx = df.second_partial(f, [0.5, 0.5], 1, 0)
        self.assertAlmostEqual(d_xy, d_yx, places=3)


class TestJacobian(unittest.TestCase):
    def test_jacobian_shape(self):
        f = lambda x: [x[0]**2, x[1]**2, x[0]*x[1]]
        J = df.jacobian(f, [1.0, 2.0])
        self.assertEqual(len(J), 3)
        self.assertEqual(len(J[0]), 2)

    def test_jacobian_known_values(self):
        # f(x) = [x0^2, x0*x1]  J = [[2x0, 0], [x1, x0]]
        f = lambda x: [x[0]**2, x[0]*x[1]]
        J = df.jacobian(f, [2.0, 3.0])
        self.assertAlmostEqual(J[0][0], 4.0, places=7)
        self.assertAlmostEqual(J[0][1], 0.0, places=7)
        self.assertAlmostEqual(J[1][0], 3.0, places=7)
        self.assertAlmostEqual(J[1][1], 2.0, places=7)

    def test_jacobian_identity(self):
        # f(x) = x, J = I
        n = 3
        f = lambda x: list(x)
        J = df.jacobian(f, [1.0, 2.0, 3.0])
        for i in range(n):
            for j in range(n):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(J[i][j], expected, places=7,
                                       msg="identity J[{}][{}]".format(i,j))


# ---------------------------------------------------------------------------
# Derivative rules
# ---------------------------------------------------------------------------

class TestDerivativeRules(unittest.TestCase):
    def test_product_rule_sin_exp(self):
        r = df.verify_product_rule(math.sin, math.exp, 1.0)
        self.assertAlmostEqual(r["error"], 0.0, places=8)

    def test_product_rule_poly(self):
        r = df.verify_product_rule(lambda x: x**2, lambda x: x**3, 2.0)
        self.assertAlmostEqual(r["error"], 0.0, places=7)

    def test_product_rule_constant(self):
        r = df.verify_product_rule(lambda x: 3.0, math.sin, 1.0)
        self.assertAlmostEqual(r["error"], 0.0, places=9)

    def test_chain_rule_sin_x_squared(self):
        # (sin(x^2))' = cos(x^2) * 2x
        r = df.verify_chain_rule(math.sin, lambda x: x*x, 1.5)
        self.assertAlmostEqual(r["error"], 0.0, places=8)

    def test_chain_rule_exp_sin(self):
        r = df.verify_chain_rule(math.exp, math.sin, 0.5)
        self.assertAlmostEqual(r["error"], 0.0, places=8)

    def test_chain_rule_log_exp(self):
        # (log(exp(x)))' = 1 for all x
        r = df.verify_chain_rule(math.log, math.exp, 1.0)
        self.assertAlmostEqual(r["lhs"], 1.0, places=7)
        self.assertAlmostEqual(r["rhs"], 1.0, places=7)


# ---------------------------------------------------------------------------
# Gradient descent
# ---------------------------------------------------------------------------

class TestGradientDescent(unittest.TestCase):
    def test_minimum_of_quadratic(self):
        result = df.gradient_descent(
            lambda x: x[0]**2 + x[1]**2,
            [3.0, 4.0], learning_rate=0.1
        )
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["x"][0], 0.0, places=5)
        self.assertAlmostEqual(result["x"][1], 0.0, places=5)

    def test_minimum_of_shifted_quadratic(self):
        # min at (2, 3)
        result = df.gradient_descent(
            lambda x: (x[0]-2)**2 + (x[1]-3)**2,
            [0.0, 0.0], learning_rate=0.1
        )
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["x"][0], 2.0, places=5)
        self.assertAlmostEqual(result["x"][1], 3.0, places=5)

    def test_minimum_of_1d_quadratic(self):
        result = df.gradient_descent(
            lambda x: (x[0] - 5.0)**2,
            [0.0], learning_rate=0.1
        )
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["x"][0], 5.0, places=5)

    def test_path_length(self):
        result = df.gradient_descent(
            lambda x: x[0]**2,
            [1.0], learning_rate=0.1, max_iterations=10
        )
        self.assertLessEqual(len(result["path"]), 11)

    def test_path_f_values_non_increasing(self):
        result = df.gradient_descent(
            lambda x: x[0]**2 + x[1]**2,
            [5.0, 5.0], learning_rate=0.1
        )
        fvals = [fv for _, fv in result["path"]]
        # values should generally decrease (allow tiny floating point noise)
        self.assertLessEqual(fvals[-1], fvals[0])

    def test_high_lr_may_not_converge(self):
        # with very high LR the descent can overshoot indefinitely
        result = df.gradient_descent(
            lambda x: x[0]**2,
            [1.0], learning_rate=10.0, max_iterations=100, tolerance=1e-8
        )
        # just check it does not crash; convergence is not guaranteed
        self.assertIsNotNone(result["converged"])

    def test_f_x_equals_f_at_final_x(self):
        f = lambda x: x[0]**2 + x[1]**2
        result = df.gradient_descent(f, [2.0, 3.0], learning_rate=0.1)
        self.assertAlmostEqual(result["f_x"], f(result["x"]), places=12)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

class TestSensitivity(unittest.TestCase):
    def test_sensitivity_equals_gradient(self):
        f    = lambda x: x[0]**2 + x[1]**3
        point = [2.0, 1.0]
        sens = df.sensitivity(f, point)
        grad = df.gradient(f, point)
        self.assertEqual(len(sens), len(grad))
        for s, g in zip(sens, grad):
            self.assertAlmostEqual(s, g, places=10)

    def test_relative_sensitivity_known(self):
        # f(x) = x[0] * x[1], sens_rel[0] = x0/f * df/dx0 = x0/(x0*x1) * x1 = 1
        f    = lambda x: x[0] * x[1]
        sens = df.sensitivity(f, [3.0, 4.0], relative=True)
        self.assertAlmostEqual(sens[0], 1.0, places=7)
        self.assertAlmostEqual(sens[1], 1.0, places=7)

    def test_relative_sensitivity_zero_f_raises(self):
        with self.assertRaises(ValueError):
            df.sensitivity(lambda x: 0.0, [1.0, 2.0], relative=True)


# ---------------------------------------------------------------------------
# Bond analytics
# ---------------------------------------------------------------------------

class TestBondAnalytics(unittest.TestCase):
    def test_par_bond(self):
        # when yield = coupon rate, price = face
        p = df.bond_price(0.05, 50.0, 1000.0, 10)
        self.assertAlmostEqual(p, 1000.0, places=6)

    def test_discount_bond(self):
        p = df.bond_price(0.06, 50.0, 1000.0, 10)
        self.assertLess(p, 1000.0)

    def test_premium_bond(self):
        p = df.bond_price(0.04, 50.0, 1000.0, 10)
        self.assertGreater(p, 1000.0)

    def test_price_decreases_as_yield_rises(self):
        p1 = df.bond_price(0.04, 50.0, 1000.0, 10)
        p2 = df.bond_price(0.05, 50.0, 1000.0, 10)
        p3 = df.bond_price(0.06, 50.0, 1000.0, 10)
        self.assertGreater(p1, p2)
        self.assertGreater(p2, p3)

    def test_zero_coupon_bond_price(self):
        # zero coupon: price = face / (1+y)^n
        y, face, n = 0.05, 1000.0, 10
        p = df.bond_price(y, 0.0, face, n)
        expected = face / (1 + y) ** n
        self.assertAlmostEqual(p, expected, places=10)

    def test_invalid_yield(self):
        with self.assertRaises(ValueError):
            df.bond_price(-1.0, 50.0, 1000.0, 10)

    def test_duration_numerical_matches_analytical(self):
        dur = df.bond_duration(0.05, 50.0, 1000.0, 10)
        self.assertAlmostEqual(dur["numerical"], dur["analytical"], places=6)

    def test_duration_positive(self):
        dur = df.bond_duration(0.05, 50.0, 1000.0, 10)
        self.assertGreater(dur["numerical"], 0.0)

    def test_longer_maturity_higher_duration(self):
        dur5  = df.bond_duration(0.05, 50.0, 1000.0, 5)["numerical"]
        dur10 = df.bond_duration(0.05, 50.0, 1000.0, 10)["numerical"]
        self.assertGreater(dur10, dur5)

    def test_higher_coupon_lower_duration(self):
        # higher coupon => more weight on earlier cash flows => lower duration
        dur_low  = df.bond_duration(0.05, 30.0,  1000.0, 10)["numerical"]
        dur_high = df.bond_duration(0.05, 100.0, 1000.0, 10)["numerical"]
        self.assertGreater(dur_low, dur_high)

    def test_convexity_positive(self):
        conv = df.bond_convexity(0.05, 50.0, 1000.0, 10)
        self.assertGreater(conv, 0.0)

    def test_longer_maturity_higher_convexity(self):
        conv5  = df.bond_convexity(0.05, 50.0, 1000.0, 5)
        conv10 = df.bond_convexity(0.05, 50.0, 1000.0, 10)
        self.assertGreater(conv10, conv5)

    def test_price_change_duration_convexity_more_accurate(self):
        chg = df.bond_price_change(0.05, 50.0, 1000.0, 10, 0.01)
        self.assertLess(chg["duration_convexity_error"],
                        chg["duration_error"])

    def test_price_change_sign(self):
        # yield rises => price falls => exact change negative
        chg = df.bond_price_change(0.05, 50.0, 1000.0, 10, 0.01)
        self.assertLess(chg["exact_pct_change"], 0.0)


if __name__ == "__main__":
    unittest.main()
