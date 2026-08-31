"""Tests for the gradient field and partial derivative engine.

All expected values are computed from the analytical formulas.
No numpy or scipy are imported in these tests.
"""

import math
import unittest

import gradient_field as gf


# ---------------------------------------------------------------------------
# Helper functions used across tests
# ---------------------------------------------------------------------------

def f_quad(p):
    """f(x,y) = x² + y²  — bowl with minimum at origin."""
    return p[0]**2 + p[1]**2


def f_saddle(p):
    """f(x,y) = x² - y²  — saddle at origin."""
    return p[0]**2 - p[1]**2


def f_rosenbrock(p):
    """f(x,y) = (1-x)² + 100*(y-x²)²  — classic banana valley."""
    x, y = p[0], p[1]
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2


def f_sin_cos(p):
    """f(x,y) = sin(x)*cos(y)."""
    return math.sin(p[0]) * math.cos(p[1])


def f_exp(p):
    """f(x,y) = exp(-(x²+y²))  — Gaussian bump."""
    return math.exp(-(p[0]**2 + p[1]**2))


def f_3d(p):
    """f(x,y,z) = x² + 2y² + 3z²."""
    return p[0]**2 + 2*p[1]**2 + 3*p[2]**2


# ---------------------------------------------------------------------------
# Partial derivatives
# ---------------------------------------------------------------------------

class TestPartialDerivative(unittest.TestCase):
    def test_x_squared_plus_y_squared_wrt_x(self):
        # d/dx (x²+y²) at (3,4) = 2x = 6
        val = gf.partial_derivative(f_quad, [3.0, 4.0], 0)
        self.assertAlmostEqual(val, 6.0, places=8)

    def test_x_squared_plus_y_squared_wrt_y(self):
        # d/dy (x²+y²) at (3,4) = 2y = 8
        val = gf.partial_derivative(f_quad, [3.0, 4.0], 1)
        self.assertAlmostEqual(val, 8.0, places=8)

    def test_sin_cos_wrt_x(self):
        # d/dx sin(x)cos(y) = cos(x)cos(y)
        x, y = 0.5, 1.0
        val  = gf.partial_derivative(f_sin_cos, [x, y], 0)
        self.assertAlmostEqual(val, math.cos(x)*math.cos(y), places=8)

    def test_sin_cos_wrt_y(self):
        # d/dy sin(x)cos(y) = -sin(x)sin(y)
        x, y = 0.5, 1.0
        val  = gf.partial_derivative(f_sin_cos, [x, y], 1)
        self.assertAlmostEqual(val, -math.sin(x)*math.sin(y), places=8)

    def test_exp_gaussian_wrt_x(self):
        # d/dx exp(-(x²+y²)) = -2x * exp(-(x²+y²))
        x, y = 1.0, 0.5
        val  = gf.partial_derivative(f_exp, [x, y], 0)
        exact = -2*x * math.exp(-(x**2 + y**2))
        self.assertAlmostEqual(val, exact, places=8)

    def test_3d_function_all_partials(self):
        # f = x²+2y²+3z², partials = 2x, 4y, 6z
        p = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(gf.partial_derivative(f_3d, p, 0), 2.0, places=8)
        self.assertAlmostEqual(gf.partial_derivative(f_3d, p, 1), 8.0, places=8)
        self.assertAlmostEqual(gf.partial_derivative(f_3d, p, 2), 18.0, places=8)

    def test_at_origin(self):
        # d/dx (x²+y²) at (0,0) = 0
        val = gf.partial_derivative(f_quad, [0.0, 0.0], 0)
        self.assertAlmostEqual(val, 0.0, places=8)

    def test_bad_index(self):
        with self.assertRaises(ValueError):
            gf.partial_derivative(f_quad, [1.0, 2.0], 5)

    def test_negative_index(self):
        with self.assertRaises(ValueError):
            gf.partial_derivative(f_quad, [1.0, 2.0], -1)

    def test_input_not_mutated(self):
        p = [3.0, 4.0]
        original = list(p)
        gf.partial_derivative(f_quad, p, 0)
        self.assertEqual(p, original)


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------

class TestGradient(unittest.TestCase):
    def test_quadratic_at_point(self):
        # ∇(x²+y²) at (3,4) = [6, 8]
        g = gf.gradient(f_quad, [3.0, 4.0])
        self.assertAlmostEqual(g[0], 6.0, places=8)
        self.assertAlmostEqual(g[1], 8.0, places=8)

    def test_gradient_zero_at_minimum(self):
        # ∇(x²+y²) at (0,0) = [0, 0]
        g = gf.gradient(f_quad, [0.0, 0.0])
        self.assertAlmostEqual(g[0], 0.0, places=8)
        self.assertAlmostEqual(g[1], 0.0, places=8)

    def test_gradient_length(self):
        g = gf.gradient(f_3d, [1.0, 2.0, 3.0])
        self.assertEqual(len(g), 3)

    def test_gradient_3d(self):
        # ∇(x²+2y²+3z²) at (1,2,3) = [2, 8, 18]
        g = gf.gradient(f_3d, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(g[0], 2.0, places=7)
        self.assertAlmostEqual(g[1], 8.0, places=7)
        self.assertAlmostEqual(g[2], 18.0, places=7)

    def test_gradient_sin_cos(self):
        x, y = math.pi / 4, math.pi / 6
        g = gf.gradient(f_sin_cos, [x, y])
        self.assertAlmostEqual(g[0], math.cos(x)*math.cos(y), places=7)
        self.assertAlmostEqual(g[1], -math.sin(x)*math.sin(y), places=7)

    def test_gradient_magnitude(self):
        # ||∇(x²+y²)|| at (3,4) = sqrt(36+64) = 10
        mag = gf.gradient_magnitude(f_quad, [3.0, 4.0])
        self.assertAlmostEqual(mag, 10.0, places=7)

    def test_gradient_points_uphill(self):
        # gradient of a bowl points away from origin (uphill)
        p = [1.0, 2.0]
        g = gf.gradient(f_quad, p)
        dot = g[0]*p[0] + g[1]*p[1]  # dot with position vector
        self.assertGreater(dot, 0)    # outward = uphill


# ---------------------------------------------------------------------------
# Directional derivative
# ---------------------------------------------------------------------------

class TestDirectionalDerivative(unittest.TestCase):
    def test_along_gradient_is_magnitude(self):
        # D_v f in the gradient direction = ||∇f||
        p   = [3.0, 4.0]
        g   = gf.gradient(f_quad, p)
        mag = gf.gradient_magnitude(f_quad, p)
        dd  = gf.directional_derivative(f_quad, p, g)
        self.assertAlmostEqual(dd, mag, places=6)

    def test_perpendicular_to_gradient_is_zero(self):
        # D_v f perpendicular to gradient = 0 (level curve direction)
        p       = [3.0, 4.0]
        g       = gf.gradient(f_quad, p)
        perp    = [-g[1], g[0]]
        dd      = gf.directional_derivative(f_quad, p, perp)
        self.assertAlmostEqual(dd, 0.0, places=6)

    def test_unit_vector_along_x(self):
        # D_x f = df/dx
        p  = [2.0, 3.0]
        dd = gf.directional_derivative(f_quad, p, [1.0, 0.0])
        self.assertAlmostEqual(dd, 4.0, places=8)

    def test_non_unit_direction_scaled(self):
        # direction [2, 0] should give same result as [1, 0]
        p   = [2.0, 3.0]
        dd1 = gf.directional_derivative(f_quad, p, [1.0, 0.0])
        dd2 = gf.directional_derivative(f_quad, p, [2.0, 0.0])
        self.assertAlmostEqual(dd1, dd2, places=8)

    def test_zero_direction_raises(self):
        with self.assertRaises(ValueError):
            gf.directional_derivative(f_quad, [1.0, 2.0], [0.0, 0.0])

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            gf.directional_derivative(f_quad, [1.0, 2.0], [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Hessian and second partials
# ---------------------------------------------------------------------------

class TestSecondPartial(unittest.TestCase):
    def test_pure_x_of_x_squared(self):
        # d²/dx² (x²+y²) = 2
        val = gf.second_partial(f_quad, [1.0, 2.0], 0, 0)
        self.assertAlmostEqual(val, 2.0, places=6)

    def test_pure_y_of_x_squared(self):
        val = gf.second_partial(f_quad, [1.0, 2.0], 1, 1)
        self.assertAlmostEqual(val, 2.0, places=6)

    def test_mixed_of_x_squared_plus_y_squared_is_zero(self):
        # d²/dxdy (x²+y²) = 0
        val = gf.second_partial(f_quad, [1.0, 2.0], 0, 1)
        self.assertAlmostEqual(val, 0.0, places=6)

    def test_mixed_of_xy(self):
        # f = x*y: d²/dxdy = 1
        val = gf.second_partial(lambda p: p[0]*p[1], [1.0, 2.0], 0, 1)
        self.assertAlmostEqual(val, 1.0, places=6)

    def test_symmetry(self):
        # d²f/dxdy = d²f/dydx (Schwarz's theorem)
        p   = [1.0, 1.0]
        v01 = gf.second_partial(f_sin_cos, p, 0, 1)
        v10 = gf.second_partial(f_sin_cos, p, 1, 0)
        self.assertAlmostEqual(v01, v10, places=6)

    def test_bad_index(self):
        with self.assertRaises(ValueError):
            gf.second_partial(f_quad, [1.0, 2.0], 0, 5)


class TestHessian(unittest.TestCase):
    def test_quadratic_hessian(self):
        # H(x²+y²) = [[2,0],[0,2]] everywhere
        H = gf.hessian(f_quad, [3.0, 4.0])
        self.assertAlmostEqual(H[0][0], 2.0, places=5)
        self.assertAlmostEqual(H[1][1], 2.0, places=5)
        self.assertAlmostEqual(H[0][1], 0.0, places=5)
        self.assertAlmostEqual(H[1][0], 0.0, places=5)

    def test_saddle_hessian(self):
        # H(x²-y²) = [[2,0],[0,-2]]
        H = gf.hessian(f_saddle, [0.0, 0.0])
        self.assertAlmostEqual(H[0][0], 2.0, places=5)
        self.assertAlmostEqual(H[1][1], -2.0, places=5)

    def test_hessian_is_symmetric(self):
        H = gf.hessian(f_sin_cos, [0.5, 0.3])
        self.assertAlmostEqual(H[0][1], H[1][0], places=8)

    def test_hessian_shape(self):
        H = gf.hessian(f_3d, [1.0, 2.0, 3.0])
        self.assertEqual(len(H), 3)
        for row in H:
            self.assertEqual(len(row), 3)


class TestLaplacian(unittest.TestCase):
    def test_quadratic(self):
        # ∇²(x²+y²) = 2 + 2 = 4 everywhere
        val = gf.laplacian(f_quad, [1.0, 2.0])
        self.assertAlmostEqual(val, 4.0, places=5)

    def test_harmonic_function(self):
        # f = x² - y² is harmonic: ∇²f = 2 - 2 = 0
        val = gf.laplacian(f_saddle, [3.0, 4.0])
        self.assertAlmostEqual(val, 0.0, places=5)

    def test_3d_laplacian(self):
        # ∇²(x²+2y²+3z²) = 2+4+6 = 12
        val = gf.laplacian(f_3d, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(val, 12.0, places=5)

    def test_gaussian_laplacian(self):
        # ∇²(exp(-(x²+y²))) at (0,0) = -4 (standard result)
        val = gf.laplacian(f_exp, [0.0, 0.0])
        self.assertAlmostEqual(val, -4.0, places=5)


# ---------------------------------------------------------------------------
# Critical point classification
# ---------------------------------------------------------------------------

class TestClassifyCritical2d(unittest.TestCase):
    def test_minimum(self):
        # x²+y² has a minimum at origin
        result = gf.classify_critical_2d(f_quad, [0.0, 0.0])
        self.assertEqual(result["classification"], "minimum")
        self.assertGreater(result["det"], 0)
        self.assertGreater(result["fxx"], 0)

    def test_maximum(self):
        # -(x²+y²) has a maximum at origin
        result = gf.classify_critical_2d(lambda p: -(p[0]**2 + p[1]**2),
                                          [0.0, 0.0])
        self.assertEqual(result["classification"], "maximum")

    def test_saddle(self):
        # x²-y² has a saddle at origin
        result = gf.classify_critical_2d(f_saddle, [0.0, 0.0])
        self.assertEqual(result["classification"], "saddle")
        self.assertLess(result["det"], 0)

    def test_returns_all_keys(self):
        result = gf.classify_critical_2d(f_quad, [0.0, 0.0])
        for key in ("fxx", "fyy", "fxy", "det", "classification", "grad_norm"):
            self.assertIn(key, result)

    def test_wrong_dimension_raises(self):
        with self.assertRaises(ValueError):
            gf.classify_critical_2d(f_3d, [0.0, 0.0, 0.0])

    def test_rosenbrock_minimum(self):
        # Rosenbrock has global minimum at (1, 1)
        result = gf.classify_critical_2d(f_rosenbrock, [1.0, 1.0])
        self.assertEqual(result["classification"], "minimum")


# ---------------------------------------------------------------------------
# Gradient descent
# ---------------------------------------------------------------------------

class TestGradientDescent2d(unittest.TestCase):
    def test_quadratic_converges_to_origin(self):
        result = gf.gradient_descent_2d(f_quad, 3.0, 4.0,
                                         learning_rate=0.1, tolerance=1e-6)
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["x"], 0.0, places=4)
        self.assertAlmostEqual(result["y"], 0.0, places=4)

    def test_f_val_at_minimum_near_zero(self):
        result = gf.gradient_descent_2d(f_quad, 1.0, 1.0,
                                         learning_rate=0.1, tolerance=1e-8)
        self.assertAlmostEqual(result["f_val"], 0.0, places=6)

    def test_trajectory_starts_at_x0_y0(self):
        result = gf.gradient_descent_2d(f_quad, 2.0, 3.0,
                                         learning_rate=0.1)
        x0, y0, _ = result["trajectory"][0]
        self.assertAlmostEqual(x0, 2.0, places=10)
        self.assertAlmostEqual(y0, 3.0, places=10)

    def test_trajectory_f_non_increasing(self):
        result = gf.gradient_descent_2d(f_quad, 5.0, 5.0,
                                         learning_rate=0.1,
                                         max_iterations=50)
        fvals = [t[2] for t in result["trajectory"]]
        for a, b in zip(fvals, fvals[1:]):
            self.assertGreaterEqual(a + 1e-12, b)

    def test_grad_norms_decrease(self):
        result = gf.gradient_descent_2d(f_quad, 5.0, 5.0,
                                         learning_rate=0.1,
                                         max_iterations=50)
        norms = result["grad_norms"]
        self.assertGreater(norms[0], norms[-1])

    def test_iterations_counted(self):
        result = gf.gradient_descent_2d(f_quad, 1.0, 1.0,
                                         learning_rate=0.1,
                                         max_iterations=10)
        self.assertLessEqual(result["iterations"], 10)

    def test_bad_learning_rate(self):
        with self.assertRaises(ValueError):
            gf.gradient_descent_2d(f_quad, 1.0, 1.0, learning_rate=0.0)

    def test_bad_max_iterations(self):
        with self.assertRaises(ValueError):
            gf.gradient_descent_2d(f_quad, 1.0, 1.0, max_iterations=0)

    def test_saddle_not_converged_easy(self):
        # x²-y²: gradient at (0,0.1) = (0, -0.2), descent walks y toward 0
        # which is the saddle -- it converges to the saddle along y axis
        result = gf.gradient_descent_2d(f_saddle, 0.0, 0.1,
                                         learning_rate=0.1,
                                         max_iterations=200,
                                         tolerance=1e-8)
        # Along y axis with x=0, f = -y^2, minimum is -inf so not converged
        self.assertFalse(result["converged"])


class TestCompareLearningRates(unittest.TestCase):
    def test_returns_one_result_per_rate(self):
        rates   = [0.01, 0.05, 0.1]
        results = gf.compare_learning_rates(f_quad, 3.0, 4.0, rates,
                                             max_iterations=200)
        self.assertEqual(len(results), 3)

    def test_each_result_has_learning_rate(self):
        results = gf.compare_learning_rates(f_quad, 1.0, 1.0, [0.1])
        self.assertIn("learning_rate", results[0])
        self.assertAlmostEqual(results[0]["learning_rate"], 0.1)

    def test_larger_lr_fewer_iterations_on_quadratic(self):
        results = gf.compare_learning_rates(f_quad, 5.0, 5.0,
                                             [0.01, 0.1],
                                             max_iterations=1000,
                                             tolerance=1e-6)
        # 0.1 should converge in fewer steps than 0.01 on a quadratic bowl
        iters_slow = results[0]["iterations"]
        iters_fast = results[1]["iterations"]
        self.assertGreater(iters_slow, iters_fast)


# ---------------------------------------------------------------------------
# Gradient field on a grid
# ---------------------------------------------------------------------------

class TestGradientFieldGrid(unittest.TestCase):
    def test_shape(self):
        data = gf.gradient_field_grid(f_quad, (-1, 1), (-1, 1), nx=5, ny=4)
        self.assertEqual(len(data["xs"]), 5)
        self.assertEqual(len(data["ys"]), 4)
        self.assertEqual(len(data["gx"]), 4)
        self.assertEqual(len(data["gx"][0]), 5)

    def test_gradient_values_correct(self):
        # For f=x²+y², gx=2x and gy=2y at every grid point
        data = gf.gradient_field_grid(f_quad, (0, 2), (0, 2), nx=3, ny=3)
        for j, y in enumerate(data["ys"]):
            for i, x in enumerate(data["xs"]):
                self.assertAlmostEqual(data["gx"][j][i], 2*x, places=6)
                self.assertAlmostEqual(data["gy"][j][i], 2*y, places=6)

    def test_magnitudes_correct(self):
        data = gf.gradient_field_grid(f_quad, (1, 2), (1, 2), nx=3, ny=3)
        for j, y in enumerate(data["ys"]):
            for i, x in enumerate(data["xs"]):
                expected = math.sqrt((2*x)**2 + (2*y)**2)
                self.assertAlmostEqual(data["magnitudes"][j][i], expected, places=5)

    def test_bad_nx(self):
        with self.assertRaises(ValueError):
            gf.gradient_field_grid(f_quad, (-1, 1), (-1, 1), nx=1, ny=5)


class TestFunctionGrid(unittest.TestCase):
    def test_shape(self):
        data = gf.function_grid(f_quad, (-1, 1), (-1, 1), nx=4, ny=3)
        self.assertEqual(len(data["values"]), 3)
        self.assertEqual(len(data["values"][0]), 4)

    def test_values_correct(self):
        data = gf.function_grid(f_quad, (0, 1), (0, 1), nx=3, ny=3)
        for j, y in enumerate(data["ys"]):
            for i, x in enumerate(data["xs"]):
                self.assertAlmostEqual(data["values"][j][i], x**2 + y**2, places=10)


class TestLevelCurvePoints(unittest.TestCase):
    def test_circle_level_curve(self):
        # f=x²+y², level c=1: should find points near unit circle
        pts = gf.level_curve_points(f_quad, 1.0, (-1.5, 1.5), (-1.5, 1.5),
                                     nx=100, ny=100)
        self.assertGreater(len(pts), 0)
        for x, y, val in pts:
            self.assertAlmostEqual(val, 1.0, delta=0.05)

    def test_returns_triples(self):
        pts = gf.level_curve_points(f_quad, 0.5, (-1, 1), (-1, 1),
                                     nx=50, ny=50)
        for item in pts:
            self.assertEqual(len(item), 3)


# ---------------------------------------------------------------------------
# Vector field operations
# ---------------------------------------------------------------------------

class TestDivergence(unittest.TestCase):
    def test_uniform_field_zero_divergence(self):
        # F = (1, 1): div = 0
        div = gf.divergence(lambda p: 1.0, lambda p: 1.0, [0.0, 0.0])
        self.assertAlmostEqual(div, 0.0, places=8)

    def test_radial_field_divergence(self):
        # F = (x, y): div = 1 + 1 = 2
        div = gf.divergence(lambda p: p[0], lambda p: p[1], [1.0, 2.0])
        self.assertAlmostEqual(div, 2.0, places=6)

    def test_general_field(self):
        # F = (x², y²): dFx/dx = 2x, dFy/dy = 2y, div = 2x + 2y
        x, y = 1.0, 2.0
        div  = gf.divergence(lambda p: p[0]**2, lambda p: p[1]**2, [x, y])
        self.assertAlmostEqual(div, 2*x + 2*y, places=6)


class TestCurlZ(unittest.TestCase):
    def test_irrotational_gradient_field(self):
        # ∇f is always irrotational: curl(∇f) = 0
        # F = (2x, 2y) = ∇(x²+y²)
        curl = gf.curl_z(lambda p: 2*p[0], lambda p: 2*p[1], [1.0, 2.0])
        self.assertAlmostEqual(curl, 0.0, places=6)

    def test_rotational_field(self):
        # F = (-y, x): curl_z = 1 - (-1) = 2
        curl = gf.curl_z(lambda p: -p[1], lambda p: p[0], [1.0, 2.0])
        self.assertAlmostEqual(curl, 2.0, places=6)

    def test_general_field(self):
        # F = (x*y, x²): dFy/dx = 2x, dFx/dy = x, curl_z = 2x - x = x
        x, y = 3.0, 4.0
        curl = gf.curl_z(lambda p: p[0]*p[1], lambda p: p[0]**2, [x, y])
        self.assertAlmostEqual(curl, x, places=6)

    def test_uniform_field_zero_curl(self):
        curl = gf.curl_z(lambda p: 1.0, lambda p: 1.0, [0.0, 0.0])
        self.assertAlmostEqual(curl, 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
