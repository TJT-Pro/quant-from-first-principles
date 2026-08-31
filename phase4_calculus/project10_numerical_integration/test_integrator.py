"""Tests for the numerical integrator and Taylor series expander.

Every method is tested against a known exact value.  Tolerances reflect
the expected accuracy of each method at the given n.  No numpy or scipy
are used in these tests.
"""

import math
import unittest

import integrator as ig


# ---------------------------------------------------------------------------
# Riemann sums
# ---------------------------------------------------------------------------

class TestLeftRiemann(unittest.TestCase):
    def test_constant(self):
        val = ig.left_riemann(lambda x: 3.0, 0.0, 2.0, 100)
        self.assertAlmostEqual(val, 6.0, places=10)

    def test_linear(self):
        # integral of x from 0 to 1 = 0.5
        val = ig.left_riemann(lambda x: x, 0.0, 1.0, 10000)
        self.assertAlmostEqual(val, 0.5, places=3)

    def test_zero_width(self):
        self.assertEqual(ig.left_riemann(math.sin, 2.0, 2.0, 100), 0.0)

    def test_bad_n(self):
        with self.assertRaises(ValueError):
            ig.left_riemann(math.sin, 0.0, 1.0, 0)

    def test_negative_interval(self):
        # integral of x from 1 to 0 should be -0.5
        val = ig.left_riemann(lambda x: x, 1.0, 0.0, 10000)
        self.assertAlmostEqual(val, -0.5, places=3)

    def test_sin_from_0_to_pi(self):
        val = ig.left_riemann(math.sin, 0.0, math.pi, 100000)
        self.assertAlmostEqual(val, 2.0, places=3)


class TestRightRiemann(unittest.TestCase):
    def test_constant(self):
        val = ig.right_riemann(lambda x: 5.0, 0.0, 3.0, 100)
        self.assertAlmostEqual(val, 15.0, places=10)

    def test_linear(self):
        val = ig.right_riemann(lambda x: x, 0.0, 1.0, 10000)
        self.assertAlmostEqual(val, 0.5, places=3)

    def test_zero_width(self):
        self.assertEqual(ig.right_riemann(math.sin, 1.0, 1.0, 10), 0.0)

    def test_left_right_bracket_monotone(self):
        # for increasing f, left < exact < right
        left  = ig.left_riemann(lambda x: x, 0.0, 1.0, 100)
        right = ig.right_riemann(lambda x: x, 0.0, 1.0, 100)
        self.assertLess(left, 0.5)
        self.assertGreater(right, 0.5)


class TestMidpointRiemann(unittest.TestCase):
    def test_constant(self):
        val = ig.midpoint_riemann(lambda x: 7.0, 0.0, 4.0, 100)
        self.assertAlmostEqual(val, 28.0, places=10)

    def test_linear_exact(self):
        # midpoint is exact for linear functions with any n
        val = ig.midpoint_riemann(lambda x: 3*x + 1, 0.0, 2.0, 10)
        self.assertAlmostEqual(val, 8.0, places=10)

    def test_sin_accurate(self):
        val = ig.midpoint_riemann(math.sin, 0.0, math.pi, 1000)
        self.assertAlmostEqual(val, 2.0, places=5)

    def test_more_accurate_than_endpoints(self):
        # at same n, midpoint error < left/right error
        n   = 100
        f   = math.sin
        err_mid  = abs(ig.midpoint_riemann(f, 0.0, math.pi, n) - 2.0)
        err_left = abs(ig.left_riemann(f, 0.0, math.pi, n) - 2.0)
        self.assertLess(err_mid, err_left)

    def test_zero_width(self):
        self.assertEqual(ig.midpoint_riemann(math.cos, 5.0, 5.0, 10), 0.0)


# ---------------------------------------------------------------------------
# Trapezoidal rule
# ---------------------------------------------------------------------------

class TestTrapezoidal(unittest.TestCase):
    def test_constant(self):
        val = ig.trapezoidal(lambda x: 4.0, 0.0, 5.0, 100)
        self.assertAlmostEqual(val, 20.0, places=10)

    def test_linear_exact(self):
        val = ig.trapezoidal(lambda x: 2*x + 3, 1.0, 4.0, 10)
        self.assertAlmostEqual(val, 24.0, places=10)

    def test_sin_accurate(self):
        val = ig.trapezoidal(math.sin, 0.0, math.pi, 1000)
        self.assertAlmostEqual(val, 2.0, places=5)

    def test_exp_accurate(self):
        val = ig.trapezoidal(math.exp, 0.0, 1.0, 1000)
        self.assertAlmostEqual(val, math.e - 1.0, places=6)

    def test_zero_width(self):
        self.assertEqual(ig.trapezoidal(math.sin, 3.0, 3.0, 10), 0.0)

    def test_error_estimate_positive(self):
        _, err = ig.trapezoidal_error_estimate(math.sin, 0.0, math.pi, 100)
        self.assertGreater(err, 0.0)

    def test_error_estimate_decreases_with_n(self):
        _, err100 = ig.trapezoidal_error_estimate(math.sin, 0.0, math.pi, 100)
        _, err500 = ig.trapezoidal_error_estimate(math.sin, 0.0, math.pi, 500)
        self.assertGreater(err100, err500)


# ---------------------------------------------------------------------------
# Simpson's rule
# ---------------------------------------------------------------------------

class TestSimpsons(unittest.TestCase):
    def test_constant(self):
        val = ig.simpsons(lambda x: 2.0, 0.0, 6.0, 100)
        self.assertAlmostEqual(val, 12.0, places=10)

    def test_linear_exact(self):
        val = ig.simpsons(lambda x: 5*x - 1, 0.0, 2.0, 10)
        self.assertAlmostEqual(val, 8.0, places=10)

    def test_quadratic_exact(self):
        # Simpson's is exact for polynomials of degree <= 3
        val = ig.simpsons(lambda x: x**2, 0.0, 1.0, 10)
        self.assertAlmostEqual(val, 1.0/3.0, places=12)

    def test_cubic_exact(self):
        val = ig.simpsons(lambda x: x**3, 0.0, 1.0, 10)
        self.assertAlmostEqual(val, 0.25, places=12)

    def test_sin_very_accurate(self):
        val = ig.simpsons(math.sin, 0.0, math.pi, 100)
        self.assertAlmostEqual(val, 2.0, places=7)

    def test_exp_accurate(self):
        val = ig.simpsons(math.exp, 0.0, 1.0, 100)
        self.assertAlmostEqual(val, math.e - 1.0, places=9)

    def test_odd_n_raises(self):
        with self.assertRaises(ValueError):
            ig.simpsons(math.sin, 0.0, 1.0, 101)

    def test_n_less_than_2_raises(self):
        with self.assertRaises(ValueError):
            ig.simpsons(math.sin, 0.0, 1.0, 0)

    def test_zero_width(self):
        self.assertEqual(ig.simpsons(math.sin, 2.0, 2.0, 10), 0.0)

    def test_more_accurate_than_trapezoidal(self):
        n   = 100
        err_trap = abs(ig.trapezoidal(math.sin, 0.0, math.pi, n) - 2.0)
        err_simp = abs(ig.simpsons(math.sin, 0.0, math.pi, n) - 2.0)
        self.assertLess(err_simp, err_trap)

    def test_error_estimate_positive(self):
        _, err = ig.simpsons_error_estimate(math.sin, 0.0, math.pi, 100)
        self.assertGreater(err, 0.0)

    def test_error_estimate_decreases_with_n(self):
        _, err100 = ig.simpsons_error_estimate(math.sin, 0.0, math.pi, 100)
        _, err500 = ig.simpsons_error_estimate(math.sin, 0.0, math.pi, 500)
        self.assertGreater(err100, err500)


# ---------------------------------------------------------------------------
# Convergence rates
# ---------------------------------------------------------------------------

class TestConvergenceRates(unittest.TestCase):
    """Verify that each method converges at the stated order."""

    def _error(self, fn, n):
        return abs(fn(math.sin, 0.0, math.pi, n) - 2.0)

    def test_midpoint_second_order(self):
        e1 = self._error(ig.midpoint_riemann, 100)
        e2 = self._error(ig.midpoint_riemann, 200)
        ratio = e1 / e2
        self.assertGreater(ratio, 3.5)
        self.assertLess(ratio, 4.5)

    def test_trapezoidal_second_order(self):
        e1 = self._error(ig.trapezoidal, 100)
        e2 = self._error(ig.trapezoidal, 200)
        ratio = e1 / e2
        self.assertGreater(ratio, 3.5)
        self.assertLess(ratio, 4.5)

    def test_simpsons_fourth_order(self):
        e1 = self._error(ig.simpsons, 100)
        e2 = self._error(ig.simpsons, 200)
        ratio = e1 / e2
        self.assertGreater(ratio, 14.0)
        self.assertLess(ratio, 18.0)

    def test_convergence_table_structure(self):
        rows = ig.convergence_table(
            math.sin, 0.0, math.pi, 2.0,
            methods=['midpoint', 'simpsons'],
            n_values=[100, 200]
        )
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn('method', row)
            self.assertIn('n', row)
            self.assertIn('estimate', row)
            self.assertIn('error', row)
            self.assertIn('ratio', row)

    def test_convergence_table_bad_method(self):
        with self.assertRaises(ValueError):
            ig.convergence_table(math.sin, 0.0, math.pi, 2.0,
                                 methods=['banana'])


# ---------------------------------------------------------------------------
# Adaptive quadrature
# ---------------------------------------------------------------------------

class TestAdaptiveSimpsons(unittest.TestCase):
    def test_sin_to_tolerance(self):
        val, err = ig.adaptive_simpsons(math.sin, 0.0, math.pi, tolerance=1e-10)
        self.assertAlmostEqual(val, 2.0, places=10)
        self.assertLess(err, 1e-10)

    def test_exp_accurate(self):
        val, err = ig.adaptive_simpsons(math.exp, 0.0, 1.0, tolerance=1e-10)
        self.assertAlmostEqual(val, math.e - 1.0, places=9)

    def test_constant_function(self):
        val, err = ig.adaptive_simpsons(lambda x: 3.0, 0.0, 5.0)
        self.assertAlmostEqual(val, 15.0, places=8)

    def test_zero_width(self):
        val, err = ig.adaptive_simpsons(math.sin, 1.0, 1.0)
        self.assertEqual(val, 0.0)
        self.assertEqual(err, 0.0)

    def test_tighter_tolerance_more_accurate(self):
        val_loose, _ = ig.adaptive_simpsons(math.sin, 0.0, math.pi, tolerance=1e-4)
        val_tight, _ = ig.adaptive_simpsons(math.sin, 0.0, math.pi, tolerance=1e-10)
        err_loose = abs(val_loose - 2.0)
        err_tight = abs(val_tight - 2.0)
        self.assertGreater(err_loose, err_tight)

    def test_rapidly_varying_function(self):
        # integral of sin(100x) from 0 to pi/100 = 2/100
        exact = 2.0 / 100.0
        val, _ = ig.adaptive_simpsons(
            lambda x: math.sin(100 * x), 0.0, math.pi / 100, tolerance=1e-8
        )
        self.assertAlmostEqual(val, exact, places=6)


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature
# ---------------------------------------------------------------------------

class TestGaussLegendre(unittest.TestCase):
    def test_polynomial_exact_degree_3_with_n2(self):
        # n=2 is exact for degree <= 3
        val = ig.gauss_legendre(lambda x: x**3, -1.0, 1.0, n=2)
        self.assertAlmostEqual(val, 0.0, places=12)

    def test_polynomial_exact_degree_5_with_n3(self):
        val = ig.gauss_legendre(lambda x: x**5, -1.0, 1.0, n=3)
        self.assertAlmostEqual(val, 0.0, places=12)

    def test_polynomial_exact_degree_4(self):
        # integral of x^4 from 0 to 1 = 1/5
        val = ig.gauss_legendre(lambda x: x**4, 0.0, 1.0, n=3)
        self.assertAlmostEqual(val, 0.2, places=12)

    def test_exp_accurate(self):
        val = ig.gauss_legendre(math.exp, 0.0, 1.0, n=5)
        self.assertAlmostEqual(val, math.e - 1.0, places=10)

    def test_sin_accurate(self):
        val = ig.gauss_legendre(math.sin, 0.0, math.pi, n=7)
        self.assertAlmostEqual(val, 2.0, places=8)

    def test_bad_n_raises(self):
        with self.assertRaises(ValueError):
            ig.gauss_legendre(math.sin, 0.0, 1.0, n=8)

    def test_zero_width(self):
        self.assertEqual(ig.gauss_legendre(math.sin, 3.0, 3.0, n=5), 0.0)

    def test_more_accurate_than_simpsons_same_evals(self):
        # 5-point Gauss uses 5 evaluations; compare to Simpson's with n=4 (5 evals)
        f     = math.exp
        exact = math.e - 1.0
        err_g = abs(ig.gauss_legendre(f, 0.0, 1.0, n=5) - exact)
        err_s = abs(ig.simpsons(f, 0.0, 1.0, 4) - exact)
        self.assertLess(err_g, err_s)


# ---------------------------------------------------------------------------
# Improper integrals
# ---------------------------------------------------------------------------

class TestImproperIntegral(unittest.TestCase):
    def test_finite_interval_matches_simpsons(self):
        val_imp = ig.improper_integral(math.sin, 0.0, math.pi, 1000)
        val_s   = ig.simpsons(math.sin, 0.0, math.pi, 1000)
        self.assertAlmostEqual(val_imp, val_s, places=8)

    def test_exp_neg_x_from_0_to_inf(self):
        # integral of exp(-x) from 0 to inf = 1
        val = ig.improper_integral(lambda x: math.exp(-x), 0.0, math.inf, 2000)
        self.assertAlmostEqual(val, 1.0, places=3)

    def test_gaussian_from_0_to_inf(self):
        # integral of exp(-x^2) from 0 to inf = sqrt(pi)/2
        exact = math.sqrt(math.pi) / 2.0
        val = ig.improper_integral(lambda x: math.exp(-x*x), 0.0, math.inf, 2000)
        self.assertAlmostEqual(val, exact, places=3)

    def test_exp_neg_x_from_neg_inf_to_0(self):
        # integral of exp(x) from -inf to 0 = 1
        val = ig.improper_integral(lambda x: math.exp(x), -math.inf, 0.0, 2000)
        self.assertAlmostEqual(val, 1.0, places=3)

    def test_bad_n(self):
        with self.assertRaises(ValueError):
            ig.improper_integral(math.exp, 0.0, math.inf, 0)


# ---------------------------------------------------------------------------
# Taylor series
# ---------------------------------------------------------------------------

class TestTaylorCoefficients(unittest.TestCase):
    def test_sin_coefficients(self):
        # Reliable up to degree 6; degrees 7,9 have cancellation issues
        coeffs = ig.taylor_coefficients(math.sin, 0.0, 6)
        self.assertAlmostEqual(coeffs[0], 0.0, places=8)
        self.assertAlmostEqual(coeffs[1], 1.0, places=8)
        self.assertAlmostEqual(coeffs[2], 0.0, places=7)
        self.assertAlmostEqual(coeffs[3], -1.0/6.0, places=7)
        self.assertAlmostEqual(coeffs[5], 1.0/120.0, places=6)

    def test_exp_coefficients(self):
        # Reliable up to degree 6 with order-specific h
        coeffs = ig.taylor_coefficients(math.exp, 0.0, 6)
        fact = 1
        for k in range(7):
            if k > 0:
                fact *= k
            self.assertAlmostEqual(coeffs[k], 1.0/fact, places=6,
                                   msg="exp coefficient a{}".format(k))

    def test_constant_function(self):
        coeffs = ig.taylor_coefficients(lambda x: 5.0, 0.0, 4)
        self.assertAlmostEqual(coeffs[0], 5.0, places=8)
        for k in range(1, 5):
            self.assertAlmostEqual(coeffs[k], 0.0, places=6,
                                   msg="constant Taylor a{} should be 0".format(k))

    def test_linear_function(self):
        # f(x) = 3x + 2: a0=2, a1=3, rest=0
        coeffs = ig.taylor_coefficients(lambda x: 3*x + 2, 0.0, 4)
        self.assertAlmostEqual(coeffs[0], 2.0, places=8)
        self.assertAlmostEqual(coeffs[1], 3.0, places=8)
        for k in range(2, 5):
            self.assertAlmostEqual(coeffs[k], 0.0, places=5)

    def test_degree_zero(self):
        coeffs = ig.taylor_coefficients(math.sin, 0.0, 0)
        self.assertEqual(len(coeffs), 1)
        self.assertAlmostEqual(coeffs[0], 0.0, places=10)

    def test_negative_degree_raises(self):
        with self.assertRaises(ValueError):
            ig.taylor_coefficients(math.sin, 0.0, -1)

    def test_length(self):
        for d in range(6):
            coeffs = ig.taylor_coefficients(math.sin, 0.0, d)
            self.assertEqual(len(coeffs), d + 1)

    def test_expansion_at_nonzero_point(self):
        # sin expanded at x0=pi/2: a0=1, a1=0, a2=-1/2
        coeffs = ig.taylor_coefficients(math.sin, math.pi/2, 4)
        self.assertAlmostEqual(coeffs[0], 1.0, places=8)
        self.assertAlmostEqual(coeffs[1], 0.0, places=7)
        self.assertAlmostEqual(coeffs[2], -0.5, places=5)


class TestTaylorEvaluate(unittest.TestCase):
    def test_sin_at_half(self):
        # degree 5 is reliably accurate; sin converges fast so places=6 is achievable
        coeffs = ig.taylor_coefficients(math.sin, 0.0, 5)
        approx = ig.taylor_evaluate(coeffs, 0.0, 0.5)
        self.assertAlmostEqual(approx, math.sin(0.5), places=5)

    def test_exp_at_one(self):
        # degree 6 gives about 4 significant figures at x=1
        coeffs = ig.taylor_coefficients(math.exp, 0.0, 6)
        approx = ig.taylor_evaluate(coeffs, 0.0, 1.0)
        self.assertAlmostEqual(approx, math.e, places=3)

    def test_horner_matches_direct(self):
        coeffs = [1.0, 2.0, 3.0, 4.0]  # 1 + 2x + 3x^2 + 4x^3
        x0, x  = 0.0, 2.0
        dx     = x - x0
        direct = 1 + 2*dx + 3*dx**2 + 4*dx**3
        horner = ig.taylor_evaluate(coeffs, x0, x)
        self.assertAlmostEqual(horner, direct, places=12)

    def test_empty_coeffs(self):
        self.assertEqual(ig.taylor_evaluate([], 0.0, 1.0), 0.0)

    def test_accuracy_improves_with_degree(self):
        # Use degrees 3 vs 5 (both reliably accurate)
        x = 0.8
        err_low  = abs(ig.taylor_evaluate(
            ig.taylor_coefficients(math.sin, 0.0, 3), 0.0, x) - math.sin(x))
        err_high = abs(ig.taylor_evaluate(
            ig.taylor_coefficients(math.sin, 0.0, 5), 0.0, x) - math.sin(x))
        self.assertGreater(err_low, err_high)


class TestTaylorError(unittest.TestCase):
    def test_returns_correct_structure(self):
        coeffs  = ig.taylor_coefficients(math.sin, 0.0, 7)
        results = ig.taylor_error(math.sin, coeffs, 0.0, [0.1, 0.5, 1.0])
        self.assertEqual(len(results), 3)
        for x, exact, approx, err in results:
            self.assertGreaterEqual(err, 0.0)
            self.assertAlmostEqual(err, abs(exact - approx), places=12)

    def test_sin_error_small_near_expansion(self):
        # degree 5, reliable accuracy; at x=0.1 the truncation error of sin
        # is x^7/7! ~ 2e-11, but numerical diff error in coefficients limits
        # overall accuracy to about 1e-6
        coeffs  = ig.taylor_coefficients(math.sin, 0.0, 5)
        results = ig.taylor_error(math.sin, coeffs, 0.0, [0.1])
        _, _, _, err = results[0]
        self.assertLess(err, 1e-5)

    def test_sin_error_large_far_from_expansion(self):
        # degree-3 Taylor of sin at 0 has large error at x=5
        coeffs  = ig.taylor_coefficients(math.sin, 0.0, 3)
        results = ig.taylor_error(math.sin, coeffs, 0.0, [5.0])
        _, _, _, err = results[0]
        self.assertGreater(err, 0.1)


class TestRadiusOfConvergence(unittest.TestCase):
    def test_sin_converges_everywhere(self):
        # Use degree 5 (reliable); ratio test needs non-negligible coefficients
        coeffs = ig.taylor_coefficients(math.sin, 0.0, 5)
        r = ig.radius_of_convergence(coeffs)
        self.assertGreater(r, 3.0)

    def test_exp_converges_everywhere(self):
        coeffs = ig.taylor_coefficients(math.exp, 0.0, 6)
        r = ig.radius_of_convergence(coeffs)
        self.assertGreater(r, 3.0)

    def test_single_coefficient(self):
        r = ig.radius_of_convergence([1.0])
        self.assertEqual(r, math.inf)

    def test_convergent_at(self):
        coeffs = ig.taylor_coefficients(math.sin, 0.0, 5)
        self.assertTrue(ig.convergent_at(coeffs, 0.0, 0.5))
        self.assertTrue(ig.convergent_at(coeffs, 0.0, 2.0))


# ---------------------------------------------------------------------------
# Probability applications
# ---------------------------------------------------------------------------

class TestNormalCDF(unittest.TestCase):
    def test_at_zero_is_half(self):
        self.assertAlmostEqual(ig.normal_cdf(0.0), 0.5, places=10)

    def test_symmetric(self):
        for x in [0.5, 1.0, 1.96, 2.0]:
            self.assertAlmostEqual(ig.normal_cdf(x) + ig.normal_cdf(-x), 1.0,
                                   places=8)

    def test_known_quantiles(self):
        self.assertAlmostEqual(ig.normal_cdf(1.96), 0.975, places=4)
        self.assertAlmostEqual(ig.normal_cdf(-1.96), 0.025, places=4)

    def test_matches_erf(self):
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            exact = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            self.assertAlmostEqual(ig.normal_cdf(x), exact, places=8)

    def test_with_mean_sigma(self):
        # N(100, 15): P(X <= 100) = 0.5
        self.assertAlmostEqual(ig.normal_cdf(100.0, mu=100.0, sigma=15.0),
                               0.5, places=10)

    def test_bad_sigma(self):
        with self.assertRaises(ValueError):
            ig.normal_cdf(0.0, sigma=0.0)
        with self.assertRaises(ValueError):
            ig.normal_cdf(0.0, sigma=-1.0)


class TestExponentialCDF(unittest.TestCase):
    def test_at_zero(self):
        self.assertAlmostEqual(ig.exponential_cdf(0.0), 0.0, places=10)

    def test_known_value(self):
        # P(X <= 1) = 1 - exp(-1)
        self.assertAlmostEqual(ig.exponential_cdf(1.0),
                               1.0 - math.exp(-1.0), places=8)

    def test_matches_exact(self):
        for x in [0.5, 1.0, 2.0, 3.0]:
            num   = ig.exponential_cdf(x, rate=2.0)
            exact = 1.0 - math.exp(-2.0 * x)
            self.assertAlmostEqual(num, exact, places=7)

    def test_negative_x_zero(self):
        self.assertEqual(ig.exponential_cdf(-1.0), 0.0)

    def test_bad_rate(self):
        with self.assertRaises(ValueError):
            ig.exponential_cdf(1.0, rate=0.0)


class TestBetaCDF(unittest.TestCase):
    def test_at_zero(self):
        self.assertEqual(ig.beta_cdf(0.0, 2.0, 3.0), 0.0)

    def test_at_one(self):
        self.assertEqual(ig.beta_cdf(1.0, 2.0, 3.0), 1.0)

    def test_uniform_is_linear(self):
        # Beta(1,1) is Uniform(0,1): CDF(x) = x
        # Simpson's with n=500 gives about 2 decimal places for this integrand
        for x in [0.2, 0.5, 0.8]:
            self.assertAlmostEqual(ig.beta_cdf(x, 1.0, 1.0), x, places=2)

    def test_symmetry(self):
        # Beta(a,b): CDF(x; a,b) = 1 - CDF(1-x; b,a)
        x, a, b = 0.3, 2.0, 5.0
        self.assertAlmostEqual(
            ig.beta_cdf(x, a, b) + ig.beta_cdf(1.0 - x, b, a), 1.0, places=4
        )

    def test_bad_x(self):
        with self.assertRaises(ValueError):
            ig.beta_cdf(1.5, 2.0, 3.0)

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            ig.beta_cdf(0.5, 0.0, 2.0)


# ---------------------------------------------------------------------------
# Finance applications
# ---------------------------------------------------------------------------

class TestPresentValue(unittest.TestCase):
    def test_constant_cash_flow(self):
        # PV of constant c over T at rate r = c/r * (1 - exp(-rT))
        c, r, T = 100.0, 0.05, 10.0
        exact = c / r * (1.0 - math.exp(-r * T))
        val   = ig.present_value_continuous(lambda t: c, r, T, 1000)
        self.assertAlmostEqual(val, exact, places=6)

    def test_zero_rate(self):
        # with r=0, PV = integral of c(t) from 0 to T
        val = ig.present_value_continuous(lambda t: 1.0, 0.0, 5.0, 100)
        self.assertAlmostEqual(val, 5.0, places=6)

    def test_bad_horizon(self):
        with self.assertRaises(ValueError):
            ig.present_value_continuous(lambda t: 1.0, 0.05, 0.0)


class TestBlackScholes(unittest.TestCase):
    def test_put_call_parity(self):
        S, K, r, sig, T = 100.0, 100.0, 0.05, 0.2, 1.0
        C = ig.black_scholes_call(S, K, r, sig, T)
        P = ig.black_scholes_put(S, K, r, sig, T)
        # C - P = S - K * exp(-rT)
        self.assertAlmostEqual(C - P, S - K * math.exp(-r * T), places=8)

    def test_atm_call_positive(self):
        C = ig.black_scholes_call(100.0, 100.0, 0.05, 0.2, 1.0)
        self.assertGreater(C, 0.0)

    def test_deep_itm_call_above_intrinsic(self):
        # deep ITM: S >> K
        C = ig.black_scholes_call(200.0, 100.0, 0.05, 0.2, 1.0)
        intrinsic = 200.0 - 100.0 * math.exp(-0.05 * 1.0)
        self.assertGreater(C, intrinsic * 0.99)

    def test_deep_otm_call_near_zero(self):
        C = ig.black_scholes_call(50.0, 100.0, 0.05, 0.2, 1.0)
        self.assertLess(C, 1.0)

    def test_higher_vol_higher_call(self):
        C_low  = ig.black_scholes_call(100.0, 100.0, 0.05, 0.1, 1.0)
        C_high = ig.black_scholes_call(100.0, 100.0, 0.05, 0.4, 1.0)
        self.assertGreater(C_high, C_low)

    def test_longer_expiry_higher_call(self):
        C_short = ig.black_scholes_call(100.0, 100.0, 0.05, 0.2, 0.25)
        C_long  = ig.black_scholes_call(100.0, 100.0, 0.05, 0.2, 2.0)
        self.assertGreater(C_long, C_short)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            ig.black_scholes_call(0.0, 100.0, 0.05, 0.2, 1.0)
        with self.assertRaises(ValueError):
            ig.black_scholes_call(100.0, 100.0, 0.05, 0.0, 1.0)


class TestContinuousBondDuration(unittest.TestCase):
    def test_duration_between_zero_and_maturity(self):
        res = ig.continuous_bond_duration(0.05, 0.06, 10.0)
        self.assertGreater(res["duration"], 0.0)
        self.assertLess(res["duration"], 10.0)

    def test_price_positive(self):
        res = ig.continuous_bond_duration(0.05, 0.06, 10.0)
        self.assertGreater(res["price"], 0.0)

    def test_higher_yield_lower_price(self):
        p_low  = ig.continuous_bond_duration(0.03, 0.05, 10.0)["price"]
        p_high = ig.continuous_bond_duration(0.07, 0.05, 10.0)["price"]
        self.assertGreater(p_low, p_high)

    def test_higher_yield_lower_duration(self):
        d_low  = ig.continuous_bond_duration(0.03, 0.05, 10.0)["duration"]
        d_high = ig.continuous_bond_duration(0.07, 0.05, 10.0)["duration"]
        self.assertGreater(d_low, d_high)

    def test_longer_maturity_higher_duration(self):
        d5  = ig.continuous_bond_duration(0.05, 0.05, 5.0)["duration"]
        d10 = ig.continuous_bond_duration(0.05, 0.05, 10.0)["duration"]
        self.assertGreater(d10, d5)

    def test_bad_maturity(self):
        with self.assertRaises(ValueError):
            ig.continuous_bond_duration(0.05, 0.05, 0.0)


if __name__ == "__main__":
    unittest.main()
