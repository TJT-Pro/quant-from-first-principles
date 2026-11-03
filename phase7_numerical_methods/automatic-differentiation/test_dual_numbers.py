import math
import unittest

import dual_numbers as dn
from dual_numbers import Dual


class TestDualArithmetic(unittest.TestCase):
    def test_addition(self):
        a = Dual(2.0, 1.0)
        b = Dual(3.0, 0.0)
        result = a + b
        self.assertEqual(result.real, 5.0)
        self.assertEqual(result.dual, 1.0)

    def test_subtraction(self):
        a = Dual(5.0, 2.0)
        b = Dual(3.0, 1.0)
        result = a - b
        self.assertEqual(result.real, 2.0)
        self.assertEqual(result.dual, 1.0)

    def test_multiplication_product_rule(self):
        # f(x) = x * x^2 at x=3: f(3)=27, f'(x)=3x^2 -> f'(3)=27
        x = Dual(3.0, 1.0)
        result = x * (x * x)
        self.assertAlmostEqual(result.real, 27.0)
        self.assertAlmostEqual(result.dual, 27.0)

    def test_division_quotient_rule(self):
        # f(x) = 1/x at x=2: f(2)=0.5, f'(x)=-1/x^2 -> f'(2)=-0.25
        x = Dual(2.0, 1.0)
        result = Dual(1.0, 0.0) / x
        self.assertAlmostEqual(result.real, 0.5)
        self.assertAlmostEqual(result.dual, -0.25)

    def test_division_by_zero_real_part_raises(self):
        with self.assertRaises(ZeroDivisionError):
            Dual(1.0, 0.0) / Dual(0.0, 1.0)

    def test_negation(self):
        result = -Dual(3.0, 2.0)
        self.assertEqual(result.real, -3.0)
        self.assertEqual(result.dual, -2.0)

    def test_power_rule(self):
        # f(x) = x^4 at x=2: f(2)=16, f'(x)=4x^3 -> f'(2)=32
        x = Dual(2.0, 1.0)
        result = x ** 4
        self.assertAlmostEqual(result.real, 16.0)
        self.assertAlmostEqual(result.dual, 32.0)

    def test_coercion_with_plain_numbers(self):
        x = Dual(3.0, 1.0)
        result = x + 5
        self.assertAlmostEqual(result.real, 8.0)
        self.assertAlmostEqual(result.dual, 1.0)
        result2 = 5 + x
        self.assertAlmostEqual(result2.real, 8.0)

    def test_dual_valued_exponent_not_supported(self):
        with self.assertRaises(NotImplementedError):
            Dual(2.0, 1.0) ** Dual(3.0, 1.0)


class TestElementaryFunctions(unittest.TestCase):
    def test_sin_derivative_is_cos(self):
        x = Dual(math.pi / 4, 1.0)
        result = dn.sin(x)
        self.assertAlmostEqual(result.real, math.sin(math.pi / 4))
        self.assertAlmostEqual(result.dual, math.cos(math.pi / 4))

    def test_cos_derivative_is_negative_sin(self):
        x = Dual(math.pi / 3, 1.0)
        result = dn.cos(x)
        self.assertAlmostEqual(result.dual, -math.sin(math.pi / 3))

    def test_exp_is_its_own_derivative(self):
        x = Dual(1.5, 1.0)
        result = dn.exp(x)
        self.assertAlmostEqual(result.real, result.dual)

    def test_log_derivative_is_reciprocal(self):
        x = Dual(4.0, 1.0)
        result = dn.log(x)
        self.assertAlmostEqual(result.dual, 0.25)

    def test_log_nonpositive_raises(self):
        with self.assertRaises(ValueError):
            dn.log(Dual(-1.0, 1.0))

    def test_sqrt_derivative(self):
        x = Dual(9.0, 1.0)
        result = dn.sqrt(x)
        self.assertAlmostEqual(result.real, 3.0)
        self.assertAlmostEqual(result.dual, 1.0 / 6.0)

    def test_sqrt_negative_raises(self):
        with self.assertRaises(ValueError):
            dn.sqrt(Dual(-4.0, 1.0))

    def test_tanh_derivative(self):
        x = Dual(0.5, 1.0)
        result = dn.tanh(x)
        expected = 1 - math.tanh(0.5) ** 2
        self.assertAlmostEqual(result.dual, expected)


class TestDerivativeHelper(unittest.TestCase):
    def test_derivative_of_polynomial(self):
        # f(x) = x^3 - 2x^2 + 5, f'(x) = 3x^2 - 4x, f'(3) = 27 - 12 = 15
        f = lambda x: x ** 3 - Dual(2.0) * x ** 2 + Dual(5.0)
        self.assertAlmostEqual(dn.derivative(f, 3.0), 15.0)

    def test_derivative_of_composite_function(self):
        # f(x) = sin(x^2), f'(x) = 2x*cos(x^2)
        f = lambda x: dn.sin(x * x)
        x0 = 1.3
        expected = 2 * x0 * math.cos(x0 ** 2)
        self.assertAlmostEqual(dn.derivative(f, x0), expected, places=8)

    def test_derivative_matches_finite_difference(self):
        f = lambda x: dn.exp(x) * dn.sin(x)
        x0 = 0.7
        h = 1e-6
        finite_diff = (f(Dual(x0 + h)).real - f(Dual(x0 - h)).real) / (2 * h)
        exact = dn.derivative(f, x0)
        self.assertAlmostEqual(exact, finite_diff, places=5)


class TestGradientForwardMode(unittest.TestCase):
    def test_gradient_of_sum_of_squares(self):
        # f(x,y,z) = x^2 + y^2 + z^2, gradient = (2x, 2y, 2z)
        f = lambda v: v[0] ** 2 + v[1] ** 2 + v[2] ** 2
        grad = dn.gradient_forward_mode(f, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(grad[0], 2.0)
        self.assertAlmostEqual(grad[1], 4.0)
        self.assertAlmostEqual(grad[2], 6.0)

    def test_gradient_of_product(self):
        # f(x,y) = x*y, gradient = (y, x)
        f = lambda v: v[0] * v[1]
        grad = dn.gradient_forward_mode(f, [3.0, 5.0])
        self.assertAlmostEqual(grad[0], 5.0)
        self.assertAlmostEqual(grad[1], 3.0)

    def test_gradient_matches_finite_difference(self):
        f = lambda v: dn.sin(v[0]) * dn.exp(v[1]) + v[0] * v[1]
        x = [0.6, -0.3]
        grad = dn.gradient_forward_mode(f, x)
        h = 1e-6
        for i in range(2):
            x_up = [Dual(xi) for xi in x]
            x_down = [Dual(xi) for xi in x]
            x_up[i] = Dual(x[i] + h)
            x_down[i] = Dual(x[i] - h)
            numerical = (f(x_up).real - f(x_down).real) / (2 * h)
            self.assertAlmostEqual(grad[i], numerical, places=4)


class TestJacobianForwardMode(unittest.TestCase):
    def test_jacobian_shape(self):
        # f: R^2 -> R^3
        f = lambda v: [v[0] + v[1], v[0] * v[1], v[0] ** 2]
        jac = dn.jacobian_forward_mode(f, [2.0, 3.0])
        self.assertEqual(len(jac), 3)
        self.assertEqual(len(jac[0]), 2)

    def test_jacobian_known_values(self):
        # f(x,y) = (x+y, x*y): J = [[1, 1], [y, x]]
        f = lambda v: [v[0] + v[1], v[0] * v[1]]
        jac = dn.jacobian_forward_mode(f, [2.0, 3.0])
        self.assertAlmostEqual(jac[0][0], 1.0)
        self.assertAlmostEqual(jac[0][1], 1.0)
        self.assertAlmostEqual(jac[1][0], 3.0)
        self.assertAlmostEqual(jac[1][1], 2.0)

    def test_jacobian_matches_finite_difference(self):
        f = lambda v: [dn.sin(v[0]) + v[1], dn.exp(v[1]) * v[0]]
        x = [0.5, 1.2]
        jac = dn.jacobian_forward_mode(f, x)
        h = 1e-6
        for j in range(2):
            x_up = list(x)
            x_down = list(x)
            x_up[j] += h
            x_down[j] -= h
            out_up = f([Dual(v) for v in x_up])
            out_down = f([Dual(v) for v in x_down])
            for row in range(2):
                numerical = (out_up[row].real - out_down[row].real) / (2 * h)
                self.assertAlmostEqual(jac[row][j], numerical, places=4)


if __name__ == "__main__":
    unittest.main()
