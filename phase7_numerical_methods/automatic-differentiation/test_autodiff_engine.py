import math
import unittest

import dual_numbers as dn
from autodiff_engine import Value, jacobian_reverse_mode


class TestValueArithmetic(unittest.TestCase):
    def test_addition_gradient(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        self.assertAlmostEqual(a.grad, 1.0)
        self.assertAlmostEqual(b.grad, 1.0)

    def test_multiplication_gradient(self):
        a = Value(3.0)
        b = Value(4.0)
        c = a * b
        c.backward()
        self.assertAlmostEqual(a.grad, 4.0)
        self.assertAlmostEqual(b.grad, 3.0)

    def test_shared_node_accumulates_gradient(self):
        # f(x) = x * x = x^2, df/dx = 2x, evaluated via two separate
        # multiplication edges from the same node x, which must
        # accumulate rather than overwrite.
        x = Value(5.0)
        y = x * x
        y.backward()
        self.assertAlmostEqual(x.grad, 10.0)

    def test_power_gradient(self):
        x = Value(2.0)
        y = x ** 3
        y.backward()
        self.assertAlmostEqual(y.data, 8.0)
        self.assertAlmostEqual(x.grad, 12.0)  # 3*x^2 = 12

    def test_division_gradient(self):
        # f(x,y) = x/y, df/dx = 1/y, df/dy = -x/y^2
        x = Value(6.0)
        y = Value(3.0)
        z = x / y
        z.backward()
        self.assertAlmostEqual(z.data, 2.0)
        self.assertAlmostEqual(x.grad, 1.0 / 3.0)
        self.assertAlmostEqual(y.grad, -6.0 / 9.0)

    def test_negation_and_subtraction(self):
        x = Value(5.0)
        y = Value(2.0)
        z = x - y
        z.backward()
        self.assertAlmostEqual(z.data, 3.0)
        self.assertAlmostEqual(x.grad, 1.0)
        self.assertAlmostEqual(y.grad, -1.0)

    def test_zero_grad_resets(self):
        x = Value(3.0)
        y = x * x
        y.backward()
        self.assertNotEqual(x.grad, 0.0)
        x.zero_grad()
        self.assertEqual(x.grad, 0.0)


class TestElementaryFunctions(unittest.TestCase):
    def test_exp_gradient(self):
        x = Value(1.5)
        y = x.exp()
        y.backward()
        self.assertAlmostEqual(y.data, math.exp(1.5))
        self.assertAlmostEqual(x.grad, math.exp(1.5))

    def test_log_gradient(self):
        x = Value(4.0)
        y = x.log()
        y.backward()
        self.assertAlmostEqual(x.grad, 0.25)

    def test_log_nonpositive_raises(self):
        with self.assertRaises(ValueError):
            Value(-1.0).log()

    def test_tanh_gradient(self):
        x = Value(0.5)
        y = x.tanh()
        y.backward()
        expected = 1 - math.tanh(0.5) ** 2
        self.assertAlmostEqual(x.grad, expected)

    def test_relu_gradient_positive_and_negative(self):
        x_pos = Value(3.0)
        y_pos = x_pos.relu()
        y_pos.backward()
        self.assertAlmostEqual(x_pos.grad, 1.0)

        x_neg = Value(-3.0)
        y_neg = x_neg.relu()
        y_neg.backward()
        self.assertAlmostEqual(x_neg.grad, 0.0)


class TestComplexExpressions(unittest.TestCase):
    def test_composite_function_gradient(self):
        # f(x) = (x^2 + 3x) * exp(x), evaluated at x=1
        # f'(x) = (2x+3)*exp(x) + (x^2+3x)*exp(x)
        x = Value(1.0)
        f = (x ** 2 + 3 * x) * x.exp()
        f.backward()
        expected = (2 * 1.0 + 3) * math.exp(1.0) + (1.0 ** 2 + 3 * 1.0) * math.exp(1.0)
        self.assertAlmostEqual(x.grad, expected, places=6)

    def test_multivariate_gradient(self):
        # f(x,y) = x^2*y + y^3, df/dx = 2xy, df/dy = x^2 + 3y^2
        x = Value(2.0)
        y = Value(3.0)
        f = x ** 2 * y + y ** 3
        f.backward()
        self.assertAlmostEqual(x.grad, 2 * 2.0 * 3.0)
        self.assertAlmostEqual(y.grad, 2.0 ** 2 + 3 * 3.0 ** 2)

    def test_matches_forward_mode_dual_numbers(self):
        # The same function, differentiated by two completely
        # independent methods, must agree: this is a strong
        # correctness signal precisely because the two implementations
        # share no code.
        def f_reverse(vals):
            x, y = vals
            return (x * x + (x * y).tanh()) * y.exp()

        def f_forward(vals):
            x, y = vals
            return (x * x + dn.tanh(x * y)) * dn.exp(y)

        x0, y0 = 0.7, -0.4
        rx, ry = Value(x0), Value(y0)
        out = f_reverse([rx, ry])
        out.backward()

        forward_grad = dn.gradient_forward_mode(f_forward, [x0, y0])

        self.assertAlmostEqual(rx.grad, forward_grad[0], places=8)
        self.assertAlmostEqual(ry.grad, forward_grad[1], places=8)

    def test_gradient_matches_finite_difference(self):
        def f(vals):
            x, y, z = vals
            return x.tanh() * y + z.exp() * x

        x0, y0, z0 = 0.3, 1.1, -0.2
        vx, vy, vz = Value(x0), Value(y0), Value(z0)
        out = f([vx, vy, vz])
        out.backward()

        def raw(a, b, c):
            return math.tanh(a) * b + math.exp(c) * a

        h = 1e-6
        expected_x = (raw(x0 + h, y0, z0) - raw(x0 - h, y0, z0)) / (2 * h)
        expected_y = (raw(x0, y0 + h, z0) - raw(x0, y0 - h, z0)) / (2 * h)
        expected_z = (raw(x0, y0, z0 + h) - raw(x0, y0, z0 - h)) / (2 * h)
        self.assertAlmostEqual(vx.grad, expected_x, places=4)
        self.assertAlmostEqual(vy.grad, expected_y, places=4)
        self.assertAlmostEqual(vz.grad, expected_z, places=4)


class TestJacobianReverseMode(unittest.TestCase):
    def test_jacobian_known_values(self):
        f = lambda v: [v[0] + v[1], v[0] * v[1]]
        jac = jacobian_reverse_mode(f, [2.0, 3.0])
        self.assertAlmostEqual(jac[0][0], 1.0)
        self.assertAlmostEqual(jac[0][1], 1.0)
        self.assertAlmostEqual(jac[1][0], 3.0)
        self.assertAlmostEqual(jac[1][1], 2.0)

    def test_jacobian_matches_forward_mode(self):
        f_reverse = lambda v: [v[0] ** 2 + v[1], (v[0] * v[1]).exp()]
        f_forward = lambda v: [v[0] ** 2 + v[1], dn.exp(v[0] * v[1])]
        x = [1.5, -0.5]
        jac_reverse = jacobian_reverse_mode(f_reverse, x)

        import dual_numbers as dn_module
        jac_forward = dn_module.jacobian_forward_mode(f_forward, x)

        for row in range(2):
            for col in range(2):
                self.assertAlmostEqual(jac_reverse[row][col], jac_forward[row][col], places=6)


if __name__ == "__main__":
    unittest.main()
