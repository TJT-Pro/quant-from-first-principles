import math
import random
import unittest

import convex_functions as cf


class TestNumericalGradient(unittest.TestCase):
    def test_gradient_of_quadratic(self):
        f = lambda x: x[0] ** 2 + 3 * x[1] ** 2
        grad = cf.numerical_gradient(f, [2.0, 1.0])
        self.assertAlmostEqual(grad[0], 4.0, places=4)
        self.assertAlmostEqual(grad[1], 6.0, places=4)

    def test_gradient_of_linear_function_is_constant(self):
        f = lambda x: 3 * x[0] + 2 * x[1] - 5
        grad = cf.numerical_gradient(f, [10.0, -3.0])
        self.assertAlmostEqual(grad[0], 3.0, places=4)
        self.assertAlmostEqual(grad[1], 2.0, places=4)


class TestNumericalHessian(unittest.TestCase):
    def test_hessian_of_quadratic_form(self):
        f = lambda x: x[0] ** 2 + 2 * x[0] * x[1] + 3 * x[1] ** 2
        hess = cf.numerical_hessian(f, [1.0, 1.0])
        self.assertAlmostEqual(hess[0][0], 2.0, places=2)
        self.assertAlmostEqual(hess[0][1], 2.0, places=2)
        self.assertAlmostEqual(hess[1][0], 2.0, places=2)
        self.assertAlmostEqual(hess[1][1], 6.0, places=2)

    def test_hessian_is_symmetric(self):
        f = lambda x: x[0] ** 3 * x[1] + x[1] ** 2
        hess = cf.numerical_hessian(f, [1.5, -0.5])
        self.assertAlmostEqual(hess[0][1], hess[1][0], places=6)

    def test_hessian_of_linear_function_is_zero(self):
        f = lambda x: 5 * x[0] - 2 * x[1] + 7
        hess = cf.numerical_hessian(f, [3.0, 4.0])
        for row in hess:
            for v in row:
                self.assertAlmostEqual(v, 0.0, places=4)


class TestConvexityByDefinition(unittest.TestCase):
    def test_quadratic_is_convex_on_samples(self):
        f = lambda x: x[0] ** 2 + x[1] ** 2
        rng = random.Random(1)
        sampler = lambda: [rng.uniform(-5, 5), rng.uniform(-5, 5)]
        result = cf.check_convexity_by_definition(f, sampler)
        self.assertTrue(result["convex_on_samples"])

    def test_exponential_is_convex_on_samples(self):
        f = lambda x: math.exp(x[0])
        rng = random.Random(2)
        sampler = lambda: [rng.uniform(-3, 3)]
        result = cf.check_convexity_by_definition(f, sampler)
        self.assertTrue(result["convex_on_samples"])

    def test_negative_quadratic_is_not_convex(self):
        f = lambda x: -(x[0] ** 2) - x[1] ** 2
        rng = random.Random(3)
        sampler = lambda: [rng.uniform(-5, 5), rng.uniform(-5, 5)]
        result = cf.check_convexity_by_definition(f, sampler)
        self.assertFalse(result["convex_on_samples"])
        self.assertGreater(result["worst_violation"], 0.0)

    def test_cubic_is_not_convex_globally(self):
        f = lambda x: x[0] ** 3
        rng = random.Random(4)
        sampler = lambda: [rng.uniform(-5, 5)]
        result = cf.check_convexity_by_definition(f, sampler)
        self.assertFalse(result["convex_on_samples"])

    def test_affine_function_is_convex(self):
        f = lambda x: 2 * x[0] - 3 * x[1] + 1
        rng = random.Random(5)
        sampler = lambda: [rng.uniform(-10, 10), rng.uniform(-10, 10)]
        result = cf.check_convexity_by_definition(f, sampler)
        self.assertTrue(result["convex_on_samples"])
        self.assertAlmostEqual(result["worst_violation"], 0.0, places=6)


class TestPositiveSemidefiniteCheck(unittest.TestCase):
    def test_identity_is_psd(self):
        self.assertTrue(cf.is_positive_semidefinite_2x2_leading_minors([[1.0, 0.0], [0.0, 1.0]]))

    def test_negative_definite_is_not_psd(self):
        self.assertFalse(cf.is_positive_semidefinite_2x2_leading_minors([[-1.0, 0.0], [0.0, -1.0]]))

    def test_indefinite_matrix_is_not_psd(self):
        self.assertFalse(cf.is_positive_semidefinite_2x2_leading_minors([[0.0, 1.0], [1.0, 0.0]]))

    def test_known_positive_definite_matrix(self):
        self.assertTrue(cf.is_positive_semidefinite_2x2_leading_minors([[2.0, 1.0], [1.0, 2.0]]))


class TestConvexityViaHessian(unittest.TestCase):
    def test_quadratic_is_convex_everywhere(self):
        f = lambda x: x[0] ** 2 + x[1] ** 2
        points = [[0.0, 0.0], [1.0, 1.0], [-2.0, 3.0], [5.0, -5.0]]
        result = cf.convexity_via_hessian(f, points)
        self.assertTrue(result["convex_by_hessian_test"])

    def test_concave_function_fails_hessian_test(self):
        f = lambda x: -(x[0] ** 2) - x[1] ** 2
        points = [[0.0, 0.0], [1.0, 1.0]]
        result = cf.convexity_via_hessian(f, points)
        self.assertFalse(result["convex_by_hessian_test"])

    def test_saddle_function_fails_hessian_test(self):
        f = lambda x: x[0] ** 2 - x[1] ** 2
        points = [[0.0, 0.0], [1.0, 1.0]]
        result = cf.convexity_via_hessian(f, points)
        self.assertFalse(result["convex_by_hessian_test"])

    def test_negative_log_is_convex_on_positive_domain(self):
        f = lambda x: -math.log(x[0])
        points = [[0.5], [1.0], [2.0], [10.0]]
        result = cf.convexity_via_hessian(f, points)
        self.assertTrue(result["convex_by_hessian_test"])


if __name__ == "__main__":
    unittest.main()
