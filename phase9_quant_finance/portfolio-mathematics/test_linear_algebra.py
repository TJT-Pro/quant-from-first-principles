import unittest

import linear_algebra as la


class TestBasicOps(unittest.TestCase):
    def test_matmul_known_result(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        self.assertEqual(la.matmul(a, b), [[19.0, 22.0], [43.0, 50.0]])

    def test_transpose(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self.assertEqual(la.transpose(a), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_dot(self):
        self.assertAlmostEqual(la.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0)

    def test_vector_norm(self):
        self.assertAlmostEqual(la.vector_norm([3.0, 4.0]), 5.0)


class TestSolveLinearSystem(unittest.TestCase):
    def test_simple_system(self):
        a = [[2.0, 1.0], [1.0, 3.0]]
        b = [5.0, 10.0]
        x = la.solve_linear_system(a, b)
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(x[1], 3.0)

    def test_singular_raises(self):
        with self.assertRaises(ValueError):
            la.solve_linear_system([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


class TestInvert(unittest.TestCase):
    def test_inverse_of_identity_is_identity(self):
        result = la.invert(la.identity(3))
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(result[i][j], expected)

    def test_a_times_a_inverse_is_identity(self):
        a = [[4.0, 7.0], [2.0, 6.0]]
        a_inv = la.invert(a)
        product = la.matmul(a, a_inv)
        for i in range(2):
            for j in range(2):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(product[i][j], expected, places=8)


if __name__ == "__main__":
    unittest.main()
