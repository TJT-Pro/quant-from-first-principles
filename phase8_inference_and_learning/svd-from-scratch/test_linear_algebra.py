import math
import unittest

import linear_algebra as la


class TestBasicOps(unittest.TestCase):
    def test_transpose(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self.assertEqual(la.transpose(a), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_matmul_known_result(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        self.assertEqual(la.matmul(a, b), [[19.0, 22.0], [43.0, 50.0]])

    def test_matmul_incompatible_shapes_raises(self):
        with self.assertRaises(ValueError):
            la.matmul([[1.0, 2.0]], [[1.0, 2.0]])

    def test_mat_vec(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        v = [1.0, 1.0]
        self.assertEqual(la.mat_vec(a, v), [3.0, 7.0])

    def test_identity(self):
        self.assertEqual(la.identity(3), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_frobenius_norm(self):
        a = [[3.0, 0.0], [0.0, 4.0]]
        self.assertAlmostEqual(la.frobenius_norm(a), 5.0)

    def test_vector_norm(self):
        self.assertAlmostEqual(la.vector_norm([3.0, 4.0]), 5.0)

    def test_normalize(self):
        v = la.normalize([3.0, 4.0])
        self.assertAlmostEqual(la.vector_norm(v), 1.0)

    def test_normalize_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            la.normalize([0.0, 0.0])

    def test_dot(self):
        self.assertAlmostEqual(la.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0)

    def test_is_symmetric_true(self):
        self.assertTrue(la.is_symmetric([[1.0, 2.0], [2.0, 3.0]]))

    def test_is_symmetric_false(self):
        self.assertFalse(la.is_symmetric([[1.0, 2.0], [3.0, 4.0]]))

    def test_is_symmetric_non_square(self):
        self.assertFalse(la.is_symmetric([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_add_and_subtract_are_inverses(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[0.5, 1.5], [2.5, 3.5]]
        self.assertEqual(la.subtract(la.add(a, b), b), a)

    def test_scalar_mul(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(la.scalar_mul(a, 2.0), [[2.0, 4.0], [6.0, 8.0]])


if __name__ == "__main__":
    unittest.main()
