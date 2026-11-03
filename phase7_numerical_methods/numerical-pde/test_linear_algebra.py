import random
import unittest

import linear_algebra as la


class TestThomasAlgorithm(unittest.TestCase):
    def test_known_small_system(self):
        lower = [0, 1, 1, 1]
        diag = [2, 3, 3, 2]
        upper = [1, 1, 1, 0]
        rhs = [1, 2, 3, 4]
        x = la.thomas_algorithm(lower, diag, upper, rhs)
        reconstructed = la.mat_vec_tridiagonal(lower, diag, upper, x)
        for a, b in zip(reconstructed, rhs):
            self.assertAlmostEqual(a, b, places=8)

    def test_identity_system(self):
        n = 5
        lower = [0.0] * n
        diag = [1.0] * n
        upper = [0.0] * n
        rhs = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = la.thomas_algorithm(lower, diag, upper, rhs)
        self.assertEqual(x, rhs)

    def test_single_equation(self):
        x = la.thomas_algorithm([0.0], [4.0], [0.0], [8.0])
        self.assertAlmostEqual(x[0], 2.0)

    def test_random_diagonally_dominant_system_matches_direct_solve(self):
        rng = random.Random(1)
        n = 15
        lower = [0.0] + [rng.uniform(-2, 2) for _ in range(n - 1)]
        diag = [rng.uniform(8, 12) for _ in range(n)]
        upper = [rng.uniform(-2, 2) for _ in range(n - 1)] + [0.0]
        rhs = [rng.uniform(-5, 5) for _ in range(n)]

        x = la.thomas_algorithm(lower, diag, upper, rhs)
        reconstructed = la.mat_vec_tridiagonal(lower, diag, upper, x)
        for a, b in zip(reconstructed, rhs):
            self.assertAlmostEqual(a, b, places=6)

    def test_zero_pivot_raises(self):
        with self.assertRaises(ValueError):
            la.thomas_algorithm([0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0])


class TestMatVecTridiagonal(unittest.TestCase):
    def test_matches_manual_computation(self):
        lower = [0, 2, 3]
        diag = [1, 1, 1]
        upper = [4, 5, 0]
        x = [1.0, 2.0, 3.0]
        result = la.mat_vec_tridiagonal(lower, diag, upper, x)
        self.assertAlmostEqual(result[0], 1 * 1 + 4 * 2)
        self.assertAlmostEqual(result[1], 2 * 1 + 1 * 2 + 5 * 3)
        self.assertAlmostEqual(result[2], 3 * 2 + 1 * 3)


class TestLinspace(unittest.TestCase):
    def test_basic_range(self):
        result = la.linspace(0.0, 1.0, 5)
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[-1], 1.0)
        self.assertAlmostEqual(result[2], 0.5)

    def test_single_point(self):
        self.assertEqual(la.linspace(3.0, 7.0, 1), [3.0])


class TestMaxAbsDiff(unittest.TestCase):
    def test_matches_manual_computation(self):
        a = [1.0, 2.0, 3.0]
        b = [1.5, 1.0, 3.2]
        self.assertAlmostEqual(la.max_abs_diff(a, b), 1.0)


if __name__ == "__main__":
    unittest.main()
