"""Tests for the matrix engine.

All expected values are computed analytically.  No numpy or scipy.
"""

import math
import unittest

import matrix_engine as mx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allclose(A, B, tol=1e-8):
    """True if A and B (matrices or vectors) are element-wise close."""
    if isinstance(A[0], list):
        return all(allclose(A[i], B[i], tol) for i in range(len(A)))
    return all(abs(a - b) <= tol for a, b in zip(A, B))


# ---------------------------------------------------------------------------
# Constructors and shape
# ---------------------------------------------------------------------------

class TestConstructors(unittest.TestCase):
    def test_zeros(self):
        Z = mx.zeros(2, 3)
        self.assertEqual(mx.shape(Z), (2, 3))
        for row in Z:
            self.assertTrue(all(v == 0.0 for v in row))

    def test_identity(self):
        I = mx.identity(3)
        self.assertEqual(mx.shape(I), (3, 3))
        for i in range(3):
            for j in range(3):
                self.assertEqual(I[i][j], 1.0 if i == j else 0.0)

    def test_copy_matrix(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = mx.copy_matrix(A)
        B[0][0] = 99.0
        self.assertEqual(A[0][0], 1.0)  # original unchanged

    def test_shape(self):
        A = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(mx.shape(A), (2, 3))


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic(unittest.TestCase):
    def setUp(self):
        self.A = [[1.0, 2.0], [3.0, 4.0]]
        self.B = [[5.0, 6.0], [7.0, 8.0]]

    def test_add(self):
        C = mx.add(self.A, self.B)
        self.assertEqual(C, [[6.0, 8.0], [10.0, 12.0]])

    def test_subtract(self):
        C = mx.subtract(self.B, self.A)
        self.assertEqual(C, [[4.0, 4.0], [4.0, 4.0]])

    def test_scalar_multiply(self):
        C = mx.scalar_multiply(2.0, self.A)
        self.assertEqual(C, [[2.0, 4.0], [6.0, 8.0]])

    def test_transpose_square(self):
        T = mx.transpose(self.A)
        self.assertEqual(T, [[1.0, 3.0], [2.0, 4.0]])

    def test_transpose_rectangular(self):
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        T = mx.transpose(A)
        self.assertEqual(mx.shape(T), (3, 2))
        self.assertEqual(T[0], [1.0, 4.0])

    def test_add_shape_mismatch(self):
        with self.assertRaises(ValueError):
            mx.add([[1, 2]], [[1, 2, 3]])


# ---------------------------------------------------------------------------
# Matrix multiplication
# ---------------------------------------------------------------------------

class TestMatmul(unittest.TestCase):
    def test_identity_left(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        I = mx.identity(2)
        self.assertEqual(mx.matmul(I, A), A)

    def test_identity_right(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        I = mx.identity(2)
        self.assertEqual(mx.matmul(A, I), A)

    def test_2x2(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        C = mx.matmul(A, B)
        self.assertEqual(C, [[19.0, 22.0], [43.0, 50.0]])

    def test_rectangular(self):
        A = [[1.0, 2.0, 3.0]]   # 1×3
        B = [[1.0], [2.0], [3.0]]  # 3×1
        C = mx.matmul(A, B)
        self.assertEqual(mx.shape(C), (1, 1))
        self.assertAlmostEqual(C[0][0], 14.0)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            mx.matmul([[1, 2]], [[1, 2]])

    def test_matvec(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        x = [1.0, 1.0]
        self.assertEqual(mx.matvec(A, x), [3.0, 7.0])

    def test_matvec_bad_length(self):
        with self.assertRaises(ValueError):
            mx.matvec([[1.0, 2.0]], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

class TestVector(unittest.TestCase):
    def test_dot(self):
        self.assertAlmostEqual(mx.dot([1, 2, 3], [4, 5, 6]), 32.0)

    def test_dot_length_mismatch(self):
        with self.assertRaises(ValueError):
            mx.dot([1, 2], [1, 2, 3])

    def test_norm_l2(self):
        self.assertAlmostEqual(mx.norm([3.0, 4.0]), 5.0)

    def test_norm_l1(self):
        self.assertAlmostEqual(mx.norm([-2.0, 3.0], p=1), 5.0)

    def test_norm_linf(self):
        self.assertAlmostEqual(mx.norm([1.0, -5.0, 3.0], p='inf'), 5.0)

    def test_norm_bad_p(self):
        with self.assertRaises(ValueError):
            mx.norm([1.0, 2.0], p=3)

    def test_normalize(self):
        v = mx.normalize([3.0, 4.0])
        self.assertAlmostEqual(mx.norm(v), 1.0)

    def test_normalize_zero_raises(self):
        with self.assertRaises(ValueError):
            mx.normalize([0.0, 0.0])

    def test_vec_add(self):
        self.assertEqual(mx.vec_add([1, 2], [3, 4]), [4, 6])

    def test_vec_sub(self):
        self.assertEqual(mx.vec_sub([5, 3], [2, 1]), [3, 2])

    def test_vec_scale(self):
        self.assertEqual(mx.vec_scale(3.0, [1.0, 2.0]), [3.0, 6.0])


# ---------------------------------------------------------------------------
# Trace and Frobenius norm
# ---------------------------------------------------------------------------

class TestScalarProperties(unittest.TestCase):
    def test_trace(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        self.assertAlmostEqual(mx.trace(A), 5.0)

    def test_trace_3x3(self):
        A = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        self.assertAlmostEqual(mx.trace(A), 6.0)

    def test_trace_non_square_raises(self):
        with self.assertRaises(ValueError):
            mx.trace([[1, 2, 3], [4, 5, 6]])

    def test_frobenius(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        self.assertAlmostEqual(mx.frobenius_norm(A), math.sqrt(30.0))


# ---------------------------------------------------------------------------
# Determinant
# ---------------------------------------------------------------------------

class TestDeterminant(unittest.TestCase):
    def test_1x1(self):
        self.assertAlmostEqual(mx.determinant([[7.0]]), 7.0)

    def test_2x2(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        self.assertAlmostEqual(mx.determinant(A), -2.0)

    def test_3x3(self):
        A = [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 10.0]]
        self.assertAlmostEqual(mx.determinant(A), -3.0)

    def test_identity(self):
        self.assertAlmostEqual(mx.determinant(mx.identity(4)), 1.0)

    def test_singular(self):
        A = [[1.0, 2.0], [2.0, 4.0]]
        self.assertAlmostEqual(mx.determinant(A), 0.0)

    def test_det_lu_matches(self):
        A = [[2.0, 1.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0]]
        self.assertAlmostEqual(mx.determinant(A), mx.det_lu(A), places=8)

    def test_non_square_raises(self):
        with self.assertRaises(ValueError):
            mx.determinant([[1, 2, 3], [4, 5, 6]])


# ---------------------------------------------------------------------------
# LU decomposition and linear solve
# ---------------------------------------------------------------------------

class TestLU(unittest.TestCase):
    def test_pa_equals_lu(self):
        A = [[2.0, 1.0, -1.0],
             [-3.0, -1.0, 2.0],
             [-2.0, 1.0, 2.0]]
        L, U, P, _ = mx.lu_decompose(A)
        PA  = mx.matmul(P, A)
        LU  = mx.matmul(L, U)
        self.assertTrue(allclose(PA, LU, tol=1e-10))

    def test_l_is_unit_lower(self):
        A = [[4.0, 3.0], [6.0, 3.0]]
        L, U, P, _ = mx.lu_decompose(A)
        n = len(L)
        for i in range(n):
            self.assertAlmostEqual(L[i][i], 1.0)
            for j in range(i + 1, n):
                self.assertAlmostEqual(L[i][j], 0.0)

    def test_u_is_upper(self):
        A = [[4.0, 3.0], [6.0, 3.0]]
        L, U, P, _ = mx.lu_decompose(A)
        n = len(U)
        for i in range(n):
            for j in range(i):
                self.assertAlmostEqual(U[i][j], 0.0)

    def test_solve_2x2(self):
        A = [[2.0, 1.0], [5.0, 7.0]]
        b = [11.0, 13.0]
        x = mx.solve(A, b)
        self.assertAlmostEqual(x[0], 7.111111111, places=6)
        self.assertAlmostEqual(x[1], -3.222222222, places=6)

    def test_solve_residual(self):
        A = [[3.0, 1.0, -1.0],
             [2.0, 4.0, 1.0],
             [-1.0, 2.0, 5.0]]
        b = [4.0, 1.0, 1.0]
        x = mx.solve(A, b)
        Ax = mx.matvec(A, x)
        self.assertTrue(allclose(Ax, b, tol=1e-8))

    def test_solve_identity(self):
        I = mx.identity(3)
        b = [1.0, 2.0, 3.0]
        x = mx.solve(I, b)
        self.assertTrue(allclose(x, b))

    def test_solve_singular_raises(self):
        A = [[1.0, 2.0], [2.0, 4.0]]
        with self.assertRaises(ValueError):
            mx.solve(A, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Matrix inverse
# ---------------------------------------------------------------------------

class TestInverse(unittest.TestCase):
    def test_inverse_2x2(self):
        A   = [[4.0, 7.0], [2.0, 6.0]]
        Ai  = mx.inverse(A)
        I   = mx.matmul(A, Ai)
        self.assertTrue(allclose(I, mx.identity(2), tol=1e-10))

    def test_inverse_times_original_is_identity(self):
        A  = [[1.0, 2.0, 3.0],
              [0.0, 4.0, 5.0],
              [1.0, 0.0, 6.0]]
        Ai = mx.inverse(A)
        I  = mx.matmul(A, Ai)
        self.assertTrue(allclose(I, mx.identity(3), tol=1e-10))

    def test_inverse_identity_is_identity(self):
        I  = mx.identity(3)
        Ii = mx.inverse(I)
        self.assertTrue(allclose(Ii, I, tol=1e-12))

    def test_singular_raises(self):
        A = [[1.0, 2.0], [2.0, 4.0]]
        with self.assertRaises(ValueError):
            mx.inverse(A)


# ---------------------------------------------------------------------------
# Eigenvalues
# ---------------------------------------------------------------------------

class TestEigenvalues(unittest.TestCase):
    def test_diagonal_2x2(self):
        A  = [[3.0, 0.0], [0.0, 5.0]]
        ev = mx.eigenvalues_2x2(A)
        self.assertIn(3.0, [round(e, 8) for e in ev])
        self.assertIn(5.0, [round(e, 8) for e in ev])

    def test_symmetric_2x2(self):
        A  = [[2.0, 1.0], [1.0, 2.0]]
        ev = mx.eigenvalues_2x2(A)
        self.assertTrue(allclose(sorted(ev), [1.0, 3.0], tol=1e-10))

    def test_trace_equals_sum_of_eigenvalues(self):
        A  = [[1.0, 2.0], [3.0, 4.0]]
        ev = mx.eigenvalues_2x2(A)
        self.assertAlmostEqual(sum(ev), mx.trace(A))

    def test_det_equals_product_of_eigenvalues(self):
        A  = [[1.0, 2.0], [3.0, 4.0]]
        ev = mx.eigenvalues_2x2(A)
        self.assertAlmostEqual(ev[0] * ev[1], mx.det_lu(A), places=8)

    def test_complex_eigenvalues(self):
        A  = [[0.0, -1.0], [1.0, 0.0]]
        ev = mx.eigenvalues_2x2(A)
        # eigenvalues are ±i, returned as (0, 1) and (0, -1)
        self.assertTrue(isinstance(ev[0], tuple))

    def test_wrong_size_raises(self):
        with self.assertRaises(ValueError):
            mx.eigenvalues_2x2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_power_iteration_dominant(self):
        # [[2,1],[1,2]]: dominant eigenvalue = 3
        A   = [[2.0, 1.0], [1.0, 2.0]]
        lam, v = mx.power_iteration(A)
        self.assertAlmostEqual(lam, 3.0, places=6)
        # Eigenvector should satisfy Av = λv
        Av  = mx.matvec(A, v)
        lv  = mx.vec_scale(lam, v)
        self.assertTrue(allclose(Av, lv, tol=1e-6))

    def test_power_iteration_eigenvector_unit(self):
        A = [[4.0, 0.0], [0.0, 1.0]]
        _, v = mx.power_iteration(A)
        self.assertAlmostEqual(mx.norm(v), 1.0, places=10)


# ---------------------------------------------------------------------------
# QR decomposition
# ---------------------------------------------------------------------------

class TestQR(unittest.TestCase):
    def test_a_equals_qr(self):
        A = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        Q, R = mx.qr_decompose(A)
        self.assertTrue(allclose(mx.matmul(Q, R), A, tol=1e-10))

    def test_q_orthonormal_columns(self):
        A = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        Q, R = mx.qr_decompose(A)
        Qt  = mx.transpose(Q)
        QtQ = mx.matmul(Qt, Q)
        self.assertTrue(allclose(QtQ, mx.identity(2), tol=1e-10))

    def test_r_upper_triangular(self):
        A = [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 10.0]]
        Q, R = mx.qr_decompose(A)
        n = len(R)
        for i in range(n):
            for j in range(i):
                self.assertAlmostEqual(R[i][j], 0.0, places=10)

    def test_square_matrix(self):
        A = [[2.0, -1.0], [1.0, 3.0]]
        Q, R = mx.qr_decompose(A)
        self.assertTrue(allclose(mx.matmul(Q, R), A, tol=1e-10))

    def test_rank_deficient_raises(self):
        A = [[1.0, 2.0], [2.0, 4.0]]
        with self.assertRaises(ValueError):
            mx.qr_decompose(A)


# ---------------------------------------------------------------------------
# Least squares
# ---------------------------------------------------------------------------

class TestLeastSquares(unittest.TestCase):
    def test_exact_solution(self):
        # Square full-rank system: least squares = exact solve
        A = [[2.0, 1.0], [1.0, 3.0]]
        b = [5.0, 10.0]
        x = mx.least_squares(A, b)
        Ax = mx.matvec(A, x)
        self.assertTrue(allclose(Ax, b, tol=1e-8))

    def test_overdetermined(self):
        # Fit y = ax + b to three points: (0,1), (1,2), (2,4)
        A = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
        b = [1.0, 2.0, 4.0]
        x = mx.least_squares(A, b)
        # Check normal equations: AᵀA x = Aᵀb
        AtA = mx.matmul(mx.transpose(A), A)
        Atb = mx.matvec(mx.transpose(A), b)
        lhs = mx.matvec(AtA, x)
        self.assertTrue(allclose(lhs, Atb, tol=1e-8))

    def test_residual_norm_positive(self):
        A = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
        b = [1.0, 2.0, 4.0]
        x = mx.least_squares(A, b)
        r = mx.residual_norm(A, b, x)
        self.assertGreater(r, 0.0)  # overdetermined, not zero

    def test_residual_smaller_than_any_other(self):
        # x_hat minimises residual; perturbing it increases the norm
        A = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
        b = [1.0, 2.0, 4.0]
        x = mx.least_squares(A, b)
        r0 = mx.residual_norm(A, b, x)
        r1 = mx.residual_norm(A, b, [x[0] + 0.1, x[1]])
        self.assertLess(r0, r1)


# ---------------------------------------------------------------------------
# Condition number
# ---------------------------------------------------------------------------

class TestConditionNumber(unittest.TestCase):
    def test_identity_condition_1(self):
        I   = mx.identity(3)
        kap = mx.condition_number(I)
        self.assertAlmostEqual(kap, 1.0, places=4)

    def test_diagonal_matrix(self):
        # [[2,0],[0,1]] has singular values 2 and 1 → κ = 2
        A   = [[2.0, 0.0], [0.0, 1.0]]
        kap = mx.condition_number(A)
        self.assertAlmostEqual(kap, 2.0, places=2)

    def test_well_conditioned_small(self):
        kap = mx.condition_number([[1.0, 0.0], [0.0, 1.0]])
        self.assertLess(kap, 2.0)

    def test_ill_conditioned_larger_than_well(self):
        well = mx.condition_number([[1.0, 0.0], [0.0, 1.0]])
        ill  = mx.condition_number([[100.0, 0.0], [0.0, 1.0]])
        self.assertGreater(ill, well)


if __name__ == "__main__":
    unittest.main()
