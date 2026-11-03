import math
from typing import List

Matrix = List[List[float]]
Vector = List[float]


def shape(a: Matrix) -> tuple:
    return len(a), (len(a[0]) if a else 0)


def transpose(a: Matrix) -> Matrix:
    rows, cols = shape(a)
    return [[a[i][j] for i in range(rows)] for j in range(cols)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    n, k1 = shape(a)
    k2, m = shape(b)
    if k1 != k2:
        raise ValueError(f"incompatible shapes for matmul: {shape(a)} and {shape(b)}")
    result = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for k in range(k1):
            r = a[i][k]
            if r == 0.0:
                continue
            for j in range(m):
                result[i][j] += r * b[k][j]
    return result


def mat_vec(a: Matrix, v: Vector) -> Vector:
    return [sum(a[i][j] * v[j] for j in range(len(v))) for i in range(len(a))]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def vector_norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def vec_sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


def vec_scale(v: Vector, s: float) -> Vector:
    return [x * s for x in v]


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def solve_linear_system(a: Matrix, b: Vector) -> Vector:
    """Solve A*x = b by Gaussian elimination with partial pivoting."""
    n = len(a)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-14:
            raise ValueError("matrix is singular or nearly singular")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot
        for row in range(n):
            if row != col and aug[row][col] != 0.0:
                factor = aug[row][col]
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def invert(a: Matrix) -> Matrix:
    n = len(a)
    columns = []
    for i in range(n):
        e_i = [1.0 if k == i else 0.0 for k in range(n)]
        columns.append(solve_linear_system([row[:] for row in a], e_i))
    return [[columns[j][i] for j in range(n)] for i in range(n)]
