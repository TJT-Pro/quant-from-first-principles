import math
from typing import List

Matrix = List[List[float]]
Vector = List[float]


def shape(a: Matrix) -> tuple:
    return len(a), len(a[0]) if a else 0


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
        row = a[i]
        for k in range(k1):
            r = row[k]
            if r == 0.0:
                continue
            brow = b[k]
            for j in range(m):
                result[i][j] += r * brow[j]
    return result


def mat_vec(a: Matrix, v: Vector) -> Vector:
    return [sum(a[i][j] * v[j] for j in range(len(v))) for i in range(len(a))]


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def scalar_mul(a: Matrix, s: float) -> Matrix:
    return [[v * s for v in row] for row in a]


def subtract(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def frobenius_norm(a: Matrix) -> float:
    return math.sqrt(sum(v * v for row in a for v in row))


def vector_norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def normalize(v: Vector) -> Vector:
    n = vector_norm(v)
    if n < 1e-300:
        raise ValueError("cannot normalise a zero (or numerically zero) vector")
    return [x / n for x in v]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def column(a: Matrix, j: int) -> Vector:
    return [row[j] for row in a]


def is_symmetric(a: Matrix, tol: float = 1e-9) -> bool:
    n, m = shape(a)
    if n != m:
        return False
    return all(abs(a[i][j] - a[j][i]) < tol for i in range(n) for j in range(n))
