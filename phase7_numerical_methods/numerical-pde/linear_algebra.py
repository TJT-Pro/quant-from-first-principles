import math
from typing import List

Vector = List[float]


def thomas_algorithm(lower: Vector, diag: Vector, upper: Vector, rhs: Vector) -> Vector:
    """Solve a tridiagonal linear system A*x = rhs, where A has
    sub-diagonal `lower` (lower[0] unused, corresponds to A[i][i-1]
    for i=1..n-1), diagonal `diag` (A[i][i]), and super-diagonal
    `upper` (upper[n-1] unused, A[i][i+1] for i=0..n-2), via the
    Thomas algorithm: a specialised form of Gaussian elimination that
    exploits the tridiagonal structure to run in O(n) time and O(n)
    memory, rather than the O(n^3) time and O(n^2) memory a generic
    dense solver would need. This is the standard tool for implicit
    and Crank-Nicolson finite-difference schemes, both of which
    reduce, at every single time step, to solving exactly one
    tridiagonal system (since a standard second-derivative stencil
    only ever couples each grid point to its two immediate neighbours).

    The algorithm proceeds in two sweeps: a forward sweep eliminates
    the sub-diagonal by expressing each row as a multiple of the row
    above it (exactly as in ordinary Gaussian elimination, but touching
    only the single nonzero sub-diagonal entry per row instead of an
    entire row), followed by a backward substitution recovering x from
    the resulting upper-bidiagonal system.
    """
    n = len(diag)
    c_prime = [0.0] * n
    d_prime = [0.0] * n

    if abs(diag[0]) < 1e-300:
        raise ValueError("thomas_algorithm: zero pivot encountered, system may be singular")
    c_prime[0] = upper[0] / diag[0] if n > 1 else 0.0
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        if abs(denom) < 1e-300:
            raise ValueError("thomas_algorithm: zero pivot encountered, system may be singular")
        c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom

    x = [0.0] * n
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def mat_vec_tridiagonal(lower: Vector, diag: Vector, upper: Vector, x: Vector) -> Vector:
    """Multiply a tridiagonal matrix (in the same (lower, diag, upper)
    representation as thomas_algorithm) by a vector, without ever
    forming the full dense matrix; used to independently verify
    thomas_algorithm's solutions by checking A*x actually reproduces
    the right-hand side it was solved against.
    """
    n = len(diag)
    result = [0.0] * n
    for i in range(n):
        total = diag[i] * x[i]
        if i > 0:
            total += lower[i] * x[i - 1]
        if i < n - 1:
            total += upper[i] * x[i + 1]
        result[i] = total
    return result


def linspace(start: float, stop: float, n: int) -> Vector:
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def max_abs_diff(a: Vector, b: Vector) -> float:
    return max(abs(x - y) for x, y in zip(a, b))
