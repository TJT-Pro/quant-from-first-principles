import math


# ---------------------------------------------------------------------------
# Type alias and validation
# ---------------------------------------------------------------------------

def _check_matrix(A, name="A"):
    if not A or not A[0]:
        raise ValueError("{} must be non-empty".format(name))
    ncols = len(A[0])
    for row in A:
        if len(row) != ncols:
            raise ValueError("{} rows must all have the same length".format(name))


def shape(A):
    """Return (nrows, ncols) of matrix A."""
    _check_matrix(A)
    return len(A), len(A[0])


def _check_square(A, name="A"):
    r, c = shape(A)
    if r != c:
        raise ValueError("{} must be square (got {}×{})".format(name, r, c))


def _check_same_shape(A, B):
    if shape(A) != shape(B):
        raise ValueError("matrices must have the same shape")


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def zeros(nrows, ncols):
    """Return an nrows×ncols matrix of zeros."""
    return [[0.0] * ncols for _ in range(nrows)]


def identity(n):
    """Return the n×n identity matrix."""
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


def copy_matrix(A):
    """Return a deep copy of matrix A."""
    return [list(row) for row in A]


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------

def add(A, B):
    """Element-wise sum A + B."""
    _check_same_shape(A, B)
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def subtract(A, B):
    """Element-wise difference A - B."""
    _check_same_shape(A, B)
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def scalar_multiply(alpha, A):
    """Multiply every element of A by scalar alpha."""
    return [[alpha * A[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def transpose(A):
    """Return the transpose of A."""
    _check_matrix(A)
    m, n = shape(A)
    return [[A[i][j] for i in range(m)] for j in range(n)]


def matmul(A, B):
    """Matrix product A @ B.

    A is m×k, B is k×n, result is m×n.
    Uses the schoolbook O(m·k·n) algorithm.
    """
    _check_matrix(A, "A")
    _check_matrix(B, "B")
    m, k  = shape(A)
    kb, n = shape(B)
    if k != kb:
        raise ValueError(
            "incompatible shapes for matmul: {}×{} and {}×{}".format(m, k, kb, n)
        )
    C = zeros(m, n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C


def matvec(A, x):
    """Matrix-vector product A @ x.

    A is m×n, x is a list of n floats, result is a list of m floats.
    """
    _check_matrix(A)
    m, n = shape(A)
    if len(x) != n:
        raise ValueError(
            "matvec: A is {}×{} but x has length {}".format(m, n, len(x))
        )
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(m)]


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

def dot(u, v):
    """Dot product of two vectors."""
    if len(u) != len(v):
        raise ValueError("dot: vectors must have the same length")
    return sum(ui * vi for ui, vi in zip(u, v))


def vec_add(u, v):
    """Element-wise sum of two vectors."""
    if len(u) != len(v):
        raise ValueError("vec_add: vectors must have the same length")
    return [ui + vi for ui, vi in zip(u, v)]


def vec_sub(u, v):
    """Element-wise difference u - v."""
    if len(u) != len(v):
        raise ValueError("vec_sub: vectors must have the same length")
    return [ui - vi for ui, vi in zip(u, v)]


def vec_scale(alpha, v):
    """Multiply vector v by scalar alpha."""
    return [alpha * vi for vi in v]


def norm(v, p=2):
    """Vector norm.  p=1 (sum of abs), p=2 (Euclidean), p='inf' (max abs).

    Parameters
    ----------
    v : list of floats
    p : 1, 2, or 'inf'
    """
    if p == 1:
        return sum(abs(vi) for vi in v)
    if p == 2:
        return math.sqrt(sum(vi * vi for vi in v))
    if p == 'inf':
        return max(abs(vi) for vi in v)
    raise ValueError("norm: p must be 1, 2, or 'inf'")


def normalize(v):
    """Return the unit vector in the direction of v."""
    n = norm(v)
    if n == 0.0:
        raise ValueError("cannot normalize the zero vector")
    return [vi / n for vi in v]


# ---------------------------------------------------------------------------
# Scalar properties
# ---------------------------------------------------------------------------

def trace(A):
    """Sum of diagonal elements of A."""
    _check_square(A)
    return sum(A[i][i] for i in range(len(A)))


def frobenius_norm(A):
    """Frobenius norm: sqrt(sum of squares of all elements)."""
    return math.sqrt(sum(A[i][j]**2 for i in range(len(A))
                         for j in range(len(A[0]))))


# ---------------------------------------------------------------------------
# Determinant
# ---------------------------------------------------------------------------

def determinant(A):
    """Determinant of a square matrix by cofactor expansion along row 0.

    Exact for integer matrices.  O(n!) — only practical for n ≤ 6 or so.
    For larger matrices use the LU-based version below.

    Parameters
    ----------
    A : square matrix (list of lists)
    """
    _check_square(A)
    n = len(A)
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    det = 0.0
    for j in range(n):
        minor = [[A[r][c] for c in range(n) if c != j]
                 for r in range(1, n)]
        sign  = 1.0 if j % 2 == 0 else -1.0
        det  += sign * A[0][j] * determinant(minor)
    return det


# ---------------------------------------------------------------------------
# LU decomposition with partial pivoting
# ---------------------------------------------------------------------------

def lu_decompose(A):
    """LU decomposition with partial pivoting: PA = LU.

    Returns (L, U, P, n_swaps) where:
        L        is unit lower triangular
        U        is upper triangular
        P        is the permutation matrix such that P @ A = L @ U
        n_swaps  is the number of row swaps (determines sign of det)

    Uses Doolittle's algorithm with partial pivoting for numerical
    stability.

    Parameters
    ----------
    A : n×n square matrix
    """
    _check_square(A)
    n       = len(A)
    L       = identity(n)
    U       = copy_matrix(A)
    P       = identity(n)
    n_swaps = 0

    for col in range(n):
        # Find pivot: row with largest absolute value in this column
        max_val = abs(U[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(U[row][col]) > max_val:
                max_val = abs(U[row][col])
                max_row = row

        if max_val < 1e-14:
            raise ValueError("matrix is singular or nearly singular")

        # Swap rows in U, P, and the already-filled part of L
        if max_row != col:
            U[col], U[max_row] = U[max_row], U[col]
            P[col], P[max_row] = P[max_row], P[col]
            for k in range(col):
                L[col][k], L[max_row][k] = L[max_row][k], L[col][k]
            n_swaps += 1

        # Eliminate below pivot
        for row in range(col + 1, n):
            if abs(U[col][col]) < 1e-14:
                raise ValueError("matrix is singular or nearly singular")
            factor       = U[row][col] / U[col][col]
            L[row][col]  = factor
            for c in range(col, n):
                U[row][c] -= factor * U[col][c]

    return L, U, P, n_swaps


def forward_substitution(L, b):
    """Solve Lx = b where L is unit lower triangular.

    Parameters
    ----------
    L : n×n unit lower triangular matrix
    b : list of n floats
    """
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        x[i] = b[i] - sum(L[i][j] * x[j] for j in range(i))
    return x


def back_substitution(U, b):
    """Solve Ux = b where U is upper triangular.

    Parameters
    ----------
    U : n×n upper triangular matrix
    b : list of n floats
    """
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < 1e-14:
            raise ValueError("matrix is singular: zero diagonal in U")
        x[i] = (b[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def solve(A, b):
    """Solve the linear system Ax = b.

    Uses LU decomposition with partial pivoting.
    Raises ValueError if A is singular.

    Parameters
    ----------
    A : n×n matrix
    b : list of n floats (right-hand side)
    """
    _check_square(A)
    if len(b) != len(A):
        raise ValueError("b must have the same length as A")
    L, U, P, _ = lu_decompose(A)
    pb       = matvec(P, b)
    y        = forward_substitution(L, pb)
    return back_substitution(U, y)


def det_lu(A):
    """Determinant via LU decomposition.  O(n³), numerically stable.

    The determinant equals the product of the diagonal elements of U
    times (-1)^(number of row swaps in partial pivoting).
    """
    _check_square(A)
    try:
        L, U, P, n_swaps = lu_decompose(A)
    except ValueError:
        return 0.0
    sign = 1.0 if n_swaps % 2 == 0 else -1.0
    prod = 1.0
    for i in range(len(U)):
        prod *= U[i][i]
    return sign * prod


def inverse(A):
    """Matrix inverse A⁻¹ via LU decomposition.

    Solves Ax = eᵢ for each standard basis vector eᵢ to build A⁻¹
    column by column.  Raises ValueError if A is singular.
    """
    _check_square(A)
    n    = len(A)
    L, U, P, _ = lu_decompose(A)
    cols = []
    for j in range(n):
        ej   = [1.0 if i == j else 0.0 for i in range(n)]
        pej  = matvec(P, ej)
        y    = forward_substitution(L, pej)
        x    = back_substitution(U, y)
        cols.append(x)
    # cols[j] is the j-th column of A⁻¹; transpose to get rows
    return [[cols[j][i] for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# Eigenvalues
# ---------------------------------------------------------------------------

def eigenvalues_2x2(A):
    """Eigenvalues of a 2×2 matrix via the quadratic formula.

    For [[a,b],[c,d]], the characteristic polynomial is:
        λ² - (a+d)λ + (ad-bc) = 0

    Returns a list of eigenvalues.  May be complex (returned as tuples
    (real, imag) when discriminant < 0, floats when discriminant ≥ 0).
    """
    _check_square(A)
    if len(A) != 2:
        raise ValueError("eigenvalues_2x2 requires a 2×2 matrix")
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]
    tr   = a + d
    det  = a * d - b * c
    disc = tr * tr - 4.0 * det
    if disc >= 0:
        sq = math.sqrt(disc)
        return [(tr + sq) / 2.0, (tr - sq) / 2.0]
    else:
        sq = math.sqrt(-disc)
        return [(tr / 2.0, sq / 2.0), (tr / 2.0, -sq / 2.0)]


def power_iteration(A, max_iter=1000, tolerance=1e-10, seed=0):
    """Dominant eigenpair of a symmetric matrix by power iteration.

    Repeatedly multiplies a random vector by A and renormalises.
    Converges to the eigenvector corresponding to the largest eigenvalue
    in absolute value.

    Returns (eigenvalue, eigenvector) as (float, list).

    Parameters
    ----------
    A         : n×n symmetric matrix
    max_iter  : maximum iterations
    tolerance : stop when eigenvalue change < tolerance
    seed      : deterministic seed for the initial vector
    """
    _check_square(A)
    n   = len(A)
    # Deterministic starting vector using simple LCG
    lcg = seed + 1
    v   = []
    for _ in range(n):
        lcg = (1664525 * lcg + 1013904223) & 0xFFFFFFFF
        v.append((lcg / 0xFFFFFFFF) - 0.5)
    v = normalize(v)

    lam_prev = 0.0
    for _ in range(max_iter):
        w   = matvec(A, v)
        lam = dot(w, v)
        n_w = norm(w)
        if n_w < 1e-14:
            break
        v = vec_scale(1.0 / n_w, w)
        if abs(lam - lam_prev) < tolerance:
            break
        lam_prev = lam

    lam = dot(matvec(A, v), v)
    return lam, v


# ---------------------------------------------------------------------------
# QR decomposition (modified Gram-Schmidt)
# ---------------------------------------------------------------------------

def qr_decompose(A):
    """QR decomposition via modified Gram-Schmidt orthogonalisation.

    A = QR where Q has orthonormal columns and R is upper triangular.

    Works for m×n matrices with m ≥ n (full column rank assumed).
    Returns (Q, R) where Q is m×n and R is n×n.

    Modified Gram-Schmidt is more numerically stable than classical
    Gram-Schmidt: it orthogonalises each new vector against all
    previously computed basis vectors rather than against the originals.

    Parameters
    ----------
    A : m×n matrix with m ≥ n
    """
    _check_matrix(A)
    m, n = shape(A)
    if m < n:
        raise ValueError("QR requires m >= n (more rows than columns)")

    Q = copy_matrix(A)   # will hold the orthonormal columns
    R = zeros(n, n)

    for j in range(n):
        # Column j of Q starts as column j of A (already copied)
        col_j = [Q[i][j] for i in range(m)]

        # Orthogonalise against all previous columns
        for k in range(j):
            col_k     = [Q[i][k] for i in range(m)]
            R[k][j]   = dot(col_k, col_j)
            for i in range(m):
                Q[i][j] -= R[k][j] * Q[i][k]

        # Normalise
        col_j    = [Q[i][j] for i in range(m)]
        R[j][j]  = norm(col_j)
        if R[j][j] < 1e-14:
            raise ValueError("columns of A are not linearly independent")
        for i in range(m):
            Q[i][j] /= R[j][j]

    return Q, R


# ---------------------------------------------------------------------------
# Least squares
# ---------------------------------------------------------------------------

def least_squares(A, b):
    """Solve the least-squares problem min ||Ax - b||₂ via QR decomposition.

    For an overdetermined system (m > n), solves the normal equations
    using QR: A = QR, so Ax = b becomes Rx = Qᵀb.

    Returns x_hat, the vector that minimises the residual norm.

    Parameters
    ----------
    A : m×n matrix (m ≥ n)
    b : list of m floats
    """
    _check_matrix(A)
    m, n = shape(A)
    if len(b) != m:
        raise ValueError("b must have length m = {}".format(m))
    if m < n:
        raise ValueError("least_squares requires m >= n")

    Q, R = qr_decompose(A)
    # Qᵀ b  (Q is m×n)
    Qt = transpose(Q)
    Qtb = matvec(Qt, b)
    return back_substitution(R, Qtb)


def residual_norm(A, b, x):
    """||Ax - b||₂, the residual of a least-squares solution."""
    Ax = matvec(A, x)
    return norm(vec_sub(Ax, b))


# ---------------------------------------------------------------------------
# Condition number approximation
# ---------------------------------------------------------------------------

def condition_number(A, max_iter=200):
    """Approximate 2-norm condition number of A via power iteration on AᵀA.

    κ(A) = σ_max / σ_min where σ are singular values.
    σ_max = sqrt(λ_max(AᵀA)), σ_min approximated via inverse iteration.

    Returns an approximation accurate to a few significant figures for
    well-conditioned matrices.

    Parameters
    ----------
    A        : m×n matrix
    max_iter : iterations for power iteration
    """
    _check_matrix(A)
    At  = transpose(A)
    AtA = matmul(At, A)

    lam_max, _ = power_iteration(AtA, max_iter=max_iter)
    sigma_max   = math.sqrt(max(lam_max, 0.0))

    # Inverse iteration on AᵀA for smallest eigenvalue:
    # Use deflation: subtract lam_max * v*vᵀ and iterate again
    n = len(AtA)
    _, v_max = power_iteration(AtA, max_iter=max_iter)
    # Deflated matrix B = AᵀA - lam_max * v_max * v_max^T
    B = copy_matrix(AtA)
    for i in range(n):
        for j in range(n):
            B[i][j] -= lam_max * v_max[i] * v_max[j]

    lam_second, _ = power_iteration(B, max_iter=max_iter, seed=1)
    sigma_min = math.sqrt(max(abs(lam_second), 1e-30))

    if sigma_min < 1e-15:
        return math.inf
    return sigma_max / sigma_min
