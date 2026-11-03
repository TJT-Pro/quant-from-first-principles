import math
from typing import Tuple

import linear_algebra as la
import jacobi_eigen as je


def singular_value_decompose(a: la.Matrix, tol: float = 1e-10) -> Tuple[la.Matrix, la.Vector, la.Matrix]:
    """Compute A = U * diag(S) * V^T for an m x n matrix A (m >= n
    assumed, the usual "tall" data-matrix shape: more observations
    than features), by eigendecomposing the n x n matrix A^T*A.

    Why the singular values are the square roots of A^T*A's
    eigenvalues: substituting A = U*Sigma*V^T into A^T*A gives

        A^T*A = V*Sigma^T*U^T*U*Sigma*V^T = V*Sigma^2*V^T

    using U^T*U = I (U has orthonormal columns), which is exactly an
    eigendecomposition of the symmetric matrix A^T*A, with eigenvalues
    sigma_i^2 and eigenvectors given by V's columns. A^T*A is always
    positive semi-definite (x^T*A^T*A*x = ||Ax||^2 >= 0 for every x),
    so every eigenvalue is non-negative and every singular value
    sigma_i = sqrt(eigenvalue_i) is a well-defined real number.

    Given V from this eigendecomposition, each corresponding column of
    U (for a nonzero singular value) follows directly from substituting
    back into A = U*Sigma*V^T and using V's orthonormality:
    A*v_i = sigma_i*u_i, so u_i = A*v_i / sigma_i. For a singular value
    that is (numerically) zero, no such formula determines u_i (any
    unit vector orthogonal to the other U columns would satisfy the
    decomposition equally well), so those columns are instead filled
    in by Gram-Schmidt against the columns already computed, completing
    an orthonormal basis of R^m without pretending a specific direction
    was recovered from the data.
    """
    m, n = la.shape(a)
    if m < n:
        raise ValueError("this implementation assumes m >= n (more rows than columns); "
                          "transpose A and swap U/V to handle the m < n case")

    at = la.transpose(a)
    ata = la.matmul(at, a)
    eigenvalues, v = je.sorted_eigen(ata)

    singular_values = [math.sqrt(ev) if ev > 0 else 0.0 for ev in eigenvalues]

    u_columns = []
    for i in range(n):
        vi = la.column(v, i)
        if singular_values[i] > tol:
            avi = la.mat_vec(a, vi)
            ui = [x / singular_values[i] for x in avi]
            u_columns.append(ui)
        else:
            u_columns.append(None)

    _fill_missing_columns_via_gram_schmidt(u_columns, m)

    u = [[u_columns[j][i] for j in range(n)] for i in range(m)]
    return u, singular_values, v


def _fill_missing_columns_via_gram_schmidt(columns, dim: int) -> None:
    """Fill any `None` entries in `columns` (a list of either unit
    vectors in R^dim or None placeholders) with new unit vectors,
    chosen via Gram-Schmidt against every column already present, so
    the final list is a complete orthonormal set. Standard basis
    vectors are tried in order as Gram-Schmidt starting points, since
    at least one must have a non-negligible component orthogonal to
    the existing (rank-deficient) span whenever more columns remain to
    be filled than the existing columns already span.
    """
    existing = [c for c in columns if c is not None]
    for idx in range(len(columns)):
        if columns[idx] is not None:
            continue
        for basis_idx in range(dim):
            candidate = [1.0 if k == basis_idx else 0.0 for k in range(dim)]
            for e in existing:
                proj = la.dot(candidate, e)
                candidate = [c - proj * ei for c, ei in zip(candidate, e)]
            norm = la.vector_norm(candidate)
            if norm > 1e-8:
                new_col = [c / norm for c in candidate]
                columns[idx] = new_col
                existing.append(new_col)
                break
        else:
            raise ValueError("could not complete an orthonormal basis; this should not happen")


def reconstruct(u: la.Matrix, s: la.Vector, v: la.Matrix) -> la.Matrix:
    if not s:
        # A degenerate (zero-width) matrix product cannot be formed by
        # generic matmul here: representing a matrix with zero columns
        # as a list of empty rows loses track of how many columns the
        # *result* should have once it is transposed and multiplied
        # again (transposing a list of empty rows produces an empty
        # list of rows entirely, not a list of zero-length rows, so
        # the final row count silently drops to zero too). Found
        # directly while testing low_rank_approximation(..., k=0):
        # rather than paper over it inside matmul/transpose generically
        # (which would need every caller of this project's plain
        # nested-list matrix type to agree on a convention for
        # zero-dimension matrices), the one place that actually needs
        # a well-shaped zero matrix constructs it explicitly.
        m = len(u)
        n = len(v)
        return la.zeros(m, n)
    sigma = [[s[i] if i == j else 0.0 for j in range(len(s))] for i in range(len(s))]
    return la.matmul(la.matmul(u, sigma), la.transpose(v))


def low_rank_approximation(u: la.Matrix, s: la.Vector, v: la.Matrix, k: int) -> la.Matrix:
    """The best rank-k approximation to A in both the Frobenius and
    spectral (operator) norms, by the Eckart-Young-Mirsky theorem:
    truncating the SVD to its k largest singular values and
    corresponding singular vectors, A_k = sum_{i=1}^k sigma_i*u_i*v_i^T,
    minimises ||A - B|| over every rank-k-or-less matrix B, for either
    norm simultaneously. Truncating in descending singular-value order
    (guaranteed by `singular_value_decompose`'s use of `sorted_eigen`)
    is what makes this the *best* such approximation rather than merely
    *a* rank-k approximation: keeping any other k singular
    triples instead would leave a strictly larger reconstruction error.
    """
    if k < 0 or k > len(s):
        raise ValueError("k must be between 0 and the number of singular values")
    u_k = [row[:k] for row in u]
    s_k = s[:k]
    v_k = [row[:k] for row in v]
    return reconstruct(u_k, s_k, v_k)


def pseudoinverse(u: la.Matrix, s: la.Vector, v: la.Matrix, tol: float = 1e-10) -> la.Matrix:
    """The Moore-Penrose pseudoinverse, A^+ = V * Sigma^+ * U^T, where
    Sigma^+ replaces every nonzero singular value with its reciprocal
    and leaves every (numerically) zero singular value as zero rather
    than dividing by it. This is the unique matrix satisfying the four
    Moore-Penrose conditions (A*A^+*A=A, A^+*A*A^+=A^+, and both
    A*A^+ and A^+*A symmetric), and for a full-column-rank A it
    coincides with (A^T*A)^{-1}*A^T, the normal-equations solution to
    least squares; for a rank-deficient A, the pseudoinverse
    specifically gives the minimum-norm solution among all vectors
    minimising ||Ax-b||, which is exactly why zero singular values are
    mapped to zero (not any other value) in Sigma^+.
    """
    sigma_plus = [1.0 / si if si > tol else 0.0 for si in s]
    sigma_plus_matrix = [[sigma_plus[i] if i == j else 0.0 for j in range(len(s))] for i in range(len(s))]
    return la.matmul(la.matmul(v, sigma_plus_matrix), la.transpose(u))


def rank_estimate(s: la.Vector, rel_tol: float = 1e-10) -> int:
    """The numerical rank: the count of singular values exceeding
    rel_tol times the largest singular value. Using a *relative*
    threshold (rather than a fixed absolute one) is standard practice,
    since it makes the rank estimate invariant to an overall rescaling
    of A (A and 1000*A should be judged to have the same rank), which
    a fixed absolute cutoff would not respect.
    """
    if not s:
        return 0
    threshold = rel_tol * s[0]
    return sum(1 for si in s if si > threshold)


def condition_number(s: la.Vector, rel_tol: float = 1e-10) -> float:
    """sigma_max / sigma_min, restricted to the numerically nonzero
    singular values (using the same relative threshold as
    `rank_estimate`, for consistency): a measure of how much a matrix
    can amplify relative error. A condition number near 1 means A is
    close to a scaled orthogonal transformation (all directions
    stretched about equally); a very large condition number means A
    is close to rank-deficient (some direction is stretched by a
    factor far smaller than another), so solving A*x=b for such an A
    can amplify small errors or noise in b into large errors in the
    recovered x. Returns infinity if fewer than two singular values
    are numerically nonzero, since a condition number is not
    meaningfully defined for a rank-0 or rank-1 (from an original
    higher-dimensional) matrix in the usual sense.
    """
    rank = rank_estimate(s, rel_tol)
    if rank < 1:
        return float("inf")
    nonzero = s[:rank]
    if len(nonzero) < 1 or nonzero[-1] <= 0:
        return float("inf")
    return nonzero[0] / nonzero[-1]
