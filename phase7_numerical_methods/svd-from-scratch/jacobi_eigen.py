import math
from typing import Tuple

import linear_algebra as la


def jacobi_eigen(a: la.Matrix, tol: float = 1e-12, max_sweeps: int = 100) -> Tuple[la.Vector, la.Matrix]:
    """Eigenvalues and eigenvectors of a symmetric matrix A, via the
    classical (cyclic) Jacobi eigenvalue algorithm.

    The idea: a symmetric matrix can be diagonalised by a sequence of
    plane rotations, each chosen to zero out one specific off-diagonal
    pair A[p][q]. Using the rotation matrix R (identity except for the
    (p,q) block, where R[p][p]=R[q][q]=c=cos(theta), R[p][q]=s=sin(theta),
    R[q][p]=-s) and applying A' = R^T*A*R, the transformed diagonal and
    off-diagonal entries in the (p,q) block work out to

        A'[p][p] = c^2*A[p][p] - 2*s*c*A[p][q] + s^2*A[q][q]
        A'[q][q] = s^2*A[p][p] + 2*s*c*A[p][q] + c^2*A[q][q]
        A'[p][q] = A'[q][p] = s*c*(A[p][p]-A[q][q]) + A[p][q]*(c^2-s^2)

    Setting A'[p][q]=0 and solving for theta gives
    theta = -0.5*atan2(2*A[p][q], A[p][p]-A[q][q]) for this specific
    sign convention of R (a different, equally valid convention for
    which entry of R carries the minus sign would flip this sign; the
    two must be paired consistently, which is exactly the bug found
    directly while testing this implementation: an earlier version
    used the textbook +0.5*atan2(...) formula together with these same
    row/column update formulas, and while the *eigenvalues* it produced
    were still correct on the diagonal, the *eigenvectors* accumulated
    into V were wrong, because that theta corresponds to a different R
    than the one the row/column updates above actually implement. The
    mismatch was invisible from the eigenvalues alone, and from V
    alone (V came out perfectly orthogonal either way, since it is a
    product of genuine rotation matrices regardless of their angles'
    sign), and only showed up when checking A*v = lambda*v directly
    for individual eigenvectors, or equivalently V^T*A*V against the
    diagonal matrix of eigenvalues, which is why both checks are run
    directly in the test suite rather than checking orthogonality or
    the eigenvalues alone.

    A rotation cannot zero every off-diagonal entry simultaneously
    unless A is already diagonal (zeroing A[p][q] generally disturbs
    other entries), but repeated full sweeps through every (p,q) pair
    provably drive the sum of squared off-diagonal entries to zero,
    with quadratic convergence once the matrix is nearly diagonal. The
    accumulated product of every rotation applied, tracked in V, is
    exactly the eigenvector matrix: A = V*Lambda*V^T is equivalent to
    V^T*A*V = Lambda, and each Jacobi rotation contributes one factor
    of that product.
    """
    if not la.is_symmetric(a):
        raise ValueError("jacobi_eigen requires a symmetric matrix")
    n = len(a)
    a_work = [row[:] for row in a]
    v = la.identity(n)

    for _ in range(max_sweeps):
        off_diag_sq = sum(a_work[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off_diag_sq < tol:
            break

        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a_work[p][q]) < 1e-300:
                    continue

                if abs(a_work[p][p] - a_work[q][q]) < 1e-300:
                    theta = -math.pi / 4 if a_work[p][q] > 0 else math.pi / 4
                else:
                    theta = -0.5 * math.atan2(2 * a_work[p][q], a_work[p][p] - a_work[q][q])
                c, s = math.cos(theta), math.sin(theta)

                app, aqq, apq = a_work[p][p], a_work[q][q], a_work[p][q]
                a_work[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
                a_work[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
                a_work[p][q] = 0.0
                a_work[q][p] = 0.0

                for k in range(n):
                    if k != p and k != q:
                        akp, akq = a_work[k][p], a_work[k][q]
                        a_work[k][p] = c * akp - s * akq
                        a_work[p][k] = a_work[k][p]
                        a_work[k][q] = s * akp + c * akq
                        a_work[q][k] = a_work[k][q]

                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq

    eigenvalues = [a_work[i][i] for i in range(n)]
    return eigenvalues, v


def sorted_eigen(a: la.Matrix, tol: float = 1e-12, max_sweeps: int = 100) -> Tuple[la.Vector, la.Matrix]:
    """jacobi_eigen's output, reordered so eigenvalues appear in
    descending order (the standard convention for SVD and PCA, where
    the first component/singular value is meant to be the dominant
    one), with the corresponding eigenvector columns reordered to
    match.
    """
    eigenvalues, v = jacobi_eigen(a, tol, max_sweeps)
    n = len(eigenvalues)
    order = sorted(range(n), key=lambda i: eigenvalues[i], reverse=True)
    sorted_values = [eigenvalues[i] for i in order]
    sorted_vectors = [[v[row][i] for i in order] for row in range(n)]
    return sorted_values, sorted_vectors
