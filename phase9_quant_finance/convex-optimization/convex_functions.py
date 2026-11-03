import random
from typing import Callable, List

import linear_algebra as la


def numerical_gradient(f: Callable[[List[float]], float], x: List[float], h: float = 1e-6) -> List[float]:
    """The gradient of f at x, via central differences in each
    coordinate: d f/d x_i ~ [f(x + h*e_i) - f(x - h*e_i)] / (2h). This
    is used throughout the project as the ground-truth numerical
    check against every analytically-derived gradient, and directly as
    the search direction for the gradient-method implementations
    themselves when a closed-form gradient is inconvenient to derive.
    """
    n = len(x)
    grad = []
    for i in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad


def numerical_hessian(f: Callable[[List[float]], float], x: List[float], h: float = 1e-4) -> la.Matrix:
    """The Hessian of f at x, via central second differences:

        d^2f/dx_i dx_j ~ [f(x+h e_i+h e_j) - f(x+h e_i-h e_j)
                          - f(x-h e_i+h e_j) + f(x-h e_i-h e_j)] / (4h^2)

    for i != j (the standard mixed-partial finite-difference stencil),
    and the plain second-difference stencil
    [f(x+h e_i) - 2f(x) + f(x-h e_i)] / h^2 for the diagonal (i=j)
    entries. The result is symmetric by construction (the mixed
    formula is symmetric in i and j), consistent with Clairaut's
    theorem that mixed partials commute for any function with
    continuous second derivatives, which every function this project
    actually applies this to satisfies.
    """
    n = len(x)
    hess = [[0.0] * n for _ in range(n)]
    fx = f(x)
    for i in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[i] += h
        x_minus[i] -= h
        hess[i][i] = (f(x_plus) - 2 * fx + f(x_minus)) / (h ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            x_pp, x_pm, x_mp, x_mm = list(x), list(x), list(x), list(x)
            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h
            value = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h ** 2)
            hess[i][j] = value
            hess[j][i] = value
    return hess


def check_convexity_by_definition(f: Callable[[List[float]], float], domain_sampler: Callable[[], List[float]],
                                   n_trials: int = 2000, seed: int = 0, tol: float = 1e-9) -> dict:
    """Test the defining inequality of convexity directly:

        f(lambda*x + (1-lambda)*y) <= lambda*f(x) + (1-lambda)*f(y)

    for many random pairs (x, y) drawn from `domain_sampler` and many
    random lambda in [0,1]. This is a Monte Carlo *falsification* test,
    not a proof: it can only ever demonstrate that a function is NOT
    convex (by finding a single violation), never prove convexity
    exactly, since convexity is a statement about every point pair and
    every lambda, an infinite set no finite sample can exhaust. It is
    reported honestly as such ("convex_on_samples", not "is_convex")
    and paired with the exact, second-order Sylvester's-criterion check
    below for functions where the Hessian is available, which can
    certify convexity exactly on the region checked.
    """
    rng = random.Random(seed)
    worst_violation = 0.0
    for _ in range(n_trials):
        x = domain_sampler()
        y = domain_sampler()
        lam = rng.random()
        mixed = [lam * xi + (1 - lam) * yi for xi, yi in zip(x, y)]
        lhs = f(mixed)
        rhs = lam * f(x) + (1 - lam) * f(y)
        violation = lhs - rhs
        if violation > worst_violation:
            worst_violation = violation
    return {"convex_on_samples": worst_violation <= tol, "worst_violation": worst_violation, "n_trials": n_trials}


def is_positive_semidefinite_2x2_leading_minors(hessian: la.Matrix, tol: float = 1e-9) -> bool:
    """Sylvester's criterion: a symmetric matrix is positive
    semi-definite (on the boundary case) / positive definite (strict
    case) according to the sign pattern of its leading principal
    minors. Checked here with a small non-negativity tolerance since
    finite-difference Hessians are never exactly symmetric or exactly
    at their true values.
    """
    n = len(hessian)
    for i in range(n):
        if hessian[i][i] < -tol:
            return False
    minors = la.leading_principal_minors(hessian)
    return all(m >= -tol for m in minors)


def convexity_via_hessian(f: Callable[[List[float]], float], sample_points: List[List[float]],
                           h: float = 1e-4) -> dict:
    """The second-order convexity condition: a twice-differentiable
    function is convex on a convex domain if and only if its Hessian
    is positive semi-definite everywhere on that domain. Unlike the
    definition-sampling check above, this check is exact (up to the
    finite-difference approximation of the Hessian itself) at each
    point tested: PSD-ness of the true Hessian at a point is a
    necessary and sufficient local condition, not a sampled
    approximation of a global one. Checked via Sylvester's criterion
    (all leading principal minors non-negative) at each of the
    supplied sample points, rather than via a full eigenvalue
    decomposition, since Sylvester's criterion needs only ordinary
    determinants of the small Hessians this project ever works with.
    """
    results = []
    all_psd = True
    for x in sample_points:
        hessian = numerical_hessian(f, x, h)
        psd = is_positive_semidefinite_2x2_leading_minors(hessian)
        results.append({"point": x, "hessian": hessian, "psd": psd})
        if not psd:
            all_psd = False
    return {"convex_by_hessian_test": all_psd, "details": results}
