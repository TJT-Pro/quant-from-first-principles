from typing import Callable, List, Optional

import linear_algebra as la
import convex_functions as cf


def project_onto_simplex(v: List[float]) -> List[float]:
    """Euclidean projection of v onto the probability simplex
    {x : x_i >= 0, sum(x) = 1}, via the standard sort-based algorithm
    (Held, Wolfe & Crowder 1974; also Duchi et al. 2008).

    The projection is the x minimising ||x-v||^2 subject to x>=0 and
    sum(x)=1. Introducing a Lagrange multiplier theta for the equality
    constraint and multipliers mu_i>=0 for each x_i>=0, the KKT
    stationarity condition gives x_i = max(v_i - theta, 0) for some
    single scalar theta shared across all coordinates (each coordinate
    is independently thresholded by the same shift). The remaining
    question is which value of theta makes sum_i max(v_i-theta,0)
    equal exactly 1. Sorting v in descending order as u_1>=...>=u_n,
    the correct theta is determined by finding the largest rho such
    that u_rho - (1/rho)*(sum_{i=1}^rho u_i - 1) is still positive
    (equivalently: including exactly the top rho coordinates as
    "active," i.e. not clipped to zero, is self-consistent), then
    theta = (1/rho)*(sum_{i=1}^rho u_i - 1). This is an exact,
    finite algorithm (no iteration needed), not an approximation.
    """
    n = len(v)
    u = sorted(v, reverse=True)
    cumulative = 0.0
    rho = 0
    theta = 0.0
    for j in range(1, n + 1):
        cumulative += u[j - 1]
        candidate_theta = (cumulative - 1.0) / j
        if u[j - 1] - candidate_theta > 0:
            rho = j
            theta = candidate_theta
    return [max(vi - theta, 0.0) for vi in v]


def project_onto_box(v: List[float], lower: List[float], upper: List[float]) -> List[float]:
    return [min(max(vi, lo), hi) for vi, lo, hi in zip(v, lower, upper)]


def projected_gradient_descent(f: Callable[[List[float]], float], x0: List[float],
                                project: Callable[[List[float]], List[float]],
                                grad_fn: Optional[Callable[[List[float]], List[float]]] = None,
                                step: float = 0.01, tol: float = 1e-9, max_iter: int = 100000) -> dict:
    """Minimise f subject to x lying in a convex set C, given a
    Euclidean projection operator onto C: take an ordinary gradient
    step, then project the result back onto C. This is guaranteed to
    converge for a convex, Lipschitz-smooth f and a small enough fixed
    step size, since projection onto a convex set is itself a
    contraction (non-expansive) map, so the projection step cannot
    undo more progress than the gradient step made, only ever move the
    iterate closer to the feasible set than an unprojected step would
    leave it relative to any feasible point. Convergence is declared
    when successive projected iterates stop moving by more than `tol`,
    since at a constrained optimum the projected-gradient step is a
    fixed point of this exact iteration (the gradient step followed by
    projection reproduces the same point), unlike unconstrained
    gradient descent's stopping rule of a small gradient norm, which
    does not apply once the true stationarity condition at a
    constrained optimum generally has a nonzero gradient balanced
    against the constraint's own multiplier.
    """
    grad_fn = grad_fn or (lambda pt: cf.numerical_gradient(f, pt))
    x = project(list(x0))
    history = [f(x)]
    for iteration in range(max_iter):
        grad = grad_fn(x)
        x_new = project(la.vec_sub(x, la.vec_scale(grad, step)))
        move = la.vector_norm(la.vec_sub(x_new, x))
        x = x_new
        history.append(f(x))
        if move < tol:
            return {"x": x, "f": f(x), "iterations": iteration + 1, "converged": True, "history": history}
    return {"x": x, "f": f(x), "iterations": max_iter, "converged": False, "history": history}


def solve_equality_constrained_qp(q: la.Matrix, c: List[float], a: la.Matrix, b: List[float]) -> dict:
    """Solve min 0.5*x^T*Q*x + c^T*x subject to A*x=b exactly, via the
    KKT (Lagrangian stationarity) linear system. The Lagrangian is
    L(x,lambda) = 0.5*x^T*Q*x + c^T*x + lambda^T*(A*x-b), and setting
    both partial derivatives to zero gives

        Q*x + c + A^T*lambda = 0
        A*x - b = 0

    a single linear system in the stacked unknowns [x; lambda]:

        [[Q, A^T], [A, 0]] * [x; lambda] = [-c; b]

    solved directly here rather than iteratively, since for a convex
    quadratic objective (Q positive semi-definite) with linear equality
    constraints, this system's solution is exactly the global
    constrained minimiser: no iteration, line search, or step-size
    tuning is needed at all, which is the entire appeal of solving a
    QP via its KKT system directly whenever Q and A are small enough
    to form and solve this stacked system exactly.
    """
    n = len(q)
    m = len(a)
    kkt_matrix = [q[i] + [a[k][i] for k in range(m)] for i in range(n)]
    kkt_matrix += [a[k] + [0.0] * m for k in range(m)]
    rhs = [-ci for ci in c] + list(b)
    solution = la.solve_linear_system(kkt_matrix, rhs)
    x = solution[:n]
    lam = solution[n:]
    return {"x": x, "lambda": lam}


def verify_kkt(grad_f: List[float], grad_h: List[List[float]], h_values: List[float], lam: List[float],
               grad_g: List[List[float]] = None, g_values: List[float] = None, mu: List[float] = None,
               tol: float = 1e-6) -> dict:
    """Verify the Karush-Kuhn-Tucker conditions at a candidate solution
    for a problem with equality constraints h_i(x)=0 (multipliers
    lambda_i, unrestricted in sign) and inequality constraints
    g_j(x)<=0 (multipliers mu_j>=0):

        Stationarity:            grad_f + sum_i lambda_i*grad_h_i + sum_j mu_j*grad_g_j = 0
        Primal feasibility:      h_i(x)=0 for all i, g_j(x)<=0 for all j
        Dual feasibility:        mu_j >= 0 for all j
        Complementary slackness: mu_j*g_j(x)=0 for all j

    For a convex problem (convex objective, affine equality
    constraints, convex inequality constraints) satisfying a
    constraint qualification, these four conditions together are both
    necessary and sufficient for global optimality, which is why they
    are checked as four separate, individually named booleans here
    rather than collapsed into a single pass/fail: a failure in one
    specific condition (e.g. dual feasibility, mu_j<0) points at a
    genuinely different kind of problem with the candidate solution
    than a failure in another (e.g. primal feasibility, an infeasible
    point) would.
    """
    grad_g = grad_g or []
    g_values = g_values or []
    mu = mu or []

    stationarity_vector = list(grad_f)
    for lam_i, gh_i in zip(lam, grad_h):
        stationarity_vector = la.vec_add(stationarity_vector, la.vec_scale(gh_i, lam_i))
    for mu_j, gg_j in zip(mu, grad_g):
        stationarity_vector = la.vec_add(stationarity_vector, la.vec_scale(gg_j, mu_j))
    stationarity_residual = la.vector_norm(stationarity_vector)
    stationarity_ok = stationarity_residual < tol

    primal_eq_ok = all(abs(h) < tol for h in h_values)
    primal_ineq_ok = all(g <= tol for g in g_values)
    dual_feasible_ok = all(m >= -tol for m in mu)
    complementary_slackness_ok = all(abs(m * g) < tol for m, g in zip(mu, g_values))

    return {
        "stationarity": stationarity_ok,
        "stationarity_residual": stationarity_residual,
        "primal_feasibility_equality": primal_eq_ok,
        "primal_feasibility_inequality": primal_ineq_ok,
        "dual_feasibility": dual_feasible_ok,
        "complementary_slackness": complementary_slackness_ok,
        "all_conditions_satisfied": (stationarity_ok and primal_eq_ok and primal_ineq_ok
                                      and dual_feasible_ok and complementary_slackness_ok),
    }
