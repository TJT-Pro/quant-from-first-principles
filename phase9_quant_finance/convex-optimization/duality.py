from typing import Callable, List

import linear_algebra as la
import newtons_method as nm


def lagrangian(f: Callable[[List[float]], float], h_list: List[Callable[[List[float]], float]],
               g_list: List[Callable[[List[float]], float]], x: List[float],
               lam: List[float], mu: List[float]) -> float:
    """L(x, lambda, mu) = f(x) + sum_i lambda_i*h_i(x) + sum_j mu_j*g_j(x),
    for equality constraints h_i(x)=0 and inequality constraints
    g_j(x)<=0. The Lagrangian folds the constraints into the objective
    with a price (multiplier) attached to each: lambda_i can be any
    sign (violating h_i=0 in either direction is penalised or rewarded
    symmetrically, since the constraint could be violated in either
    direction), while mu_j must eventually be required non-negative
    (see dual_function), since only g_j(x)>0 (a genuine violation)
    should ever be penalised, never rewarded.
    """
    value = f(x)
    for lam_i, h_i in zip(lam, h_list):
        value += lam_i * h_i(x)
    for mu_j, g_j in zip(mu, g_list):
        value += mu_j * g_j(x)
    return value


def dual_function(f: Callable[[List[float]], float], h_list: List[Callable[[List[float]], float]],
                   g_list: List[Callable[[List[float]], float]], lam: List[float], mu: List[float],
                   x0: List[float]) -> dict:
    """The Lagrangian dual function g(lambda, mu) = inf_x L(x, lambda, mu):
    for *fixed* multipliers, minimise the Lagrangian over x with no
    constraints at all (the constraints have already been absorbed
    into the objective via the multiplier terms). This inner
    minimisation is carried out here with the project's own Newton's
    method, since the Lagrangian of a convex problem with affine or
    convex constraints is itself convex in x for any fixed, dual-
    feasible (mu>=0) multipliers, making unconstrained Newton's method
    a natural and fast choice for evaluating the dual function at a
    given point.
    """
    l_of_x = lambda x: lagrangian(f, h_list, g_list, x, lam, mu)
    result = nm.newtons_method(l_of_x, x0)
    return {"dual_value": result["f"], "minimizing_x": result["x"], "converged": result["converged"]}


def verify_weak_duality(primal_value: float, dual_value: float, tol: float = 1e-9) -> dict:
    """Weak duality, g(lambda,mu) <= f(x) for ANY primal-feasible x and
    ANY dual-feasible (mu>=0) multipliers, holds unconditionally for
    every optimization problem, convex or not: it follows immediately
    from L(x,lambda,mu) <= f(x) for feasible x and mu>=0 (since
    h_i(x)=0 contributes nothing and mu_j*g_j(x)<=0 for feasible x with
    g_j(x)<=0), combined with g(lambda,mu) being an infimum over ALL x,
    hence no larger than the Lagrangian's value at this particular
    feasible x. The duality gap, primal_value - dual_value, is
    therefore always non-negative; this function checks exactly that.
    """
    gap = primal_value - dual_value
    return {"weak_duality_holds": gap >= -tol, "duality_gap": gap}


def verify_strong_duality(primal_optimal_value: float, dual_optimal_value: float, tol: float = 1e-4) -> dict:
    """Strong duality, a duality gap of exactly zero at the optimum, is
    NOT automatic (unlike weak duality above); it requires additional
    structure, most commonly convexity of the primal problem together
    with a constraint qualification such as Slater's condition
    (existence of a strictly feasible point for every inequality
    constraint). It is checked here as an approximate numerical
    equality rather than asserted from the convexity of the problems
    this project constructs, since the actual gap achieved by any
    concrete numerical dual-ascent procedure will not be exactly zero
    in floating point.
    """
    gap = primal_optimal_value - dual_optimal_value
    return {"strong_duality_holds": abs(gap) < tol, "duality_gap": gap}


def maximize_dual_via_projected_gradient_ascent(f: Callable[[List[float]], float],
                                                 h_list: List[Callable[[List[float]], float]],
                                                 g_list: List[Callable[[List[float]], float]],
                                                 lam0: List[float], mu0: List[float], x0: List[float],
                                                 step: float = 0.1, tol: float = 1e-6,
                                                 max_iter: int = 200) -> dict:
    """Maximise the dual function over (lambda, mu) with mu>=0, by
    projected gradient ascent: the dual function's gradient with
    respect to the multipliers is exactly the constraint values
    themselves at the Lagrangian's current minimiser
    (d g/d lambda_i = h_i(x*(lambda,mu)), d g/d mu_j = g_j(x*(lambda,mu)),
    a consequence of the envelope theorem applied to an inner
    minimisation), so no separate gradient computation is needed
    beyond evaluating the constraints at each step's inner minimiser.
    mu is projected onto mu>=0 after each ascent step (a simple box
    projection at zero), enforcing dual feasibility throughout;
    lambda is left unconstrained, matching that equality-constraint
    multipliers carry no sign restriction.
    """
    lam, mu = list(lam0), list(mu0)
    history = []
    for iteration in range(max_iter):
        result = dual_function(f, h_list, g_list, lam, mu, x0)
        x_star = result["minimizing_x"]
        history.append(result["dual_value"])

        h_values = [h(x_star) for h in h_list]
        g_values = [g(x_star) for g in g_list]

        lam_new = la.vec_add(lam, la.vec_scale(h_values, step)) if lam else []
        mu_new = [max(m + step * gv, 0.0) for m, gv in zip(mu, g_values)]

        move = la.vector_norm(la.vec_sub(lam_new, lam)) + la.vector_norm(la.vec_sub(mu_new, mu)) if lam else \
            la.vector_norm(la.vec_sub(mu_new, mu))
        lam, mu = lam_new, mu_new
        x0 = x_star
        if move < tol:
            return {"lam": lam, "mu": mu, "dual_value": result["dual_value"], "x": x_star,
                    "iterations": iteration + 1, "converged": True, "history": history}
    return {"lam": lam, "mu": mu, "dual_value": history[-1] if history else None, "x": x0,
            "iterations": max_iter, "converged": False, "history": history}
