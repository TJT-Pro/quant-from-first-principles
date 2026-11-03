from typing import Callable, List, Optional

import linear_algebra as la
import convex_functions as cf


def backtracking_line_search(f: Callable[[List[float]], float], x: List[float], grad: List[float],
                              direction: List[float], alpha_init: float = 1.0,
                              c: float = 1e-4, rho: float = 0.5, max_iter: int = 50) -> float:
    """The Armijo backtracking line search: starting from a generous
    step size alpha_init, shrink it by a factor of rho until the
    sufficient-decrease condition

        f(x + alpha*d) <= f(x) + c*alpha*grad(x).d

    holds, for a small constant c (typically 1e-4). The right-hand
    side is the value a first-order Taylor approximation would predict
    if the function decreased at only a small fraction (c) of the
    rate the gradient itself suggests; requiring at least that much
    guaranteed decrease, rather than the full amount the linear
    approximation predicts, is what keeps the condition satisfiable
    even though the linear approximation is only locally accurate.
    Without some such condition, plain gradient descent with a fixed
    step size can easily overshoot and diverge on a function whose
    curvature varies across the domain.
    """
    alpha = alpha_init
    fx = f(x)
    directional_derivative = la.dot(grad, direction)
    for _ in range(max_iter):
        x_new = la.vec_add(x, la.vec_scale(direction, alpha))
        if f(x_new) <= fx + c * alpha * directional_derivative:
            return alpha
        alpha *= rho
    return alpha


def gradient_descent(f: Callable[[List[float]], float], x0: List[float],
                      grad_fn: Optional[Callable[[List[float]], List[float]]] = None,
                      tol: float = 1e-8, max_iter: int = 10000, use_line_search: bool = True,
                      fixed_step: float = 0.01) -> dict:
    """Minimise f by steepest descent: at each iterate, move in the
    direction of steepest local decrease (the negative gradient),
    since among all unit directions d, the directional derivative
    grad(x).d is minimised (most negative) exactly when d points
    opposite the gradient (by the Cauchy-Schwarz inequality, with
    equality when d is anti-parallel to the gradient). Either a fixed
    step size or an Armijo backtracking line search sets how far to
    move along that direction each iteration; the line-search variant
    is the default because it removes the need to hand-tune a step
    size for each problem's curvature, at the cost of extra function
    evaluations per iteration.
    """
    grad_fn = grad_fn or (lambda x: cf.numerical_gradient(f, x))
    x = list(x0)
    history = [f(x)]
    for iteration in range(max_iter):
        grad = grad_fn(x)
        grad_norm = la.vector_norm(grad)
        if grad_norm < tol:
            return {"x": x, "f": f(x), "iterations": iteration, "converged": True, "history": history}
        direction = la.vec_scale(grad, -1.0)
        if use_line_search:
            step = backtracking_line_search(f, x, grad, direction)
        else:
            step = fixed_step
        x = la.vec_add(x, la.vec_scale(direction, step))
        history.append(f(x))
    return {"x": x, "f": f(x), "iterations": max_iter, "converged": False, "history": history}
