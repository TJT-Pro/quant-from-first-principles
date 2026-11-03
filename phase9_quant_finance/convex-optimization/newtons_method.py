from typing import Callable, List, Optional

import linear_algebra as la
import convex_functions as cf


def newtons_method(f: Callable[[List[float]], float], x0: List[float],
                    grad_fn: Optional[Callable[[List[float]], List[float]]] = None,
                    hess_fn: Optional[Callable[[List[float]], la.Matrix]] = None,
                    tol: float = 1e-10, max_iter: int = 100) -> dict:
    """Minimise f by Newton's method: at each iterate, minimise the
    local second-order (quadratic) Taylor approximation of f exactly,
    rather than only following the first-order (gradient) direction.
    That quadratic approximation is

        f(x+d) ~ f(x) + grad(x).d + 0.5*d^T*Hess(x)*d,

    minimised over d by setting its gradient (in d) to zero:
    grad(x) + Hess(x)*d = 0, i.e. d = -Hess(x)^{-1}*grad(x), the
    Newton step solved here via linear_algebra.solve_linear_system
    rather than by forming the inverse explicitly (solving is both
    cheaper and better conditioned than inverting).

    Two properties are demonstrated directly in this project rather
    than only cited: (1) for a genuinely quadratic f, the second-order
    Taylor approximation used to derive the step *is* f itself,
    exactly, with no higher-order terms dropped, so Newton's method
    reaches the exact minimiser in exactly one step regardless of
    starting point or the quadratic's conditioning (in sharp contrast
    to gradient descent, whose iteration count for an ill-conditioned
    quadratic grows with the condition number); (2) near the optimum
    of a smooth, strongly convex function, Newton's method converges
    quadratically, error_{k+1} ~ C*error_k^2, versus gradient
    descent's linear convergence, error_{k+1} ~ r*error_k for some
    r<1, both checked empirically in the test suite via the actual
    sequence of errors produced, not merely asserted from theory.
    """
    grad_fn = grad_fn or (lambda x: cf.numerical_gradient(f, x))
    hess_fn = hess_fn or (lambda x: cf.numerical_hessian(f, x))

    x = list(x0)
    history = [f(x)]
    x_history = [list(x)]
    for iteration in range(max_iter):
        grad = grad_fn(x)
        if la.vector_norm(grad) < tol:
            return {"x": x, "f": f(x), "iterations": iteration, "converged": True,
                    "history": history, "x_history": x_history}
        hess = hess_fn(x)
        step = la.solve_linear_system(hess, [-g for g in grad])
        x = la.vec_add(x, step)
        history.append(f(x))
        x_history.append(list(x))
    return {"x": x, "f": f(x), "iterations": max_iter, "converged": False,
            "history": history, "x_history": x_history}
