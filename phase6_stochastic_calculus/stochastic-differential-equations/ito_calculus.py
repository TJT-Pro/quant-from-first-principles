import math
import statistics
from typing import Callable

import sde_solvers as sde


def apply_ito_lemma(f: Callable[[float], float], f_x: Callable[[float], float],
                     f_xx: Callable[[float], float], mu: Callable[[float, float], float],
                     sigma: Callable[[float, float], float]):
    """Given an SDE dX_t = mu(X_t,t)dt + sigma(X_t,t)dW_t and a
    twice-differentiable transform f, Ito's lemma gives the SDE
    satisfied by Y_t = f(X_t):

        dY_t = [f'(X_t)*mu(X_t,t) + 0.5*f''(X_t)*sigma(X_t,t)^2] dt
               + f'(X_t)*sigma(X_t,t) dW_t

    The extra 0.5*f''*sigma^2 term, absent from an ordinary
    (deterministic) chain rule, comes directly from Brownian motion's
    quadratic variation: expanding f(X_t+dX_t) to second order in dX_t
    and substituting dX_t = mu*dt + sigma*dW_t, the (dW_t)^2 term
    inside (dX_t)^2 does not vanish at order dt the way it would for
    an ordinary differentiable increment (see brownian_motion.py's
    quadratic-variation discussion), leaving a genuine dt-order
    contribution once (dW_t)^2 is replaced by its expected value dt.
    Returns the transformed process's own (mu_Y, sigma_Y) pair,
    expressed as functions of the original state x (since Y is a
    function of X, its own drift and diffusion are most naturally
    written in terms of X, not by inverting f to get Y's own state).
    These are intended for direct inspection (e.g. printing the
    resulting formula, as in `gbm_log_transform_check` below, where
    they turn out to be constants) or for evaluation at a known,
    externally-tracked X_t; they are NOT safe to hand to
    `sde_solvers.euler_maruyama` as if they were Y's own autonomous
    (mu, sigma) pair, since that would evaluate them at Y's current
    value rather than X's, silently wrong except in the special case
    (such as the log-transform of GBM) where the result happens not to
    depend on x at all. `verify_ito_transform_matches_direct_simulation`
    below tracks X_t explicitly for exactly this reason.
    """
    mu_y = lambda x, t: f_x(x) * mu(x, t) + 0.5 * f_xx(x) * sigma(x, t) ** 2
    sigma_y = lambda x, t: f_x(x) * sigma(x, t)
    return mu_y, sigma_y


def verify_ito_transform_matches_direct_simulation(f: Callable[[float], float], f_x: Callable[[float], float],
                                                     f_xx: Callable[[float], float],
                                                     mu: Callable[[float, float], float],
                                                     sigma: Callable[[float, float], float],
                                                     x0: float, t_end: float, n_steps: int,
                                                     n_paths: int, seed: int = 0) -> dict:
    """The central check: simulate X_t directly via Euler-Maruyama and
    apply f to its terminal value, versus separately accumulating
    Y_t = f(X_t) step by step using Ito's lemma's own dY formula,
    evaluated at the *current* X_t at every step (using the same
    underlying Brownian increments for both). If Ito's lemma is
    correctly derived and applied, both routes should agree up to
    ordinary Euler-Maruyama discretization error common to both.

    Note on a design choice this function deliberately avoids: an
    earlier version of this function instead derived Y's own
    (mu_Y, sigma_Y) pair via `apply_ito_lemma` and simulated Y as an
    independent, autonomous process via a *second* call to
    `sde_solvers.euler_maruyama`. That is wrong in general, because
    `apply_ito_lemma`'s returned functions are written in terms of the
    *original* state x, but simulating Y autonomously calls them with
    Y's own current value instead (there being no separate X_t
    available once Y is being simulated on its own) -- correct only by
    coincidence when the transformed coefficients happen not to depend
    on x at all, which happens to be true for the GBM-log-transform
    example below but is not true in general. This function instead
    tracks X_t explicitly and evaluates every Ito-lemma coefficient at
    X_t's own current value at each step, which is correct for any
    twice-differentiable f, invertible or not.
    """
    f_x_over_x = f_x  # kept as a distinct name for clarity at call sites
    y0 = f(x0)

    direct_diffs = []
    for i in range(n_paths):
        dw = sde.generate_fine_brownian_increments(t_end, n_steps, seed=seed * 10 ** 6 + i)
        dt = t_end / n_steps

        x_result = sde.euler_maruyama(mu, sigma, x0, 0.0, t_end, dw)
        y_from_x = f(x_result["x"][-1])

        x = x0
        y = y0
        t = 0.0
        for k in range(n_steps):
            m = mu(x, t)
            s = sigma(x, t)
            dy = (f_x_over_x(x) * m + 0.5 * f_xx(x) * s ** 2) * dt + f_x_over_x(x) * s * dw[k]
            x = x + m * dt + s * dw[k]
            y = y + dy
            t += dt

        direct_diffs.append(abs(y_from_x - y))

    return {"mean_absolute_difference": statistics.mean(direct_diffs), "max_difference": max(direct_diffs)}


def gbm_log_transform_check(mu_param: float, sigma_param: float, x0: float, t_end: float,
                             n_steps: int, seed: int = 0) -> dict:
    """A fully worked, closed-form-checkable example: for GBM
    dX_t = mu*X_t*dt + sigma*X_t*dW_t and f(x)=ln(x), Ito's lemma gives

        d(ln X_t) = (mu - 0.5*sigma^2) dt + sigma dW_t,

    directly from f'(x)=1/x, f''(x)=-1/x^2 substituted into the
    general formula above: f'*mu(x) = mu*x/x = mu, and
    0.5*f''*sigma(x)^2 = 0.5*(-1/x^2)*(sigma*x)^2 = -0.5*sigma^2,
    giving the drift mu-0.5*sigma^2 with NO remaining x-dependence at
    all: ln(X_t) is a drifted Brownian motion outright, not merely
    another, more complicated diffusion. This closed-form drift and
    diffusion is checked directly against a single, explicit Brownian
    path's realised ln(X_T) - ln(X_0), which should equal
    (mu-0.5*sigma^2)*T + sigma*W_T exactly (this being the exact GBM
    solution itself), not merely approximately.
    """
    dw = sde.generate_fine_brownian_increments(t_end, n_steps, seed=seed)
    w_t = sum(dw)
    exact_log_change = (mu_param - 0.5 * sigma_param ** 2) * t_end + sigma_param * w_t

    x_t = x0 * math.exp(exact_log_change)
    simulated_log_change = math.log(x_t) - math.log(x0)

    return {"exact_log_change": exact_log_change, "simulated_log_change": simulated_log_change,
            "difference": abs(exact_log_change - simulated_log_change)}
