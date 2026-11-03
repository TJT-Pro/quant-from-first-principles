import math
import random
import statistics
from typing import List, Optional, Dict

from optimization import nelder_mead
from ar_models import is_stationary
from ma_models import is_invertible


def simulate_arma(n: int, c: float, phis: List[float], thetas: List[float],
                   sigma: float = 1.0, seed: Optional[int] = None, burn_in: int = 200) -> List[float]:
    """Simulate an ARMA(p,q) process:

        X_t = c + sum_i phi_i * X_{t-i} + eps_t + sum_j theta_j * eps_{t-j}.

    This is the natural combination of `ar_models.simulate_ar` and
    `ma_models.simulate_ma`: at each step, both the autoregressive
    feedback from past X values and the moving-average feedback from
    past innovations contribute to the current value. A burn-in period
    is discarded for the same reason as in the pure-AR case: the
    process needs time to forget its arbitrary zero starting condition
    and settle near its stationary distribution (assuming the AR part
    is actually stationary; see `is_stationary`).
    """
    rng = random.Random(seed)
    p, q = len(phis), len(thetas)
    x = [0.0] * p
    eps_history = [0.0] * q
    for _ in range(n + burn_in):
        e = rng.gauss(0, sigma)
        ar_part = sum(phis[i] * x[-(i + 1)] for i in range(p)) if p > 0 else 0.0
        ma_part = sum(thetas[j] * eps_history[-(j + 1)] for j in range(q)) if q > 0 else 0.0
        value = c + ar_part + e + ma_part
        x.append(value)
        eps_history.append(e)
    return x[p + burn_in:]


def css_residuals(x: List[float], c: float, phis: List[float], thetas: List[float]) -> List[float]:
    """Conditional-sum-of-squares residuals for an ARMA(p,q) model:
    eps_t = x_t - c - sum_i phi_i*x_{t-i} - sum_j theta_j*eps_{t-j},
    conditioning on the first p observations of x and the pre-sample
    innovations both being fixed (the p observed values are used
    as-is; the missing pre-sample innovations are set to zero), the
    direct ARMA generalisation of `ma_models.css_residuals`.
    """
    p, q = len(phis), len(thetas)
    n = len(x)
    if n <= p:
        raise ValueError("series too short for the requested AR order")
    eps = [0.0] * q
    residuals = []
    for t in range(p, n):
        ar_part = sum(phis[i] * x[t - 1 - i] for i in range(p)) if p > 0 else 0.0
        ma_part = sum(thetas[j] * eps[-(j + 1)] for j in range(q)) if q > 0 else 0.0
        e = x[t] - c - ar_part - ma_part
        eps.append(e)
        residuals.append(e)
    return residuals


def _css_loss(x: List[float], params: List[float], p: int, q: int) -> float:
    """Sum of squared CSS residuals, guarded against numerical
    overflow. Nelder-Mead has no notion of "this region of parameter
    space is invalid" built in: its reflection and expansion steps can
    and do propose AR coefficients whose implied dynamics are
    explosive (a transient step outside the stationary region, even if
    the search eventually settles inside it), which makes the
    recursive residual computation blow up to float overflow. Rather
    than let that propagate as an uncaught exception and crash the
    whole fit (which is what happened during development, tracked down
    to exactly this cause), an explosive candidate point is scored as
    positive infinity: Nelder-Mead's ordering-based logic (sort by
    score, reflect away from the worst) handles an infinite score
    correctly with no special-casing needed elsewhere, and the search
    simply treats that point as arbitrarily bad and moves away from it.
    """
    c = params[0]
    phis = params[1:1 + p]
    thetas = params[1 + p:1 + p + q]
    try:
        residuals = css_residuals(x, c, phis, thetas)
        return sum(e ** 2 for e in residuals)
    except OverflowError:
        return float("inf")


def fit_css(x: List[float], p: int, q: int,
            initial_phis: Optional[List[float]] = None,
            initial_thetas: Optional[List[float]] = None) -> Dict[str, float]:
    """Fit an ARMA(p,q) model by minimising the conditional sum of
    squared residuals via Nelder-Mead, generalising `ma_models.fit_css`
    to include autoregressive terms. As with the pure-MA case, no
    closed form exists once moving-average terms are present, since
    the residuals depend nonlinearly on the theta parameters through
    the recursion.
    """
    if initial_phis is None:
        initial_phis = [0.1] * p
    if initial_thetas is None:
        initial_thetas = [0.1] * q
    x0 = [statistics.mean(x)] + list(initial_phis) + list(initial_thetas)
    loss = lambda params: _css_loss(x, params, p, q)
    result = nelder_mead(loss, x0, step=0.1, max_iter=4000)

    c = result["x"][0]
    phis = result["x"][1:1 + p]
    thetas = result["x"][1 + p:1 + p + q]
    n = len(x)
    sigma2 = result["fun"] / n
    return {
        "c": c, "phis": phis, "thetas": thetas, "sigma2": sigma2,
        "sse": result["fun"], "iterations": result["iterations"],
        "stationary": is_stationary(phis) if phis else True,
        "invertible": is_invertible(thetas) if thetas else True,
    }


def log_likelihood_gaussian(residuals: List[float], sigma2: float) -> float:
    """Conditional Gaussian log-likelihood of a set of residuals given
    their variance:

        ln L = -(n/2) * ln(2*pi*sigma2) - SSE / (2*sigma2)

    the sum of n independent N(0, sigma2) log-densities. Used by
    `model_selection.py` to compute AIC/BIC, since both criteria are
    defined in terms of the maximised log-likelihood, not the raw sum
    of squared residuals directly.
    """
    n = len(residuals)
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    sse = sum(e ** 2 for e in residuals)
    return -0.5 * n * math.log(2 * math.pi * sigma2) - sse / (2 * sigma2)
