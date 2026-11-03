import random
import statistics
from typing import List, Optional, Dict

from ar_models import durand_kerner_roots
from optimization import nelder_mead


def simulate_ma(n: int, mu: float, thetas: List[float], sigma: float = 1.0,
                 seed: Optional[int] = None) -> List[float]:
    """Simulate an MA(q) process X_t = mu + eps_t + sum_i theta_i * eps_{t-i}.
    Unlike an AR process, an MA process is automatically stationary
    for any finite set of theta coefficients (it is a finite linear
    combination of white noise terms, which always has finite,
    time-invariant variance), so no burn-in period is needed.
    """
    rng = random.Random(seed)
    q = len(thetas)
    eps = [rng.gauss(0, sigma) for _ in range(n + q)]
    x = []
    for t in range(q, n + q):
        value = mu + eps[t] + sum(thetas[i] * eps[t - 1 - i] for i in range(q))
        x.append(value)
    return x


def is_invertible(thetas: List[float]) -> bool:
    """An MA(q) process is invertible (representable as an infinite-
    order AR process, which is what makes maximum-likelihood
    estimation via the recursive residual construction below
    well-defined and unique) iff every root of
    1 + theta_1*z + ... + theta_q*z^q lies strictly outside the unit
    circle, the direct MA analogue of the AR stationarity condition,
    checked with the same Durand-Kerner root finder.
    """
    if not thetas:
        return True
    if len(thetas) == 1:
        return abs(thetas[0]) < 1.0
    coeffs = [1.0] + list(thetas)
    roots = durand_kerner_roots(coeffs)
    return all(abs(r) > 1.0 for r in roots)


def css_residuals(x: List[float], mu: float, thetas: List[float]) -> List[float]:
    """Conditional-sum-of-squares residuals: eps_t = x_t - mu -
    sum_i theta_i * eps_{t-i}, computed recursively forward in time
    with the pre-sample innovations eps_0, eps_{-1}, ... set to zero
    (the "conditional" in conditional least squares/maximum
    likelihood: the exact likelihood would instead integrate over the
    unknown pre-sample values' own distribution, which requires the
    more involved innovations algorithm; conditioning on zero is the
    standard, simpler approximation that becomes asymptotically
    negligible as the sample grows).
    """
    q = len(thetas)
    n = len(x)
    eps = [0.0] * q
    for t in range(n):
        predicted = mu + sum(thetas[i] * eps[-(i + 1)] for i in range(q))
        eps.append(x[t] - predicted)
    return eps[q:]


def _css_loss(x: List[float], params: List[float], q: int) -> float:
    """See `arma_models._css_loss` for why overflow is caught and
    penalised with infinity rather than allowed to crash the fit.
    """
    mu = params[0]
    thetas = params[1:1 + q]
    try:
        residuals = css_residuals(x, mu, thetas)
        return sum(e ** 2 for e in residuals)
    except OverflowError:
        return float("inf")


def fit_css(x: List[float], q: int, initial_thetas: Optional[List[float]] = None) -> Dict[str, float]:
    """Fit an MA(q) model by minimising the conditional sum of squared
    residuals via Nelder-Mead (no closed-form solution exists for MA
    parameters the way Yule-Walker gives one for AR parameters,
    because the residuals depend on the parameters recursively and
    nonlinearly rather than linearly). The initial guess defaults to
    small values near zero (a mild, generic starting point), since an
    MA(q) fit is generally far more sensitive to starting values than
    an AR fit is.
    """
    if initial_thetas is None:
        initial_thetas = [0.1] * q
    x0 = [statistics.mean(x)] + list(initial_thetas)
    loss = lambda params: _css_loss(x, params, q)
    result = nelder_mead(loss, x0, step=0.1, max_iter=3000)
    mu = result["x"][0]
    thetas = result["x"][1:1 + q]
    n = len(x)
    sigma2 = result["fun"] / n
    return {"mu": mu, "thetas": thetas, "sigma2": sigma2, "sse": result["fun"], "iterations": result["iterations"]}
