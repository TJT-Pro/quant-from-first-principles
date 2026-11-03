import math
import random
from typing import List, Dict

import linear_algebra as la


def kelly_fraction_single_asset(mu: float, sigma: float) -> float:
    """The growth-optimal (Kelly) leverage for a single risky asset
    with per-period mean return mu and standard deviation sigma:
    f* = mu / sigma^2. This maximises the expected long-run compound
    growth rate E[log(1+f*R)], not the expected return itself: for
    small per-period returns R, a second-order Taylor expansion gives
    log(1+f*R) ~ f*R - 0.5*f^2*R^2, so
    E[log(1+f*R)] ~ f*mu - 0.5*f^2*sigma^2 (using E[R^2]~sigma^2 when
    mu is small relative to sigma, the usual regime for financial
    returns), a simple downward parabola in f maximised exactly at
    f* = mu/sigma^2.
    """
    return mu / sigma ** 2


def expected_log_growth_rate(f: float, mu: float, sigma: float) -> float:
    """The quadratic approximation to expected log growth,
    f*mu - 0.5*f^2*sigma^2, used both to derive the Kelly fraction
    above and to demonstrate directly (via its own curvature) why
    over-betting is punished more severely than under-betting: this is
    a downward parabola in f, symmetric around its own vertex at
    f*=mu/sigma^2, but the realised growth rate it approximates
    degrades in a genuinely asymmetric way once leverage is large
    enough that the small-return approximation itself breaks down
    (checked directly in fx_application.py via full-precision log
    growth, not just this quadratic proxy).
    """
    return f * mu - 0.5 * f ** 2 * sigma ** 2


def kelly_weights_multi_asset(mu: List[float], sigma: List[List[float]]) -> List[float]:
    """The multi-asset growth-optimal leverage vector: maximising the
    same quadratic approximation to expected log growth,
    w^T*mu - 0.5*w^T*Sigma*w, over unconstrained w (no budget
    constraint at all, since these weights represent leverage
    fractions applied to the underlying capital, not a normalised
    allocation of a fixed budget) gives, by setting the gradient
    mu - Sigma*w to zero, w* = Sigma^{-1}*mu directly: the exact
    multivariate generalisation of the single-asset mu/sigma^2
    formula, using Sigma^{-1} in place of division by the scalar
    sigma^2.

    This is, remarkably, the same direction as the numerator of the
    tangency portfolio (see tangency_portfolio.py) when the risk-free
    rate is zero: w_tan is proportional to Sigma^{-1}*(mu-0*1) =
    Sigma^{-1}*mu, identical up to the normalisation that forces
    tangency weights to sum to 1. Growth-optimal betting theory and
    mean-variance tangency-portfolio theory, developed from entirely
    different first principles (maximising expected log wealth versus
    maximising the Sharpe ratio), arrive at the same optimal risky-
    asset direction; only the overall scale differs, since Kelly
    weights are leverage fractions that need not sum to any particular
    value, whereas tangency weights are normalised portfolio shares.
    """
    sigma_inv = la.invert(sigma)
    return la.mat_vec(sigma_inv, mu)


def simulate_compound_growth(f: float, mu: float, sigma: float, n_periods: int, n_paths: int,
                              seed: int = 0) -> float:
    """A direct Monte Carlo cross-check of the analytic Kelly formula,
    with no small-return quadratic approximation involved at all:
    simulate n_paths independent sequences of n_periods returns
    (Gaussian, mean mu, standard deviation sigma), compound wealth
    period by period under leverage f (wealth_{t+1} =
    wealth_t*(1+f*R_t)), and return the average realised per-period
    log growth rate across paths, log(final_wealth)/n_periods averaged
    across paths. Because this simulates full compounding exactly
    rather than relying on the quadratic Taylor approximation
    expected_log_growth_rate uses, it is a genuine, independent check
    of whether the analytically derived f* actually maximises realised
    long-run growth, not merely the quadratic proxy for it.
    """
    rng = random.Random(seed)
    total_log_growth = 0.0
    for _ in range(n_paths):
        log_wealth = 0.0
        for _ in range(n_periods):
            r = rng.gauss(mu, sigma)
            growth_factor = 1 + f * r
            if growth_factor <= 0:
                log_wealth = float("-inf")
                break
            log_wealth += math.log(growth_factor)
        total_log_growth += log_wealth / n_periods
    return total_log_growth / n_paths


def fractional_kelly_comparison(f_star: float, mu: float, sigma: float, n_periods: int,
                                 n_paths: int, fractions: List[float], seed: int = 0) -> Dict:
    """Compare realised growth rate and terminal wealth volatility
    across several fractions of full Kelly leverage: full Kelly (f=1x
    f_star) maximises expected log growth, but practitioners commonly
    use a fraction of it (e.g. "half Kelly") specifically to reduce
    the volatility of the wealth path, since the growth-rate function
    is quite flat near its own maximum (a direct consequence of it
    being a smooth parabola, whose derivative vanishes at the peak)
    while the variance of terminal wealth continues to grow with
    leverage throughout, giving a much better volatility-reduction-
    per-unit-of-growth-sacrificed trade-off near full Kelly than one
    might expect from the growth curve's shape alone.
    """
    results = []
    for frac in fractions:
        f = frac * f_star
        rng = random.Random(seed)
        terminal_log_wealths = []
        for _ in range(n_paths):
            log_wealth = 0.0
            for _ in range(n_periods):
                r = rng.gauss(mu, sigma)
                growth_factor = 1 + f * r
                if growth_factor <= 0:
                    log_wealth = float("-inf")
                    break
                log_wealth += math.log(growth_factor)
            terminal_log_wealths.append(log_wealth)
        finite = [w for w in terminal_log_wealths if w != float("-inf")]
        mean_log_wealth = sum(finite) / len(finite) if finite else float("-inf")
        variance = (sum((w - mean_log_wealth) ** 2 for w in finite) / len(finite)) if len(finite) > 1 else 0.0
        results.append({"fraction_of_kelly": frac, "leverage": f,
                         "mean_log_growth_per_period": mean_log_wealth / n_periods,
                         "std_terminal_log_wealth": math.sqrt(variance),
                         "fraction_ruined": 1 - len(finite) / n_paths})
    return {"f_star": f_star, "results": results}
