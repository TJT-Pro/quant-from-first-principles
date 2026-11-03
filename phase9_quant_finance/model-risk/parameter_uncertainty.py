import math
import random
import statistics
from typing import List, Callable, Dict


def bootstrap_resample(data: List, seed: int) -> List:
    """A single bootstrap resample: draw len(data) observations from
    data with replacement. This is the standard nonparametric bootstrap
    (Efron 1979), which approximates the sampling distribution of any
    estimator by treating the observed sample itself as a stand-in for
    the true (unknown) population, resampling from it repeatedly, and
    recomputing the estimator on each resample.
    """
    rng = random.Random(seed)
    n = len(data)
    return [data[rng.randrange(n)] for _ in range(n)]


def bootstrap_parameter_distribution(data: List[float], estimator_fn: Callable[[List[float]], float],
                                      n_bootstrap: int, seed: int = 0) -> List[float]:
    """The bootstrap sampling distribution of a parameter estimator:
    apply estimator_fn to n_bootstrap independent resamples of the
    original data, giving a distribution of estimates that directly
    reflects the estimator's own sampling uncertainty, purely from
    finite-sample noise in the original data, without any assumption
    about the estimator's theoretical distribution.
    """
    estimates = []
    for i in range(n_bootstrap):
        resample = bootstrap_resample(data, seed=seed * 10 ** 6 + i)
        estimates.append(estimator_fn(resample))
    return estimates


def propagate_to_model_output(data: List[float], estimator_fn: Callable[[List[float]], float],
                               output_fn: Callable[[float], float], n_bootstrap: int,
                               seed: int = 0) -> Dict:
    """Propagate parameter estimation uncertainty all the way through
    to a downstream model output: for each bootstrap resample,
    re-estimate the parameter (e.g. volatility) and recompute the
    model's output (e.g. an option price) using that specific
    re-estimated parameter, giving a distribution of outputs driven
    entirely by uncertainty in the input estimate. This is a genuinely
    distinct and often underappreciated form of model risk: even when
    the model's functional form is exactly correct (Black-Scholes
    really is the right pricing formula for the assumed dynamics), the
    parameters it is fed are themselves estimated from a finite
    historical sample and carry real uncertainty that should propagate
    into any price or decision based on the model's output, not be
    treated as if the estimated parameter were the true one known with
    certainty.
    """
    param_distribution = bootstrap_parameter_distribution(data, estimator_fn, n_bootstrap, seed)
    output_distribution = [output_fn(p) for p in param_distribution]

    return {
        "parameter_estimate": estimator_fn(data),
        "parameter_std": statistics.pstdev(param_distribution),
        "parameter_5th_percentile": sorted(param_distribution)[int(0.05 * n_bootstrap)],
        "parameter_95th_percentile": sorted(param_distribution)[int(0.95 * n_bootstrap)],
        "output_point_estimate": output_fn(estimator_fn(data)),
        "output_mean": statistics.mean(output_distribution),
        "output_std": statistics.pstdev(output_distribution),
        "output_5th_percentile": sorted(output_distribution)[int(0.05 * n_bootstrap)],
        "output_95th_percentile": sorted(output_distribution)[int(0.95 * n_bootstrap)],
        "output_distribution": output_distribution,
    }


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_call(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """A fresh, from-scratch closed-form European call price, used
    purely as a representative "model output" whose sensitivity to its
    own volatility input this module demonstrates; sharing no code
    with any sibling repository's own closed-form module.

    As sigma -> 0, the terminal price S_T = S_0*exp((r-0.5*sigma^2)*T)
    becomes deterministic, converging to the risk-free forward price
    S_0*exp(r*T), so the discounted call payoff converges to
    exp(-r*T)*max(S_0*exp(r*T)-K,0) = max(S_0-K*exp(-r*T),0): the
    discounted-forward intrinsic value, not the plain (undiscounted)
    intrinsic value max(S_0-K,0). An earlier version of this function
    used the undiscounted shortcut for the sigma<=0 special case and
    was caught by its own test: at S_0=110, K=100, r=0.05, T=1, it
    returned 10.0 (the undiscounted intrinsic value) against the
    correct 14.877 (S_0-K*exp(-r*T)), a difference of nearly 4.9, far
    too large to be rounding.
    """
    if sigma <= 0 or t <= 0:
        return max(s0 - k * math.exp(-r * t), 0.0)
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return s0 * norm_cdf(d1) - k * math.exp(-r * t) * norm_cdf(d2)


def annualized_volatility_estimator(returns: List[float], periods_per_year: int = 252) -> float:
    return statistics.pstdev(returns) * math.sqrt(periods_per_year)
