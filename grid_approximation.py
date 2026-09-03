import math
from typing import Callable, List, Tuple


def log_sum_exp(values: List[float]) -> float:
    """ln(sum(exp(v) for v in values)), computed by factoring out the
    largest value before exponentiating the rest:

        ln(sum exp(v_i)) = v_max + ln(sum exp(v_i - v_max)).

    Every v_i - v_max is now <= 0, so exp() of it never overflows,
    which a direct sum of exp(v_i) would risk doing whenever any log-
    density in this project (a log-posterior evaluated over a fine
    grid, in particular) is large in magnitude.
    """
    if not values:
        raise ValueError("values must be non-empty")
    v_max = max(values)
    if v_max == float("-inf"):
        return float("-inf")
    return v_max + math.log(sum(math.exp(v - v_max) for v in values))


def grid_posterior(log_prior: Callable[[float], float], log_likelihood: Callable[[float], float],
                    grid: List[float]) -> List[float]:
    """A normalised posterior over `grid`, by brute-force evaluation:
    the unnormalised log-posterior at each grid point is
    log_prior(theta) + log_likelihood(theta) (Bayes' theorem in log
    form, where the two terms simply add instead of multiply), and
    normalising divides by the total probability mass summed
    (approximately integrated) across the grid. This needs no
    conjugacy and no assumption about the posterior's shape at all,
    at the cost of scaling exponentially poorly to more than a
    handful of parameters, which is exactly the problem Metropolis-
    Hastings exists to avoid; grid approximation is used in this
    project both as a method in its own right for single-parameter
    problems and as an independent ground truth to check the MCMC
    sampler against on those same problems.
    """
    log_unnormalized = [log_prior(theta) + log_likelihood(theta) for theta in grid]
    log_total = log_sum_exp(log_unnormalized)
    return [math.exp(lu - log_total) for lu in log_unnormalized]


def posterior_mean(grid: List[float], probs: List[float]) -> float:
    return sum(theta * p for theta, p in zip(grid, probs))


def posterior_variance(grid: List[float], probs: List[float]) -> float:
    mean = posterior_mean(grid, probs)
    return sum(p * (theta - mean) ** 2 for theta, p in zip(grid, probs))


def credible_interval(grid: List[float], probs: List[float], level: float = 0.95) -> Tuple[float, float]:
    """An equal-tailed credible interval read directly off the
    discretised cumulative distribution built from the grid: find the
    grid points where the running cumulative probability first crosses
    the lower and upper tail thresholds.
    """
    tail = (1 - level) / 2
    cumulative = 0.0
    lower = grid[0]
    for theta, p in zip(grid, probs):
        cumulative += p
        if cumulative >= tail:
            lower = theta
            break
    cumulative = 0.0
    upper = grid[-1]
    for theta, p in zip(reversed(grid), reversed(probs)):
        cumulative += p
        if cumulative >= tail:
            upper = theta
            break
    return lower, upper
