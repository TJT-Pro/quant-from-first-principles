import math
import random
import statistics
from typing import Callable, List

import low_discrepancy as ld


def monte_carlo_integrate(f: Callable[[List[float]], float], dim: int, n_points: int,
                           seed: int = 0) -> float:
    """Estimate the integral of f over [0,1]^dim by plain Monte Carlo:
    average f at n_points independent, uniformly random points. This
    is an unbiased estimator (E[average] equals the true integral
    exactly, since each sampled point is uniformly distributed over
    the domain), with standard error scaling as O(1/sqrt(n_points))
    regardless of dimension, by the ordinary central limit theorem
    applied to the sequence of i.i.d. values f(x_1), f(x_2), ....
    """
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n_points):
        point = [rng.random() for _ in range(dim)]
        total += f(point)
    return total / n_points


def qmc_integrate(f: Callable[[List[float]], float], dim: int, n_points: int, skip: int = 0) -> float:
    """Estimate the same integral using a Halton low-discrepancy
    sequence instead of independent random points: average f across
    the first n_points Halton points. Because a low-discrepancy
    sequence fills the domain far more evenly than random sampling
    does (see discrepancy.py for a direct, quantitative comparison),
    the Koksma-Hlawka inequality bounds this estimator's error by
    (the function's total variation) times (the point set's star
    discrepancy), which for a low-discrepancy sequence decays roughly
    like (log n_points)^dim / n_points, asymptotically much faster
    than random sampling's O(1/sqrt(n_points)) for smooth, low-
    dimensional integrands, at the cost of the estimator no longer
    being unbiased in the same probabilistic sense (a single
    deterministic point set either is or is not close to the truth,
    with no independent-trials averaging argument the way random
    sampling has).
    """
    points = ld.halton_sequence(n_points, dim, skip=skip)
    total = sum(f(p) for p in points)
    return total / n_points


def estimate_convergence_order(errors: List[float], n_values: List[int]) -> float:
    """Fit log(error) = log(C) - p*log(n) by ordinary least squares
    across several sample sizes, and return the fitted slope p as the
    empirically observed convergence order (using -log(n) rather than
    log(h) as in this project's own PDE/ODE/SDE repositories, since
    here the natural axis is the number of points n rather than a
    step size h, with error expected to shrink, not grow, as n
    increases).
    """
    log_n = [math.log(n) for n in n_values]
    log_err = [math.log(e) for e in errors]
    mean_x = statistics.mean(log_n)
    mean_y = statistics.mean(log_err)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_n, log_err))
    den = sum((x - mean_x) ** 2 for x in log_n)
    slope = num / den
    return -slope


def monte_carlo_repeated_error(f: Callable[[List[float]], float], dim: int, n_points: int,
                                exact_value: float, n_repeats: int, seed: int = 0) -> float:
    """The root-mean-square error of Monte Carlo integration at a
    given n_points, averaged over n_repeats independent runs (each
    with its own random points): needed because a single Monte Carlo
    run's error is itself a random variable, so any single run's error
    could over- or under-state the estimator's typical accuracy purely
    by chance; averaging over many independent repeats gives a stable
    estimate of the error Monte Carlo integration actually achieves
    at that sample size, suitable for a convergence-order regression.
    QMC needs no such averaging, since a Halton sequence is fully
    deterministic and gives the identical estimate on every run at a
    given n_points.
    """
    rng = random.Random(seed)
    squared_errors = []
    for _ in range(n_repeats):
        run_seed = rng.randint(0, 2 ** 31)
        estimate = monte_carlo_integrate(f, dim, n_points, seed=run_seed)
        squared_errors.append((estimate - exact_value) ** 2)
    return math.sqrt(statistics.mean(squared_errors))
