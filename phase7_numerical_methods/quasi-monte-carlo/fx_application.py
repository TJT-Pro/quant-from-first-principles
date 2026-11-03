import csv
import math
import random
import statistics
from typing import List, Dict

import low_discrepancy as ld
import gaussian_transform as gt
import integration as intg


def load_prices_and_returns(csv_path: str) -> Dict[str, List[float]]:
    prices, returns = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prices.append(float(row["price"]))
            returns.append(float(row["log_return"]))
    return {"prices": prices, "returns": returns}


def calibrate_volatility(returns: List[float], periods_per_year: int = 252) -> float:
    return statistics.pstdev(returns) * math.sqrt(periods_per_year)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_closed_form(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """The closed-form European call price, reimplemented fresh here
    (sharing no code with the sibling black-scholes or numerical-pde
    repositories' own closed-form modules) purely as ground truth for
    the Monte Carlo and Quasi-Monte Carlo estimators below.
    """
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return s0 * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)


def discounted_call_payoff(z: float, s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """The discounted payoff of a European call given a single
    standard normal draw z, under the risk-neutral GBM terminal
    distribution S_T = S_0*exp((r-0.5*sigma^2)*T + sigma*sqrt(T)*z):
    the function being integrated (against the standard normal
    density, over z) for both the Monte Carlo and QMC estimators
    below.
    """
    s_t = s0 * math.exp((r - 0.5 * sigma ** 2) * t + sigma * math.sqrt(t) * z)
    return math.exp(-r * t) * max(s_t - k, 0.0)


def mc_option_price(s0: float, k: float, r: float, sigma: float, t: float, n_paths: int,
                     seed: int = 0) -> float:
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n_paths):
        z = rng.gauss(0, 1)
        total += discounted_call_payoff(z, s0, k, r, sigma, t)
    return total / n_paths


def qmc_option_price(s0: float, k: float, r: float, sigma: float, t: float, n_paths: int,
                      skip: int = 100) -> float:
    """Price the same option via Quasi-Monte Carlo: transform each
    point of a 1D Halton sequence to a standard normal draw via the
    inverse CDF (gaussian_transform.inverse_normal_cdf), never via
    Box-Muller, for exactly the reason documented in
    gaussian_transform.py. `skip` discards the first several Halton
    points by default, since van der Corput's very first points
    (n=1,2,3,...) are the most extreme (closest to 0 or 1) and
    disproportionately influence a small-sample estimate; skipping
    them is a simple, standard mitigation.
    """
    points = ld.halton_sequence(n_paths, 1, skip=skip)
    total = 0.0
    for p in points:
        z = gt.inverse_normal_cdf(p[0])
        total += discounted_call_payoff(z, s0, k, r, sigma, t)
    return total / n_paths


def convergence_comparison(s0: float, k: float, r: float, sigma: float, t: float,
                            n_values: List[int], n_mc_repeats: int = 20, seed: int = 0) -> Dict:
    """Compare Monte Carlo's root-mean-square error (averaged over
    n_mc_repeats independent runs, since a single MC run's error is
    itself random) against QMC's error (a single deterministic
    estimate, since a Halton sequence gives the identical result every
    time at a given n_paths) across a range of sample sizes, and fit
    the empirical convergence order for each.
    """
    exact = black_scholes_closed_form(s0, k, r, sigma, t)
    rng = random.Random(seed)

    mc_errors = []
    for n in n_values:
        squared_errors = []
        for _ in range(n_mc_repeats):
            run_seed = rng.randint(0, 2 ** 31)
            price = mc_option_price(s0, k, r, sigma, t, n, seed=run_seed)
            squared_errors.append((price - exact) ** 2)
        mc_errors.append(math.sqrt(statistics.mean(squared_errors)))

    qmc_errors = [abs(qmc_option_price(s0, k, r, sigma, t, n) - exact) for n in n_values]

    return {
        "exact_price": exact,
        "n_values": n_values,
        "mc_errors": mc_errors,
        "qmc_errors": qmc_errors,
        "mc_order": intg.estimate_convergence_order(mc_errors, n_values),
        "qmc_order": intg.estimate_convergence_order(qmc_errors, n_values),
    }
