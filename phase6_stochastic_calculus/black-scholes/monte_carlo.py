import math
import random
import statistics
from typing import Callable, Dict, List, Optional


def simulate_terminal_prices(s: float, t: float, r: float, sigma: float, n_paths: int,
                              q: float = 0.0, seed: Optional[int] = None,
                              antithetic: bool = False) -> List[float]:
    """Simulate S_T under the risk-neutral measure (drift r-q instead
    of the real-world mu, since option pricing discounts a
    risk-neutral expectation) using the exact lognormal solution of
    GBM rather than stepping through intermediate time points, since
    only the terminal value is needed for a European payoff.

    With `antithetic=True`, each standard normal draw Z is paired with
    its mirror image -Z, generating two paths per random draw. Because
    S_T is a monotone function of Z, pairing positive and negative
    shocks induces negative correlation between the two paths' payoffs
    for a monotone payoff function, which reduces the variance of the
    *average* of the pair below what two independent draws would give,
    without introducing any bias (the antithetic pair's average is
    still an unbiased estimator of E[S_T], since Z and -Z have the
    same distribution).
    """
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    rng = random.Random(seed)
    drift = (r - q - 0.5 * sigma ** 2) * t
    diffusion_scale = sigma * math.sqrt(t)

    prices = []
    if antithetic:
        half = (n_paths + 1) // 2
        for _ in range(half):
            z = rng.gauss(0.0, 1.0)
            prices.append(s * math.exp(drift + diffusion_scale * z))
            prices.append(s * math.exp(drift - diffusion_scale * z))
        prices = prices[:n_paths]
    else:
        for _ in range(n_paths):
            z = rng.gauss(0.0, 1.0)
            prices.append(s * math.exp(drift + diffusion_scale * z))
    return prices


def monte_carlo_price(s: float, k: float, t: float, r: float, sigma: float, n_paths: int,
                       option_type: str = "call", q: float = 0.0,
                       seed: Optional[int] = None, antithetic: bool = False) -> Dict[str, float]:
    """Discounted average payoff over simulated terminal prices, with a
    95% confidence interval from the sample standard error. The
    standard error shrinks as 1/sqrt(n_paths), the defining (and
    comparatively slow) convergence rate of plain Monte Carlo, which
    is exactly what `monte_carlo_convergence_series` demonstrates
    directly rather than only asserting.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    terminal_prices = simulate_terminal_prices(s, t, r, sigma, n_paths, q, seed, antithetic)
    if option_type == "call":
        payoffs = [max(p - k, 0.0) for p in terminal_prices]
    else:
        payoffs = [max(k - p, 0.0) for p in terminal_prices]

    discount = math.exp(-r * t)
    discounted_payoffs = [discount * p for p in payoffs]
    mean_price = statistics.mean(discounted_payoffs)
    std_error = statistics.stdev(discounted_payoffs) / math.sqrt(len(discounted_payoffs)) if len(discounted_payoffs) > 1 else float("nan")

    return {
        "price": mean_price,
        "std_error": std_error,
        "ci_low": mean_price - 1.96 * std_error,
        "ci_high": mean_price + 1.96 * std_error,
        "n_paths": len(discounted_payoffs),
    }


def monte_carlo_convergence_series(s: float, k: float, t: float, r: float, sigma: float,
                                    option_type: str, path_counts, q: float = 0.0,
                                    seed: Optional[int] = None, antithetic: bool = False) -> List[Dict[str, float]]:
    """Monte Carlo price and standard error at each path count in
    `path_counts`, all drawn from the same underlying random stream
    (by reusing a single seeded generator across increasing path
    counts) so that the sequence of estimates is a genuine refinement
    rather than independent unrelated runs.
    """
    results = []
    rng = random.Random(seed)
    drift = (r - q - 0.5 * sigma ** 2) * t
    diffusion_scale = sigma * math.sqrt(t)
    all_prices: List[float] = []

    previous_count = 0
    for count in path_counts:
        n_new = count - previous_count
        if antithetic:
            half = (n_new + 1) // 2
            for _ in range(half):
                z = rng.gauss(0.0, 1.0)
                all_prices.append(s * math.exp(drift + diffusion_scale * z))
                all_prices.append(s * math.exp(drift - diffusion_scale * z))
        else:
            for _ in range(n_new):
                z = rng.gauss(0.0, 1.0)
                all_prices.append(s * math.exp(drift + diffusion_scale * z))
        previous_count = count

        current_prices = all_prices[:count]
        if option_type == "call":
            payoffs = [max(p - k, 0.0) for p in current_prices]
        else:
            payoffs = [max(k - p, 0.0) for p in current_prices]
        discount = math.exp(-r * t)
        discounted = [discount * p for p in payoffs]
        mean_price = statistics.mean(discounted)
        std_error = statistics.stdev(discounted) / math.sqrt(len(discounted)) if len(discounted) > 1 else float("nan")
        results.append({"n_paths": count, "price": mean_price, "std_error": std_error})
    return results
