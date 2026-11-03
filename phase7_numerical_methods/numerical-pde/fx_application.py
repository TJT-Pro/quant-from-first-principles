import csv
import math
import statistics
from typing import List, Dict

import black_scholes_pde as bs


def load_prices_and_returns(csv_path: str) -> Dict[str, List[float]]:
    prices, returns = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prices.append(float(row["price"]))
            returns.append(float(row["log_return"]))
    return {"prices": prices, "returns": returns}


def calibrate_volatility(returns: List[float], periods_per_year: int = 252) -> float:
    """The standard realised-volatility estimator: the sample standard
    deviation of daily log returns, annualised by the square root of
    trading days per year (variance scales linearly with time for an
    i.i.d. return process, so standard deviation scales with its
    square root).
    """
    return statistics.pstdev(returns) * math.sqrt(periods_per_year)


def price_option_all_schemes(s0: float, k: float, r: float, sigma: float, t: float,
                              n_space: int = 300, n_time: int = 1000,
                              option_type: str = "call") -> Dict:
    """Price a European option on the given spot via all three PDE
    schemes plus the closed-form formula, using a log-price grid wide
    enough (5 standard deviations of the terminal log-price
    distribution on either side of the strike, plus a small margin) to
    keep the truncated-domain boundary conditions from meaningfully
    contaminating the price near the region of interest.
    """
    x_min = math.log(k) - 5 * sigma * math.sqrt(t) - 0.1
    x_max = math.log(k) + 5 * sigma * math.sqrt(t) + 0.1

    closed_form = bs.black_scholes_closed_form(s0, k, r, sigma, t, option_type)
    explicit = bs.solve_bs_pde_explicit(k, r, sigma, t, x_min, x_max, n_space, n_time, option_type)
    implicit = bs.solve_bs_pde_implicit(k, r, sigma, t, x_min, x_max, n_space, n_time, option_type)
    crank_nicolson = bs.solve_bs_pde_crank_nicolson(k, r, sigma, t, x_min, x_max, n_space, n_time, option_type)

    return {
        "closed_form": closed_form,
        "explicit": bs.price_at_spot(explicit, s0),
        "implicit": bs.price_at_spot(implicit, s0),
        "crank_nicolson": bs.price_at_spot(crank_nicolson, s0),
    }


def pde_convergence_study(s0: float, k: float, r: float, sigma: float, t: float,
                           n_space_values: List[int], n_time: int, option_type: str = "call") -> List[Dict]:
    """Measure how each scheme's error against the closed-form price
    shrinks as the spatial grid is refined (holding the time grid
    fixed), the natural convergence axis for comparing schemes whose
    known theoretical accuracy is second-order in space regardless of
    their time-order (both the implicit and Crank-Nicolson schemes use
    the same centred, second-order-in-space discretisation; the
    difference between them shows up in time, not space, see
    ode-engine and heat_equation.py for that distinction).
    """
    closed_form = bs.black_scholes_closed_form(s0, k, r, sigma, t, option_type)
    results = []
    for n_space in n_space_values:
        x_min = math.log(k) - 5 * sigma * math.sqrt(t) - 0.1
        x_max = math.log(k) + 5 * sigma * math.sqrt(t) + 0.1
        cn_result = bs.solve_bs_pde_crank_nicolson(k, r, sigma, t, x_min, x_max, n_space, n_time, option_type)
        cn_price = bs.price_at_spot(cn_result, s0)
        results.append({"n_space": n_space, "crank_nicolson_price": cn_price,
                         "error": abs(cn_price - closed_form)})
    return results
