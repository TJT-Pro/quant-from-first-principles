import math
import statistics
from typing import Callable, Dict, List

import ode_solvers as ode


def global_error(f: Callable, t0: float, y0: float, t_end: float, h: float,
                  method: str, true_solution: Callable[[float], float]) -> float:
    """|numerical solution - true solution| at t_end, for a scalar IVP
    with a known closed-form solution. This is the *global* error (the
    accumulated effect of every step's local truncation error, not
    just one step's), which is the quantity whose scaling with h
    reveals a method's practical convergence order.
    """
    result = ode.solve_ivp(f, t0, y0, t_end, h, method)
    return abs(result["y"][-1] - true_solution(t_end))


def estimate_convergence_order(f: Callable, t0: float, y0: float, t_end: float,
                                h_values: List[float], method: str,
                                true_solution: Callable[[float], float]) -> Dict:
    """Fit log(error) = log(C) + p*log(h) by ordinary least squares
    across several step sizes, and read the fitted slope p off as the
    method's empirically observed convergence order. This is a direct
    empirical measurement, not a citation of the theoretical order: if
    global error truly scales as C*h^p, then log(error) is exactly
    linear in log(h) with slope p, and fitting that line to
    numerically computed errors at several h values recovers p (and
    the constant C, from the intercept) from data the same way any
    power-law relationship would be identified.
    """
    log_h = []
    log_err = []
    errors = []
    for h in h_values:
        err = global_error(f, t0, y0, t_end, h, method, true_solution)
        errors.append(err)
        if err > 0:
            log_h.append(math.log(h))
            log_err.append(math.log(err))

    if len(log_h) < 2:
        raise ValueError("need at least two nonzero-error step sizes to estimate an order")

    mean_x = statistics.mean(log_h)
    mean_y = statistics.mean(log_err)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_h, log_err))
    den = sum((x - mean_x) ** 2 for x in log_h)
    slope = num / den
    intercept = mean_y - slope * mean_x

    return {"h_values": h_values, "errors": errors, "estimated_order": slope,
            "log_constant": intercept}


def richardson_extrapolate(f: Callable, t0: float, y0: float, t_end: float, h: float,
                            method: str, order: int) -> float:
    """Richardson extrapolation: given numerical solutions at step
    size h and h/2 from a method with known order p, the leading error
    term can be cancelled algebraically. If y(h) = y_true + C*h^p +
    O(h^(p+1)) and y(h/2) = y_true + C*(h/2)^p + O(h^(p+1)), eliminating
    C between the two equations gives

        y_true ~ [2^p * y(h/2) - y(h)] / (2^p - 1),

    an estimate accurate to O(h^(p+1)), one order better than either
    individual solution, purely from combining two runs of the same
    method at two step sizes, without needing a higher-order method at
    all.
    """
    result_h = ode.solve_ivp(f, t0, y0, t_end, h, method)
    result_h2 = ode.solve_ivp(f, t0, y0, t_end, h / 2, method)
    y_h = result_h["y"][-1]
    y_h2 = result_h2["y"][-1]
    return (2 ** order * y_h2 - y_h) / (2 ** order - 1)
