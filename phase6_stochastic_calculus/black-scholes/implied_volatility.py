import math
from typing import Callable, Dict

import black_scholes as bs


def implied_vol_bisection(market_price: float, s: float, k: float, t: float, r: float,
                           q: float = 0.0, option_type: str = "call",
                           lo: float = 1e-6, hi: float = 5.0,
                           tol: float = 1e-8, max_iter: int = 200) -> Dict[str, float]:
    """Solve for the volatility that reprices `market_price`, by
    bisection on the pricing function. Bisection needs only that the
    price is monotonically increasing in sigma (true here, since vega
    is strictly positive for any option with positive time value) and
    that the true root lies inside [lo, hi]; it does not need a
    derivative, which makes it slow but unconditionally reliable in
    the sense of always converging to *some* value where the priced
    residual is within `tol` in absolute terms.

    That last qualification matters: for options whose price is
    already extremely small (deep out-of-the-money, short-dated), an
    absolute price tolerance of 1e-8 is satisfied by a wide range of
    sigma values, since the pricing function is very flat near zero
    there. The solver will report success and a small residual, but
    the recovered sigma can differ meaningfully from the sigma that
    actually generated the price; see the module-level note in
    `implied_vol_newton` for a worked example and the corresponding
    test in the test suite.
    """
    price_fn = bs.call_price if option_type == "call" else bs.put_price

    f_lo = price_fn(s, k, t, r, lo, q) - market_price
    f_hi = price_fn(s, k, t, r, hi, q) - market_price
    if f_lo * f_hi > 0:
        raise ValueError("market_price is not bracketed by [lo, hi]; no solution in range")

    iterations = 0
    for iterations in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        f_mid = price_fn(s, k, t, r, mid, q) - market_price
        if abs(f_mid) < tol:
            break
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return {"sigma": mid, "iterations": iterations, "residual": f_mid}


def implied_vol_newton(market_price: float, s: float, k: float, t: float, r: float,
                        q: float = 0.0, option_type: str = "call",
                        initial_guess: float = 0.3, tol: float = 1e-8,
                        max_iter: int = 100) -> Dict[str, float]:
    """Newton-Raphson: sigma_{n+1} = sigma_n - f(sigma_n)/f'(sigma_n),
    where f(sigma) = price(sigma) - market_price and f'(sigma) = vega
    (shared between calls and puts, computed once per iteration).
    Converges quadratically once close to the root, far faster than
    bisection, but can fail to converge (or step into a negative or
    absurdly large sigma) from a poor starting guess or when vega is
    extremely small (deep in/out-of-the-money, very short-dated
    options), which is why a floor is applied to the step and the
    iteration falls back to failure rather than silently returning a
    nonsensical value.

    A related, subtler pitfall affects both this function and
    `implied_vol_bisection`: when the option's true price is already
    extremely small (a deep out-of-the-money, short-dated call, say),
    the absolute residual tolerance used to declare convergence is
    satisfied by a wide band of sigma values, because the pricing
    function is nearly flat near zero in that regime. A concrete
    example: S=100, K=140, T=0.1, r=0.02, sigma=0.15 gives a call
    price on the order of 1e-13; both solvers report a converged
    result with a tiny residual, but the recovered sigma can land
    around 0.16-0.18 rather than the true 0.15, purely because many
    different volatilities all price the option to within 1e-8 of
    zero. This is a real, well-known limitation of absolute-tolerance
    root-finding for implied volatility, not specific to this
    implementation; production systems typically use a relative price
    tolerance, a vega floor below which they refuse to quote an
    implied vol at all, or both.
    """
    price_fn = bs.call_price if option_type == "call" else bs.put_price
    sigma = initial_guess
    iterations = 0
    for iterations in range(1, max_iter + 1):
        price = price_fn(s, k, t, r, sigma, q)
        diff = price - market_price
        if abs(diff) < tol:
            return {"sigma": sigma, "iterations": iterations, "residual": diff, "converged": True}
        v = bs.vega(s, k, t, r, sigma, q)
        if v < 1e-12:
            return {"sigma": sigma, "iterations": iterations, "residual": diff, "converged": False}
        step = diff / v
        sigma -= step
        if sigma <= 0:
            sigma = 1e-6
    price = price_fn(s, k, t, r, sigma, q)
    return {"sigma": sigma, "iterations": iterations, "residual": price - market_price, "converged": False}


def implied_vol_newton_with_bisection_fallback(market_price: float, s: float, k: float, t: float,
                                                r: float, q: float = 0.0, option_type: str = "call",
                                                initial_guess: float = 0.3) -> Dict[str, float]:
    """Try Newton-Raphson first; if it fails to converge (poor start,
    vanishing vega, or a step that overshoots into an unreasonable
    region), fall back to bisection, which is slower but always
    converges given a valid bracket. This mirrors the standard
    production pattern for root-finding in pricing libraries: prefer
    speed, guarantee correctness.
    """
    newton_result = implied_vol_newton(market_price, s, k, t, r, q, option_type, initial_guess)
    if newton_result.get("converged", False):
        newton_result["method"] = "newton"
        return newton_result
    bisection_result = implied_vol_bisection(market_price, s, k, t, r, q, option_type)
    bisection_result["method"] = "bisection"
    bisection_result["newton_iterations_before_fallback"] = newton_result["iterations"]
    return bisection_result
