import math
from typing import Callable


def crr_parameters(sigma: float, r: float, q: float, dt: float):
    """Cox-Ross-Rubinstein (1979) parameters for a recombining binomial
    tree that converges to geometric Brownian motion.

    u = e^{sigma*sqrt(dt)} and d = 1/u are chosen so that the tree's
    up/down moves match the *volatility* of GBM in the limit dt -> 0
    (matching the variance of log-price increments, sigma^2*dt, to
    second order), while the risk-neutral probability

        p = (e^{(r-q)dt} - d) / (u - d)

    is chosen so that the tree's expected one-step growth matches the
    risk-neutral drift (r-q) exactly at every step, not just in the
    limit: e^{(r-q)dt} = p*u + (1-p)*d by construction.
    """
    if sigma <= 0 or dt <= 0:
        raise ValueError("sigma and dt must be positive")
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError("risk-neutral probability out of [0,1]; step size dt is too large for these parameters")
    return u, d, p


def binomial_price(s: float, k: float, t: float, r: float, sigma: float, n_steps: int,
                    option_type: str = "call", exercise: str = "european", q: float = 0.0) -> float:
    """Price a European or American option by backward induction on a
    CRR binomial tree.

    At maturity, the option value at each terminal node is the payoff.
    Working backward, each node's value is the discounted risk-neutral
    expectation of its two children, e^{-r*dt}*(p*V_up + (1-p)*V_down);
    for an American option, this continuation value is compared against
    the immediate exercise value at that node, and the larger of the
    two is kept, since the holder is always free to exercise early.
    Only the current column of node values needs to be stored, since
    each step depends only on the previous one.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if exercise not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'")

    dt = t / n_steps
    u, d, p = crr_parameters(sigma, r, q, dt)
    discount = math.exp(-r * dt)

    def payoff(spot: float) -> float:
        return max(spot - k, 0.0) if option_type == "call" else max(k - spot, 0.0)

    values = [payoff(s * (u ** j) * (d ** (n_steps - j))) for j in range(n_steps + 1)]

    for step in range(n_steps - 1, -1, -1):
        new_values = []
        for j in range(step + 1):
            continuation = discount * (p * values[j + 1] + (1 - p) * values[j])
            if exercise == "american":
                spot = s * (u ** j) * (d ** (step - j))
                new_values.append(max(continuation, payoff(spot)))
            else:
                new_values.append(continuation)
        values = new_values
    return values[0]


def binomial_convergence_series(s: float, k: float, t: float, r: float, sigma: float,
                                 option_type: str, step_counts, q: float = 0.0):
    """Binomial price at each step count in `step_counts`, for studying
    convergence toward the closed-form Black-Scholes price as the
    number of steps grows. Convergence is not perfectly monotone
    (CRR trees oscillate slightly depending on whether the strike
    falls exactly on a tree node), which is itself worth observing
    directly rather than only citing.
    """
    return [binomial_price(s, k, t, r, sigma, n, option_type, "european", q) for n in step_counts]
