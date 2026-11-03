import math
import random
from typing import List, Tuple


def simulate_gbm_exact(x0: float, mu: float, sigma: float, t_end: float, n_steps: int,
                        seed: int = None) -> Tuple[List[float], List[float]]:
    """Exact simulation of geometric Brownian motion,
    dX_t = mu*X_t*dt + sigma*X_t*dW_t, using its closed-form solution

        X_t = X_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t),

    obtained by applying Ito's lemma to f(x)=ln(x) (see
    ito_calculus.py for the derivation) to turn the multiplicative SDE
    into an additive one for ln(X_t), which is then just ln(X_0) plus
    a drifted Brownian motion, solvable directly. Because the exact
    solution is known, this function has zero discretization error at
    any step size: it needs only to sample W_t at each requested time
    point (via cumulative Brownian increments) and evaluate the closed
    form there, unlike Euler-Maruyama or Milstein, which approximate
    the SDE step by step and accumulate error as they go.
    """
    rng = random.Random(seed)
    dt = t_end / n_steps
    ts = [0.0]
    xs = [x0]
    w = 0.0
    for _ in range(n_steps):
        w += rng.gauss(0, math.sqrt(dt))
        t = ts[-1] + dt
        x = x0 * math.exp((mu - 0.5 * sigma ** 2) * t + sigma * w)
        ts.append(t)
        xs.append(x)
    return ts, xs


def simulate_ou_exact(x0: float, theta: float, mu: float, sigma: float, t_end: float, n_steps: int,
                       seed: int = None) -> Tuple[List[float], List[float]]:
    """Exact simulation of the Ornstein-Uhlenbeck process,
    dX_t = theta*(mu - X_t)*dt + sigma*dW_t, using its known exact
    Gaussian transition density rather than a discretized
    approximation: conditional on X_t, the next value X_{t+dt} is
    exactly

        X_{t+dt} ~ N( X_t*e^{-theta*dt} + mu*(1-e^{-theta*dt}),
                       sigma^2/(2*theta) * (1-e^{-2*theta*dt}) )

    This follows from solving the linear SDE directly (via an
    integrating-factor argument identical in spirit to solving a
    linear ODE, applied to Ito's formula for d(e^{theta*t}*X_t)) and
    then using that Ito integrals of a deterministic integrand against
    dW_s are themselves exactly Gaussian, with the variance given by
    Ito's isometry. Sampling from this transition exactly at each step
    means this simulation has zero discretization error regardless of
    how large dt is, unlike Euler-Maruyama or Milstein applied to the
    same SDE, which are increasingly biased as dt grows.
    """
    rng = random.Random(seed)
    dt = t_end / n_steps
    decay = math.exp(-theta * dt)
    variance = (sigma ** 2) / (2 * theta) * (1 - math.exp(-2 * theta * dt)) if theta > 0 else sigma ** 2 * dt
    std = math.sqrt(variance)

    ts = [0.0]
    xs = [x0]
    x = x0
    for _ in range(n_steps):
        mean = x * decay + mu * (1 - decay)
        x = rng.gauss(mean, std)
        ts.append(ts[-1] + dt)
        xs.append(x)
    return ts, xs


def ou_stationary_distribution(theta: float, mu: float, sigma: float) -> Tuple[float, float]:
    """The stationary (long-run, t->infinity) distribution of an OU
    process is N(mu, sigma^2/(2*theta)): taking t->infinity in the
    exact transition variance formula above, e^{-2*theta*t}->0
    (for theta>0), leaving sigma^2/(2*theta) as the process's own
    long-run variance, independent of the starting point, exactly the
    balance point where the process's constant diffusion input and its
    mean-reverting pull toward mu offset each other on average.
    """
    if theta <= 0:
        raise ValueError("stationary distribution requires theta > 0 (genuine mean reversion)")
    return mu, sigma ** 2 / (2 * theta)
