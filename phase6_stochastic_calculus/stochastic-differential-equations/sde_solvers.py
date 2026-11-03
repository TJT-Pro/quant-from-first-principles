import math
import random
from typing import Callable, List, Optional


def generate_fine_brownian_increments(t_end: float, n_fine: int, seed: int) -> List[float]:
    """Generate the increments of a single Brownian path at the finest
    resolution a convergence study will need, dW_i ~ N(0, dt_fine)
    independently for each of n_fine steps. Coarser-resolution
    increments for the same underlying path are obtained afterward by
    summing consecutive fine increments (aggregate_increments), not by
    re-simulating independently at each resolution: this is what makes
    a strong-error comparison meaningful, since strong convergence is
    a statement about how close a numerical scheme's path gets to the
    specific Brownian path realisation driving it, not merely whether
    the two paths have matching distributions.
    """
    rng = random.Random(seed)
    dt = t_end / n_fine
    sigma = math.sqrt(dt)
    return [rng.gauss(0, sigma) for _ in range(n_fine)]


def aggregate_increments(fine_increments: List[float], n_coarse: int) -> List[float]:
    """Sum consecutive groups of fine increments to obtain the
    increments of the identical underlying Brownian path at a coarser
    resolution. This is an exact identity for Brownian motion, not an
    approximation: W_{t+dt_coarse} - W_t equals the sum of every fine
    increment spanning that interval, directly from the definition of
    Brownian motion as having independent, additive increments.
    """
    n_fine = len(fine_increments)
    if n_fine % n_coarse != 0:
        raise ValueError("n_fine must be an exact multiple of n_coarse")
    stride = n_fine // n_coarse
    return [sum(fine_increments[i * stride:(i + 1) * stride]) for i in range(n_coarse)]


def euler_maruyama(mu: Callable[[float, float], float], sigma: Callable[[float, float], float],
                    x0: float, t0: float, t_end: float, dw: List[float]) -> dict:
    """The Euler-Maruyama scheme, the direct stochastic analogue of
    forward Euler for ODEs: X_{n+1} = X_n + mu(X_n,t_n)*dt +
    sigma(X_n,t_n)*dW_n, replacing the deterministic increment
    mu*dt with mu*dt plus a random shock sigma*dW_n matching the SDE's
    own diffusion term over that step. Just as forward Euler for ODEs
    has global error O(h) from truncating the Taylor expansion at
    first order, Euler-Maruyama's strong error (how close a single
    simulated path gets to the true path driven by the identical
    Brownian motion) is O(h^0.5), one power of h worse than its
    deterministic counterpart: the stochastic Taylor expansion
    Euler-Maruyama truncates has a term of order dW^2 ~ dt (from
    Brownian motion's quadratic variation, see brownian_motion.py)
    that an ordinary Taylor expansion would treat as negligible but
    an Ito-calculus expansion cannot, and this term is exactly what
    the Milstein scheme below adds back in.
    """
    n_steps = len(dw)
    dt = (t_end - t0) / n_steps
    ts = [t0]
    xs = [x0]
    x = x0
    for i in range(n_steps):
        t = ts[-1]
        x = x + mu(x, t) * dt + sigma(x, t) * dw[i]
        ts.append(t + dt)
        xs.append(x)
    return {"t": ts, "x": xs}


def milstein(mu: Callable[[float, float], float], sigma: Callable[[float, float], float],
             x0: float, t0: float, t_end: float, dw: List[float],
             sigma_x: Optional[Callable[[float, float], float]] = None, h: float = 1e-5) -> dict:
    """The Milstein scheme adds the one missing term from
    Euler-Maruyama's stochastic Taylor expansion:

        X_{n+1} = X_n + mu*dt + sigma*dW_n + 0.5*sigma*(d sigma/d x)*(dW_n^2 - dt)

    The correction term comes directly from applying Ito's lemma to
    sigma(X_t,t) itself along the path and substituting the result
    back into the SDE's own stochastic Taylor expansion: since
    dW_n^2 has mean dt (matching Brownian motion's quadratic variation
    exactly) but fluctuates around that mean at order dt itself (not
    negligible relative to the dt-order terms already kept), subtracting
    dt recentres the correction to have zero mean while still capturing
    its dt-order fluctuation. This raises the strong convergence order
    to O(h) (matching deterministic Euler's order, unlike
    Euler-Maruyama's O(h^0.5)), at the cost of needing sigma's
    derivative with respect to x, either supplied analytically via
    sigma_x or approximated here by central finite differences when
    not supplied.
    """
    sigma_x_fn = sigma_x or (lambda x, t: (sigma(x + h, t) - sigma(x - h, t)) / (2 * h))
    n_steps = len(dw)
    dt = (t_end - t0) / n_steps
    ts = [t0]
    xs = [x0]
    x = x0
    for i in range(n_steps):
        t = ts[-1]
        s = sigma(x, t)
        s_x = sigma_x_fn(x, t)
        x = x + mu(x, t) * dt + s * dw[i] + 0.5 * s * s_x * (dw[i] ** 2 - dt)
        ts.append(t + dt)
        xs.append(x)
    return {"t": ts, "x": xs}
