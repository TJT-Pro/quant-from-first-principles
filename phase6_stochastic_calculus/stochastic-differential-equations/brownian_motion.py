import math
import random
import statistics
from typing import List, Tuple


def simulate_brownian_path(t_end: float, n_steps: int, seed: int = None) -> Tuple[List[float], List[float]]:
    """Simulate a single path of standard Brownian motion W_t on
    [0, t_end] with n_steps equal increments, via the defining
    property of a Wiener process: independent increments
    W_{t+dt} - W_t ~ N(0, dt) for every dt > 0. Sampling each
    increment as sqrt(dt)*Z for Z ~ N(0,1) and accumulating them gives
    a discretely-sampled path whose increments are, by construction,
    independent and normally distributed with the correct variance at
    every resolution, which is the property every check in this
    module tests for.
    """
    rng = random.Random(seed)
    dt = t_end / n_steps
    ts = [0.0]
    ws = [0.0]
    w = 0.0
    for _ in range(n_steps):
        w += rng.gauss(0, math.sqrt(dt))
        ts.append(ts[-1] + dt)
        ws.append(w)
    return ts, ws


def verify_terminal_distribution(t_end: float, n_steps: int, n_paths: int, seed: int = 0) -> dict:
    """Simulate many independent paths and check the terminal value's
    empirical mean and variance against the exact theoretical values
    for standard Brownian motion, E[W_t]=0 and Var[W_t]=t, which
    follow directly from W_t being a sum of n_steps independent
    N(0, dt) increments (mean adds to 0, variance adds to n_steps*dt=t).
    """
    terminal_values = []
    for i in range(n_paths):
        _, ws = simulate_brownian_path(t_end, n_steps, seed=seed * 1_000_003 + i)
        terminal_values.append(ws[-1])
    return {
        "empirical_mean": statistics.mean(terminal_values),
        "empirical_variance": statistics.variance(terminal_values),
        "theoretical_mean": 0.0,
        "theoretical_variance": t_end,
    }


def quadratic_variation(ws: List[float]) -> float:
    """The realised quadratic variation of a path,
    sum_i (W_{t_{i+1}} - W_{t_i})^2, which converges (as the partition
    is refined) to exactly t for standard Brownian motion, a defining
    and famously counter-intuitive property: Brownian motion
    accumulates a finite, deterministic amount of squared fluctuation
    over any interval, even though its ordinary (unsquared) total
    variation is infinite. This finite quadratic variation is the
    entire reason Ito's lemma has a second-order correction term that
    an ordinary (finite-variation) calculus does not: (dW_t)^2 behaves
    like dt, not like a negligible higher-order term the way it would
    for a smooth, differentiable path.
    """
    return sum((ws[i + 1] - ws[i]) ** 2 for i in range(len(ws) - 1))


def verify_quadratic_variation_converges_to_t(t_end: float, n_steps_list: List[int], seed: int = 0) -> dict:
    """Compute the realised quadratic variation of the same underlying
    Brownian path at progressively finer resolutions (generated
    consistently, not independently re-simulated at each resolution,
    since re-simulating independently would test whether quadratic
    variation has the right expectation across paths, a weaker and
    less interesting claim than convergence along a single path), and
    confirm it approaches t as the partition is refined.
    """
    finest_n = n_steps_list[-1]
    _, ws_fine = simulate_brownian_path(t_end, finest_n, seed=seed)
    results = []
    for n_steps in n_steps_list:
        stride = finest_n // n_steps
        ws_coarse = ws_fine[::stride]
        qv = quadratic_variation(ws_coarse)
        results.append({"n_steps": n_steps, "quadratic_variation": qv})
    return {"results": results, "theoretical_limit": t_end}


def verify_scaling_property(c: float, t_end: float, n_steps: int, n_paths: int, seed: int = 0) -> dict:
    """Brownian scaling: for any c>0, the rescaled process
    (1/sqrt(c))*W_{c*t} is itself a standard Brownian motion on the
    same time interval, since its increments over an interval of
    length dt are (1/sqrt(c)) times an increment of the original
    process over an interval of length c*dt, which is N(0, c*dt),
    giving variance (1/c)*c*dt = dt after rescaling: exactly the
    increment distribution standard Brownian motion itself has.
    Checked here by comparing the empirical variance of
    (1/sqrt(c))*W_{c*t_end} across many simulated paths against
    t_end, the same target a standard, unscaled path's terminal
    variance should hit.
    """
    terminal_values = []
    for i in range(n_paths):
        _, ws = simulate_brownian_path(c * t_end, n_steps, seed=seed * 1_000_003 + i)
        scaled_terminal = ws[-1] / math.sqrt(c)
        terminal_values.append(scaled_terminal)
    return {
        "empirical_variance_of_scaled_process": statistics.variance(terminal_values),
        "theoretical_variance": t_end,
    }
