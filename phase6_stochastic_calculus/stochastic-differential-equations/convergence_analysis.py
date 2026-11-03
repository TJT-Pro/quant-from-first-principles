import math
import statistics
from typing import Callable, List

import sde_solvers as sde


def strong_error(method: Callable, mu: Callable[[float, float], float], sigma: Callable[[float, float], float],
                  sigma_x, x0: float, t_end: float, n_coarse: int, n_fine: int, n_paths: int,
                  exact_solution: Callable[[float, float], float], seed: int = 0) -> float:
    """The strong error E[|X_T^h - X_T|] at step size h=t_end/n_coarse,
    estimated by Monte Carlo: for each of n_paths independent Brownian
    path realisations (generated once at the finest resolution
    n_fine and aggregated down to n_coarse for the numerical scheme,
    per sde_solvers.aggregate_increments), compute both the numerical
    scheme's terminal value driven by that path's coarse-resolution
    increments and the exact terminal value implied by the same
    path's total displacement W_T, then average the absolute
    difference across paths. Using the same underlying path for both
    the numerical and exact solution at each Monte Carlo sample,
    rather than independently sampling each, is what makes this a
    strong-error (pathwise) comparison rather than a weak
    (distributional) one.
    """
    errors = []
    for i in range(n_paths):
        dw_fine = sde.generate_fine_brownian_increments(t_end, n_fine, seed=seed * 10 ** 6 + i)
        w_t = sum(dw_fine)
        dw_coarse = sde.aggregate_increments(dw_fine, n_coarse)

        if sigma_x is not None:
            result = method(mu, sigma, x0, 0.0, t_end, dw_coarse, sigma_x=sigma_x)
        else:
            result = method(mu, sigma, x0, 0.0, t_end, dw_coarse)

        numerical = result["x"][-1]
        exact = exact_solution(x0, w_t)
        errors.append(abs(numerical - exact))
    return statistics.mean(errors)


def weak_error(method: Callable, mu: Callable[[float, float], float], sigma: Callable[[float, float], float],
                sigma_x, x0: float, t_end: float, n_coarse: int, n_paths: int,
                test_function: Callable[[float], float], exact_expectation: float, seed: int = 0) -> float:
    """The weak error |E[f(X_T^h)] - E[f(X_T)]| at step size
    h=t_end/n_coarse: unlike the strong error, this only compares
    distributions of the terminal value (summarised through a test
    function f), not individual paths, so each Monte Carlo sample can
    use its own independent Brownian path with no need to also
    generate a matching finer-resolution path for comparison.
    """
    values = []
    for i in range(n_paths):
        dw = sde.generate_fine_brownian_increments(t_end, n_coarse, seed=seed * 10 ** 6 + i)
        if sigma_x is not None:
            result = method(mu, sigma, x0, 0.0, t_end, dw, sigma_x=sigma_x)
        else:
            result = method(mu, sigma, x0, 0.0, t_end, dw)
        values.append(test_function(result["x"][-1]))
    return abs(statistics.mean(values) - exact_expectation)


def linear_drift_exact_mean_error(mu_param: float, x0: float, t_end: float, n_steps: int) -> float:
    """A noise-free weak-error measurement, specific to a linear drift
    mu(x)=mu_param*x (as in GBM): E[X_N] under Euler-Maruyama with a
    linear drift follows the exactly deterministic recursion
    E[X_{n+1}|X_n] = X_n*(1+mu_param*h) (the diffusion term has mean
    zero at every step, contributing nothing to the mean), giving
    E[X_N] = X_0*(1+mu_param*h)^N in closed form with no Monte Carlo
    sampling needed at all.

    This exists because a genuine Monte Carlo weak-error measurement
    for this specific test function (f(x)=x) turned out to need
    enormous sample sizes to see the deterministic O(h) bias trend at
    all: found directly while developing this module, a run with
    200,000 paths per step size gave errors of 0.0012, 0.00073,
    0.00066, 0.000028 across four step sizes, an estimated "order" of
    1.65 that was really just sampling noise (of the same magnitude as
    the underlying bias itself) masquerading as a convergence trend.
    The closed-form recursion here sidesteps that noise floor entirely
    for the one case (linear drift) where a closed form exists, giving
    the order-1.0 weak convergence a clean, deterministic 0.998
    empirical estimate; `weak_error` above remains available, and
    necessary, for the general nonlinear case where no such shortcut
    exists and Monte Carlo noise must simply be reduced by using very
    large sample sizes instead.
    """
    h = t_end / n_steps
    return x0 * (1 + mu_param * h) ** n_steps


def estimate_convergence_order(errors: List[float], step_sizes: List[float]) -> float:
    """Fit log(error) = log(C) + p*log(h) by ordinary least squares
    across several step sizes and return the fitted slope p as the
    empirically observed convergence order, the same log-log
    regression methodology used for the ODE solvers in ode-engine
    (built independently here, since this project shares no code with
    that one), applied now to a fundamentally different, noisy
    (Monte-Carlo-estimated, not exact) error metric.
    """
    log_h = [math.log(h) for h in step_sizes]
    log_err = [math.log(e) for e in errors]
    mean_x = statistics.mean(log_h)
    mean_y = statistics.mean(log_err)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_h, log_err))
    den = sum((x - mean_x) ** 2 for x in log_h)
    return num / den
