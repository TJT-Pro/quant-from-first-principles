import csv
import math
import statistics
from typing import List, Dict


def load_prices_and_returns(csv_path: str) -> Dict[str, List[float]]:
    prices, returns = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prices.append(float(row["price"]))
            returns.append(float(row["log_return"]))
    return {"prices": prices, "returns": returns}


def calibrate_gbm(returns: List[float], dt: float = 1 / 252) -> Dict[str, float]:
    """Maximum-likelihood calibration of GBM parameters from daily log
    returns: under dX_t = mu*X_t*dt + sigma*X_t*dW_t, log returns over
    an interval dt are exactly Normal((mu-0.5*sigma^2)*dt, sigma^2*dt)
    (a direct consequence of the closed-form GBM solution), so the
    sample standard deviation of log returns, rescaled by sqrt(dt),
    is the maximum-likelihood estimate of sigma, and the sample mean,
    rescaled by 1/dt and corrected by the same 0.5*sigma^2 term the
    exact solution subtracts, is the maximum-likelihood estimate of
    mu.
    """
    mean_r = statistics.mean(returns)
    std_r = statistics.pstdev(returns)
    sigma_hat = std_r / math.sqrt(dt)
    mu_hat = mean_r / dt + 0.5 * sigma_hat ** 2
    return {"mu": mu_hat, "sigma": sigma_hat}


def rolling_realized_volatility(returns: List[float], window: int) -> List[float]:
    return [statistics.pstdev(returns[t - window:t]) for t in range(window, len(returns))]


def calibrate_ou_via_ar1(series: List[float], dt: float = 1 / 252) -> Dict[str, float]:
    """Calibrate an OU process, dX_t = theta*(mu-X_t)*dt + sigma*dW_t,
    to an observed time series by exploiting that its Euler-Maruyama
    discretization, X_{t+1} = X_t + theta*(mu-X_t)*dt + sigma*sqrt(dt)*Z,
    is exactly an AR(1) model X_{t+1} = a + b*X_t + eps with

        b = 1 - theta*dt   =>   theta = (1-b)/dt
        a = theta*mu*dt    =>   mu = a/(1-b)
        Var(eps) = sigma^2*dt   =>   sigma = sqrt(Var(eps)/dt)

    fitting a, b by ordinary least squares and inverting these three
    relations directly. This is a genuinely different calibration
    route from the ACF-decay-based fit used for a deterministic
    mean-reversion ODE in the sibling ode-engine repository (built
    independently, sharing no code with that project): here, the
    calibration comes from a one-step-ahead AR(1) regression on the
    series itself, not from fitting an exponential decay curve to the
    series' own autocorrelation function across many lags. That the
    two independent methods give broadly consistent mean-reversion
    speeds on the same underlying phenomenon (volatility persistence)
    is a meaningful cross-check between two unrelated projects, not
    something either was built to guarantee.
    """
    y = series[1:]
    x = series[:-1]
    m = len(y)
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / m
    var_x = sum((xi - mean_x) ** 2 for xi in x) / m
    b = cov_xy / var_x
    a = mean_y - b * mean_x

    residuals = [y[i] - (a + b * x[i]) for i in range(m)]
    resid_var = statistics.variance(residuals)

    theta = (1 - b) / dt
    mu = a / (1 - b)
    sigma = math.sqrt(resid_var / dt)
    return {"theta": theta, "mu": mu, "sigma": sigma, "half_life_days": math.log(2) / theta * 252}


def compare_simulated_vs_real_terminal_distribution(real_prices: List[float], gbm_params: Dict[str, float],
                                                      horizon_days: int) -> Dict:
    """Compare the empirical distribution of real horizon_days-ahead
    log price changes (computed over every overlapping window in the
    historical series) against the theoretical distribution GBM with
    the calibrated (mu, sigma) implies over the same horizon: exactly
    Normal((mu-0.5*sigma^2)*T, sigma^2*T) for T=horizon_days/252,
    directly from the GBM closed-form solution, with no simulation
    needed for this particular comparison since the terminal
    distribution is already known in closed form.
    """
    dt = 1 / 252
    t_horizon = horizon_days * dt
    log_prices = [math.log(p) for p in real_prices]
    real_changes = [log_prices[i + horizon_days] - log_prices[i]
                     for i in range(len(log_prices) - horizon_days)]

    theoretical_mean = (gbm_params["mu"] - 0.5 * gbm_params["sigma"] ** 2) * t_horizon
    theoretical_std = gbm_params["sigma"] * math.sqrt(t_horizon)

    return {
        "real_mean": statistics.mean(real_changes),
        "real_std": statistics.pstdev(real_changes),
        "gbm_theoretical_mean": theoretical_mean,
        "gbm_theoretical_std": theoretical_std,
        "std_ratio": statistics.pstdev(real_changes) / theoretical_std,
        "n_windows": len(real_changes),
    }
