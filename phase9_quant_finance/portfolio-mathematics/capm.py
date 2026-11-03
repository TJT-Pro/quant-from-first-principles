import statistics
from typing import List, Dict


def estimate_beta_alpha(asset_returns: List[float], market_returns: List[float]) -> Dict:
    """Ordinary least squares estimates of CAPM's beta and alpha from
    the single-factor regression R_i = alpha_i + beta_i*R_m + epsilon_i:

        beta_i = Cov(R_i, R_m) / Var(R_m)
        alpha_i = mean(R_i) - beta_i * mean(R_m)

    Beta measures systematic risk: how much of the asset's return
    co-moves with the market, in market-return units. Alpha (Jensen's
    alpha) measures the average return left over after accounting for
    that co-movement, the return CAPM attributes to something other
    than exposure to the market factor.
    """
    mean_r = statistics.mean(asset_returns)
    mean_m = statistics.mean(market_returns)
    n = len(asset_returns)

    cov = sum((asset_returns[i] - mean_r) * (market_returns[i] - mean_m) for i in range(n)) / n
    var_m = sum((market_returns[i] - mean_m) ** 2 for i in range(n)) / n

    beta = cov / var_m
    alpha = mean_r - beta * mean_m
    return {"beta": beta, "alpha": alpha}


def risk_decomposition(asset_returns: List[float], market_returns: List[float]) -> Dict:
    """Decompose an asset's total return variance into systematic
    (market-driven) and idiosyncratic (asset-specific) components:

        Var(R_i) = beta_i^2 * Var(R_m) + Var(epsilon_i)

    This identity holds exactly for OLS residuals, not just
    approximately, as a direct consequence of the OLS normal
    equations: OLS is defined by choosing (alpha, beta) to make the
    residuals epsilon_i = R_i - alpha - beta*R_m exactly uncorrelated
    with the regressor R_m (Cov(R_m, epsilon)=0, one of OLS's two
    defining first-order conditions). Since R_i = alpha + beta*R_m +
    epsilon, Var(R_i) = beta^2*Var(R_m) + Var(epsilon) +
    2*beta*Cov(R_m,epsilon), and that last cross term vanishes exactly
    because of that same orthogonality, leaving a clean additive split
    with no cross term left over, checked directly here rather than
    assumed.
    """
    fit = estimate_beta_alpha(asset_returns, market_returns)
    beta, alpha = fit["beta"], fit["alpha"]

    residuals = [asset_returns[i] - alpha - beta * market_returns[i] for i in range(len(asset_returns))]
    var_market = statistics.pvariance(market_returns)
    var_residual = statistics.pvariance(residuals)
    var_total = statistics.pvariance(asset_returns)

    systematic_variance = beta ** 2 * var_market
    r_squared = systematic_variance / var_total if var_total > 0 else 0.0

    return {"beta": beta, "alpha": alpha, "total_variance": var_total,
            "systematic_variance": systematic_variance, "idiosyncratic_variance": var_residual,
            "r_squared": r_squared, "residuals": residuals}


def security_market_line(beta: float, rf: float, market_return: float) -> float:
    """CAPM's central prediction: the expected return an asset with a
    given beta should earn if priced consistently with the market,
    E[R] = rf + beta*(E[R_m]-rf), a straight line in (beta, expected
    return) space through (0, rf) with slope equal to the market's own
    excess return. Jensen's alpha (see estimate_beta_alpha) is exactly
    the gap between an asset's actual average return and this line's
    prediction at that asset's own beta.
    """
    return rf + beta * (market_return - rf)
