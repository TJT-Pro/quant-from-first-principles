import math
from typing import List, Dict

import linear_algebra as la
import mean_variance as mv


def sharpe_ratio(w: List[float], mu: List[float], sigma: List[List[float]], rf: float) -> float:
    excess_return = mv.portfolio_return(w, mu) - rf
    std = math.sqrt(mv.portfolio_variance(w, sigma))
    return excess_return / std


def tangency_portfolio(mu: List[float], sigma: List[List[float]], rf: float) -> Dict:
    """The tangency portfolio: the unique portfolio of risky assets
    that maximises the Sharpe ratio, and the point where a line drawn
    from (0, rf) in (standard deviation, return) space is tangent to
    the efficient frontier. In closed form,

        w_tan = Sigma^{-1}*(mu - rf*1) / [1^T * Sigma^{-1}*(mu - rf*1)],

    obtained by maximising (w^T*mu - rf) / sqrt(w^T*Sigma*w) subject
    only to w^T*1=1 (no target-return constraint at all, since the
    tangency portfolio is not "the minimum-variance portfolio at some
    particular return," it is the single portfolio whose risk-return
    trade-off is best among literally every combination of risky
    assets). Setting the Lagrangian's gradient to zero gives
    Sigma*w proportional to (mu - rf*1), i.e. w proportional to
    Sigma^{-1}*(mu-rf*1); normalising by w^T*1=1 pins the exact
    proportionality constant, giving the denominator above.
    """
    n = len(mu)
    sigma_inv = la.invert(sigma)
    excess = [mu[i] - rf for i in range(n)]
    raw = la.mat_vec(sigma_inv, excess)
    denom = sum(raw)
    w = [x / denom for x in raw]
    return {"weights": w, "return": mv.portfolio_return(w, mu),
            "variance": mv.portfolio_variance(w, sigma),
            "sharpe": sharpe_ratio(w, mu, sigma, rf)}


def capital_market_line(tangency: Dict, rf: float, target_std: float) -> float:
    """The Capital Market Line: the expected return achievable by
    combining the risk-free asset with the tangency portfolio at a
    chosen leverage alpha, expressed directly as a function of the
    resulting portfolio's standard deviation rather than of alpha
    itself. For a combination alpha*w_tan + (1-alpha)*(risk-free),
    the portfolio return is rf + alpha*(R_tan-rf) and the portfolio
    standard deviation is alpha*sigma_tan (the risk-free asset
    contributes no variance and no covariance), so eliminating alpha
    between the two gives

        R_p = rf + [(R_tan - rf) / sigma_tan] * sigma_p = rf + Sharpe_tan * sigma_p,

    a straight line in (standard deviation, return) space with
    intercept rf and slope exactly equal to the tangency portfolio's
    own Sharpe ratio: every point on this line is achievable, and
    (this is the Two-Fund Separation Theorem) every one of them
    dominates every risky-asset-only portfolio at the same standard
    deviation that is not itself proportional to the tangency
    portfolio.
    """
    sigma_tan = math.sqrt(tangency["variance"])
    return rf + tangency["sharpe"] * target_std


def two_fund_combination(tangency: Dict, rf: float, alpha: float) -> Dict:
    """A specific point on the Capital Market Line: alpha invested in
    the tangency portfolio, (1-alpha) in the risk-free asset (alpha>1
    corresponds to borrowing at the risk-free rate to lever up the
    tangency position). Both the expected return and the standard
    deviation scale linearly and proportionally with alpha, which is
    exactly why the CML is a straight line rather than a curve: the
    risk-free asset has zero variance and zero covariance with
    everything else, so mixing it in never bends the risk-return
    trade-off the way mixing two risky assets with imperfect
    correlation does.
    """
    r = rf + alpha * (tangency["return"] - rf)
    std = alpha * math.sqrt(tangency["variance"])
    return {"alpha": alpha, "return": r, "std": std, "variance": std ** 2}
