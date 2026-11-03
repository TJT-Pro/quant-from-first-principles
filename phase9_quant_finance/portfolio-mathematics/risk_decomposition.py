import math
from typing import List, Dict

import linear_algebra as la


def portfolio_volatility(w: List[float], sigma: List[List[float]]) -> float:
    return math.sqrt(la.dot(w, la.mat_vec(sigma, w)))


def marginal_contribution_to_risk(w: List[float], sigma: List[List[float]]) -> List[float]:
    """MCTR_i = d(sigma_p)/d(w_i) = (Sigma*w)_i / sigma_p: how much the
    portfolio's total volatility changes per unit change in asset i's
    weight, holding every other weight fixed. Differentiating
    sigma_p(w) = sqrt(w^T*Sigma*w) directly:
    d(sigma_p)/d(w_i) = (1/(2*sigma_p)) * d(w^T*Sigma*w)/d(w_i) =
    (1/(2*sigma_p)) * 2*(Sigma*w)_i = (Sigma*w)_i / sigma_p, using that
    Sigma is symmetric so the two copies of w in the quadratic form
    contribute identically to the derivative.
    """
    sigma_p = portfolio_volatility(w, sigma)
    sigma_w = la.mat_vec(sigma, w)
    return [x / sigma_p for x in sigma_w]


def component_contribution_to_risk(w: List[float], sigma: List[List[float]]) -> List[float]:
    """CCTR_i = w_i * MCTR_i: asset i's own share of total portfolio
    volatility, in the same units as sigma_p itself (unlike MCTR,
    which is a sensitivity, CCTR is an actual additive decomposition,
    verified below to sum to exactly sigma_p).
    """
    mctr = marginal_contribution_to_risk(w, sigma)
    return [w[i] * mctr[i] for i in range(len(w))]


def verify_euler_decomposition(w: List[float], sigma: List[List[float]]) -> Dict:
    """Euler's theorem for homogeneous functions: portfolio volatility
    sigma_p(w) = sqrt(w^T*Sigma*w) is homogeneous of degree 1 in w
    (scaling every weight by a positive constant t scales sigma_p by
    exactly the same t, directly from the square root of a quadratic
    form), and Euler's theorem for any degree-1-homogeneous function f
    states sum_i w_i * df/dw_i = f(w) exactly. Applied here, this
    means the component contributions to risk sum to exactly the total
    portfolio volatility, with no leftover or double-counted risk: a
    genuinely additive risk decomposition, not merely a set of
    sensitivities that happen to be reported together. Checked
    directly here (not merely cited) by comparing the sum of component
    contributions against portfolio_volatility computed independently.
    """
    cctr = component_contribution_to_risk(w, sigma)
    sigma_p = portfolio_volatility(w, sigma)
    return {"component_contributions": cctr, "sum_of_components": sum(cctr),
            "portfolio_volatility": sigma_p, "matches": math.isclose(sum(cctr), sigma_p, rel_tol=1e-9)}


def risk_contribution_percentages(w: List[float], sigma: List[List[float]]) -> List[float]:
    """Each asset's share of total portfolio risk, as a fraction
    summing to exactly 1 (a direct consequence of the Euler
    decomposition above): useful for identifying risk concentration
    that a plain weight (capital allocation) percentage can hide
    entirely, since a small-weight but highly volatile or highly
    correlated-with-the-rest-of-the-portfolio asset can contribute a
    disproportionate share of total risk relative to its capital share.
    """
    cctr = component_contribution_to_risk(w, sigma)
    sigma_p = portfolio_volatility(w, sigma)
    return [c / sigma_p for c in cctr]
