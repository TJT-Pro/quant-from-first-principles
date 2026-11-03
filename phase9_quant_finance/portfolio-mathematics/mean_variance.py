from typing import List, Dict

import linear_algebra as la


def _abcd(mu: List[float], sigma: List[List[float]]) -> Dict[str, float]:
    """The four scalar constants that appear throughout classical
    mean-variance portfolio theory, all built from Sigma^{-1}:

        A = 1^T * Sigma^{-1} * 1
        B = 1^T * Sigma^{-1} * mu   (= mu^T * Sigma^{-1} * 1)
        C = mu^T * Sigma^{-1} * mu
        D = A*C - B^2

    These arise directly from solving the two-constraint (target
    return, full investment) minimum-variance problem via Lagrange
    multipliers: minimizing 0.5*w^T*Sigma*w subject to w^T*mu=target
    and w^T*1=1 gives stationarity condition
    Sigma*w = lambda_1*mu + lambda_2*1, so w = Sigma^{-1}*(lambda_1*mu
    + lambda_2*1); substituting back into both constraints gives a
    2x2 linear system in (lambda_1, lambda_2) whose coefficients are
    exactly A, B, C. D (which must be strictly positive whenever mu is
    not a constant multiple of the all-ones vector, i.e. whenever
    assets genuinely differ in expected return relative to their risk)
    is what makes that 2x2 system solvable at all.
    """
    n = len(mu)
    sigma_inv = la.invert(sigma)
    ones = [1.0] * n
    sigma_inv_ones = la.mat_vec(sigma_inv, ones)
    sigma_inv_mu = la.mat_vec(sigma_inv, mu)

    a = la.dot(ones, sigma_inv_ones)
    b = la.dot(ones, sigma_inv_mu)
    c = la.dot(mu, sigma_inv_mu)
    d = a * c - b ** 2
    return {"A": a, "B": b, "C": c, "D": d, "sigma_inv": sigma_inv}


def efficient_frontier_portfolio(mu: List[float], sigma: List[List[float]],
                                  target_return: float) -> Dict:
    """The minimum-variance portfolio achieving exactly `target_return`,
    in closed form: solving the 2x2 system for the Lagrange
    multipliers,

        lambda_1 = (A*target - B) / D
        lambda_2 = (C - B*target) / D

    then w = Sigma^{-1}*(lambda_1*mu + lambda_2*1) directly, no
    iteration needed. The achieved variance simplifies elegantly using
    the stationarity condition itself: since Sigma*w = lambda_1*mu +
    lambda_2*1 exactly, w^T*Sigma*w = lambda_1*(w^T*mu) +
    lambda_2*(w^T*1) = lambda_1*target + lambda_2, avoiding a full
    quadratic-form computation.
    """
    stats = _abcd(mu, sigma)
    a, b, c, d = stats["A"], stats["B"], stats["C"], stats["D"]
    sigma_inv = stats["sigma_inv"]
    n = len(mu)

    lambda1 = (a * target_return - b) / d
    lambda2 = (c - b * target_return) / d

    combined = [lambda1 * mu[i] + lambda2 for i in range(n)]
    w = la.mat_vec(sigma_inv, combined)
    variance = lambda1 * target_return + lambda2

    return {"weights": w, "target_return": target_return, "variance": variance}


def frontier_variance_formula(mu: List[float], sigma: List[List[float]], target_return: float) -> float:
    """The efficient frontier's variance as an explicit function of
    target return, without solving for portfolio weights at all:

        sigma_p^2(target) = (A*target^2 - 2*B*target + C) / D

    obtained by substituting the closed-form lambda_1, lambda_2 back
    into variance = lambda_1*target + lambda_2 and simplifying. This
    is a parabola in (variance, return) space -- the classical
    "Markowitz bullet" is exactly this parabola's image under the
    square-root map into (standard deviation, return) space, which is
    why the frontier's efficient (upper) half looks like one branch of
    a hyperbola instead. Provided here as an independent formula to
    check efficient_frontier_portfolio's computed variance against.
    """
    stats = _abcd(mu, sigma)
    a, b, c, d = stats["A"], stats["B"], stats["C"], stats["D"]
    return (a * target_return ** 2 - 2 * b * target_return + c) / d


def global_minimum_variance_portfolio(mu: List[float], sigma: List[List[float]]) -> Dict:
    """The single portfolio on the efficient frontier with the lowest
    possible variance of all: minimizing the parabola
    sigma_p^2(target) = (A*target^2-2*B*target+C)/D over target
    directly (d/d(target) = (2*A*target-2*B)/D = 0) gives
    target* = B/A, the expected return of the global minimum-variance
    portfolio. Substituting back into the general closed form gives
    weights that simplify to exactly w = Sigma^{-1}*1/A, the same
    formula obtainable directly by dropping the return constraint
    entirely and minimizing variance subject only to full investment
    (verified to agree in the test suite): the target-return
    constraint becomes non-binding exactly at this one specific point
    on the frontier.
    """
    stats = _abcd(mu, sigma)
    a = stats["A"]
    target = stats["B"] / a
    return efficient_frontier_portfolio(mu, sigma, target)


def efficient_frontier_curve(mu: List[float], sigma: List[List[float]],
                              target_returns: List[float]) -> List[Dict]:
    return [efficient_frontier_portfolio(mu, sigma, t) for t in target_returns]


def portfolio_return(w: List[float], mu: List[float]) -> float:
    return la.dot(w, mu)


def portfolio_variance(w: List[float], sigma: List[List[float]]) -> float:
    return la.dot(w, la.mat_vec(sigma, w))
