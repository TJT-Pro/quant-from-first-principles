import random
from typing import List


def l2_star_discrepancy_squared(points: List[List[float]]) -> float:
    """The squared L2-star discrepancy of a point set in [0,1]^d,
    via Warnock's (1972) closed-form formula.

    The L2-star discrepancy is defined as
    (D_N^*)^2 = integral_{[0,1]^d} (F_N(t) - V(t))^2 dt, where
    F_N(t) = (1/N)*#{points x_i with every coordinate < t's
    corresponding coordinate} is the point set's empirical CDF for the
    axis-aligned box [0,t), and V(t) = prod_j t_j is that box's actual
    volume: a point set fills space evenly exactly when its own
    empirical fraction inside every such box matches the box's true
    volume everywhere, and this integral measures the total squared
    mismatch across every box simultaneously.

    Expanding the square and integrating each of the three resulting
    terms directly (rather than approximating the integral
    numerically) gives a closed form with no supremum search or
    numerical integration needed:

        integral V(t)^2 dt = (1/3)^d   (prod_j integral_0^1 t_j^2 dt_j)

        integral F_N(t)*V(t) dt = (1/N) sum_i prod_j (1-x_ij^2)/2
        (the box [x_i, 1] is exactly where point i contributes to
        F_N(t), so integrating V(t) over just that region and
        averaging across points gives this term directly)

        integral F_N(t)^2 dt = (1/N^2) sum_{i,k} prod_j (1-max(x_ij,x_kj))
        (a point counts toward F_N(t) at t exactly when every
        coordinate of t exceeds that point's; two points i,k both
        count exactly when every coordinate of t exceeds
        max(x_ij, x_kj), integrated directly)

    Combining these three terms with the factor of 2 from expanding
    the square, (D_N^*)^2 = term1 - 2*term2 + term3, gives

        (1/3)^d - (2/N)*sum_i prod_j[(1-x_ij^2)/2] + (1/N^2)*sum_{i,k} prod_j[1-max(x_ij,x_kj)]

    A genuine bug was caught here directly by testing: an earlier
    version applied the (1-x_ij^2)/2 factor's division by 2 only once
    per point (treating the whole d-dimensional product as if
    dividing by 2 a single time were equivalent to dividing each of
    the d factors by 2 individually), when the division belongs inside
    the product, once per dimension, since this integral is itself a
    product of d separate one-dimensional integrals, each contributing
    its own factor of 1/2. The bug produced a *negative* squared
    discrepancy for a simple two-dimensional test case (a single point
    at (0.5, 0.5): -0.2014 instead of the correct, positive 0.0799,
    confirmed independently against a fine numerical grid-integration
    of the defining formula), an impossible value for a quantity
    defined as an integral of a squared quantity, which is exactly
    what made the error obvious rather than silently wrong. This is
    checked directly against a Monte-Carlo estimate of the same
    defining integral in the test suite, as an independent
    verification that the closed form and the definition it claims to
    equal actually agree numerically.
    """
    n = len(points)
    d = len(points[0])

    term1 = (1.0 / 3) ** d
    term2 = (2.0 / n) * sum(_prod((1 - xj ** 2) / 2 for xj in x) for x in points)

    term3 = 0.0
    for xi in points:
        for xk in points:
            term3 += _prod(1 - max(xij, xkj) for xij, xkj in zip(xi, xk))
    term3 /= n ** 2

    return term1 - term2 + term3


def _prod(values) -> float:
    result = 1.0
    for v in values:
        result *= v
    return result


def monte_carlo_l2_star_discrepancy_estimate(points: List[List[float]], n_test_boxes: int = 20000,
                                              seed: int = 0) -> float:
    """An independent, direct Monte Carlo estimate of the same
    (D_N^*)^2 integral l2_star_discrepancy_squared computes in closed
    form: sample many random boxes [0,t) uniformly, compute
    (F_N(t)-V(t))^2 at each, and average, exactly the defining
    integral's own Monte Carlo estimator, with no reliance on
    Warnock's formula at all. Used purely to cross-check the closed
    form against the definition it is supposed to equal.
    """
    rng = random.Random(seed)
    n = len(points)
    d = len(points[0])
    total = 0.0
    for _ in range(n_test_boxes):
        t = [rng.random() for _ in range(d)]
        count = sum(1 for x in points if all(xj < tj for xj, tj in zip(x, t)))
        f_n = count / n
        v = _prod(t)
        total += (f_n - v) ** 2
    return total / n_test_boxes


def random_uniform_points(n_points: int, dim: int, seed: int = 0) -> List[List[float]]:
    rng = random.Random(seed)
    return [[rng.random() for _ in range(dim)] for _ in range(n_points)]
