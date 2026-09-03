import math


_A = (2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637)
_B = (-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833)
_C = (0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863,
      0.0038405729373609, 0.0003951896511919, 0.0000321767881768, 0.0000002888167364,
      0.0000003960315187)


def inverse_normal_cdf(u: float) -> float:
    """The inverse standard normal CDF (quantile function), via the
    Beasley-Springer-Moro algorithm: a rational-polynomial
    approximation for the bulk of the distribution (|u-0.5| < 0.42),
    switched to a more accurate asymptotic expansion in
    log(-log(min(u,1-u))) for the tails, which Moro's refinement
    specifically improves the accuracy of relative to the plain
    Beasley-Springer approximation. Tail accuracy matters more here
    than it would for an ordinary pseudo-random Monte Carlo simulation:
    a low-discrepancy sequence's very first points are often close to
    0 or 1 in some coordinate by construction (see low_discrepancy.py:
    van der Corput's first few points systematically explore extreme
    values before filling in the middle), so this transform is
    evaluated at genuinely extreme arguments far more often, in
    relative terms, than a uniform pseudo-random sequence would ever
    produce, making an inverse-CDF approximation's tail behaviour a
    real practical concern here, not just a theoretical nicety.

    This is why Box-Muller (which pairs two uniforms into two
    Gaussians via a trigonometric transform) is deliberately avoided
    for QMC applications in this project: applying it to two
    coordinates of a Halton sequence would mix two dimensions that
    were specifically constructed to be low-discrepancy independently
    of each other, entangling their carefully-designed uniformity
    properties in a way the inverse-CDF approach (applied
    coordinate-by-coordinate, preserving each dimension's own
    low-discrepancy structure) does not.
    """
    if u <= 0.0 or u >= 1.0:
        raise ValueError("inverse_normal_cdf requires 0 < u < 1")

    y = u - 0.5
    if abs(y) < 0.42:
        r = y * y
        a0, a1, a2, a3 = _A
        b1, b2, b3, b4 = _B
        numerator = y * (((a3 * r + a2) * r + a1) * r + a0)
        denominator = (((b4 * r + b3) * r + b2) * r + b1) * r + 1
        return numerator / denominator

    r = u if y < 0 else 1 - u
    r = math.log(-math.log(r))
    c0, c1, c2, c3, c4, c5, c6, c7, c8 = _C
    x = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * (c7 + r * c8)))))))
    return -x if y < 0 else x
