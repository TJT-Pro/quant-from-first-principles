import math
from collections import Counter
from typing import Dict, Sequence, Tuple

import entropy as ent


def kl_divergence(p: Dict, q: Dict, base: float = 2.0) -> float:
    """The Kullback-Leibler divergence D(P||Q) = sum_x p(x)*log(p(x)/q(x)),
    the expected number of extra bits needed to encode samples from P
    using a code optimised for Q instead of one optimised for P
    itself. Gibbs' inequality guarantees D(P||Q) >= 0 always, with
    equality if and only if P=Q: a direct consequence of the concavity
    of log and Jensen's inequality applied to
    -D(P||Q) = sum_x p(x)*log(q(x)/p(x)) <= log(sum_x p(x)*q(x)/p(x)) = log(1) = 0
    (Jensen's inequality for the concave function log says the log of
    an average is at least the average of the logs, i.e.
    E[log(Z)] <= log(E[Z]) for any Z, applied here with Z=q(x)/p(x)
    under the distribution P). D(P||Q) is NOT symmetric in general
    (D(P||Q) != D(Q||P)), so it is a divergence, not a true distance
    metric, even though it behaves somewhat like a squared-distance
    for distributions close together.

    A term where p(x)>0 but q(x)=0 makes D(P||Q) infinite (Q assigns
    zero probability to an outcome P says can happen, an infinitely
    costly mismatch for any code built assuming Q); this is the
    correct mathematical value, returned directly here rather than
    silently avoided.
    """
    total = 0.0
    for x, px in p.items():
        if px <= 0:
            continue
        qx = q.get(x, 0.0)
        if qx <= 0:
            return float("inf")
        total += px * math.log(px / qx, base)
    return total


def cross_entropy(p: Dict, q: Dict, base: float = 2.0) -> float:
    """H(P,Q) = -sum_x p(x)*log(q(x)), the expected number of bits
    needed to encode samples from the true distribution P using a code
    that was optimised for (built assuming) Q instead. The identity
    H(P,Q) = H(P) + D(P||Q) follows directly from expanding
    D(P||Q) = sum p*log(p) - sum p*log(q) = -H(P) - sum p*log(q), which
    rearranges to sum p*log(q) = -H(P) - D(P||Q), i.e.
    H(P,Q) = -sum p*log(q) = H(P) + D(P||Q): the "extra cost" of using
    the wrong code (D(P||Q)) added on top of the unavoidable minimum
    cost (H(P)) any code faces even when perfectly matched to P.
    """
    total = 0.0
    for x, px in p.items():
        if px <= 0:
            continue
        qx = q.get(x, 0.0)
        if qx <= 0:
            return float("inf")
        total -= px * math.log(qx, base)
    return total


def mutual_information_from_pairs(pairs: Sequence[Tuple], base: float = 2.0) -> float:
    """I(X;Y) = H(X) + H(Y) - H(X,Y), the reduction in uncertainty
    about one variable gained by observing the other. Equivalently,
    I(X;Y) = D(P(X,Y) || P(X)*P(Y)): the KL divergence between the true
    joint distribution and the product-of-marginals distribution that
    would hold exactly if X and Y were independent, making I(X;Y) a
    direct, principled measure of "how far from independent" the two
    variables are. Unlike ordinary (Pearson) correlation, mutual
    information captures ANY statistical dependence, linear or not:
    two variables can have zero correlation while still being strongly
    (nonlinearly) dependent, and I(X;Y) would correctly detect that
    dependence where a correlation coefficient would report exactly
    zero. Both defining formulas are checked against each other
    directly in the test suite, not merely asserted equivalent.
    """
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    h_x = ent.entropy_from_counts(xs, base)
    h_y = ent.entropy_from_counts(ys, base)
    h_xy = ent.joint_entropy_from_pairs(pairs, base)
    return h_x + h_y - h_xy


def mutual_information_via_kl(pairs: Sequence[Tuple], base: float = 2.0) -> float:
    """I(X;Y) computed via the direct D(P(X,Y) || P(X)*P(Y)) definition,
    as an independent cross-check against mutual_information_from_pairs'
    entropy-based formula: both should agree exactly (up to floating-
    point rounding) for any finite empirical joint distribution, since
    they are algebraically the same quantity derived two different
    ways.
    """
    n = len(pairs)
    joint_counts = Counter(pairs)
    x_counts = Counter(p[0] for p in pairs)
    y_counts = Counter(p[1] for p in pairs)

    joint_dist = {k: v / n for k, v in joint_counts.items()}
    independent_dist = {(x, y): (x_counts[x] / n) * (y_counts[y] / n) for x, y in joint_counts}
    return kl_divergence(joint_dist, independent_dist, base)


def normalized_mutual_information(pairs: Sequence[Tuple], base: float = 2.0) -> float:
    """I(X;Y) normalised by min(H(X), H(Y)), giving a quantity in
    [0, 1]: 0 for independence, 1 when one variable is a deterministic
    function of the other (the maximum possible mutual information
    given the smaller variable's own entropy budget, since I(X;Y) can
    never exceed either individual marginal entropy: knowing Y cannot
    reduce X's uncertainty by more than the total uncertainty X had to
    begin with). This normalisation makes mutual information values
    comparable across variable pairs with different alphabet sizes,
    unlike the raw (unnormalised) bit value, which is influenced by
    the entropy scale of the variables involved as well as by their
    genuine dependence.
    """
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    h_x = ent.entropy_from_counts(xs, base)
    h_y = ent.entropy_from_counts(ys, base)
    mi = mutual_information_from_pairs(pairs, base)
    denom = min(h_x, h_y)
    return mi / denom if denom > 0 else 0.0
