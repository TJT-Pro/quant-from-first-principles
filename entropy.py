import math
from collections import Counter
from typing import Dict, Sequence, Tuple


def entropy(probabilities: Dict, base: float = 2.0) -> float:
    """Shannon entropy H(X) = -sum_x p(x)*log(p(x)), the expected
    number of bits (for base=2) needed to describe the outcome of a
    random variable X on average, under an optimally-designed code.
    Zero-probability outcomes contribute 0 (not -inf*0, which is
    mathematically undefined): the standard convention, justified by
    the limit p*log(p) -> 0 as p -> 0, since log(p) diverges to
    -infinity more slowly than p shrinks to 0.
    """
    total = 0.0
    for p in probabilities.values():
        if p > 0:
            total -= p * math.log(p, base)
    return total


def entropy_from_counts(counts: Sequence, base: float = 2.0) -> float:
    """Estimate H(X) from a sequence of observed outcomes, via the
    empirical (maximum-likelihood) frequency distribution: count each
    distinct value's occurrences, normalise to probabilities, and
    apply the entropy formula directly. This is a plug-in estimator
    (it treats the empirical frequencies as if they were the true
    probabilities), which is known to be biased downward for finite
    samples (finite-sample noise in the frequency estimates makes the
    empirical distribution look slightly more concentrated/predictable
    than the true one, understating the true entropy); the bias
    shrinks as the sample size grows relative to the alphabet size,
    and is negligible for the small alphabets and large sample sizes
    (thousands of observations, alphabets of 2-8 symbols) this project
    actually applies it to.
    """
    n = len(counts)
    freq = Counter(counts)
    probabilities = {k: v / n for k, v in freq.items()}
    return entropy(probabilities, base)


def joint_entropy_from_pairs(pairs: Sequence[Tuple], base: float = 2.0) -> float:
    """H(X,Y) estimated the same plug-in way as entropy_from_counts,
    but treating each (x,y) pair as a single joint outcome: the
    entropy of the joint distribution over the Cartesian product of
    both variables' alphabets.
    """
    return entropy_from_counts(list(pairs), base)


def conditional_entropy(pairs: Sequence[Tuple], base: float = 2.0) -> float:
    """H(Y|X) = H(X,Y) - H(X), the chain rule for entropy: the
    remaining uncertainty about Y once X is known equals the total
    joint uncertainty minus what X alone already resolves. Derived
    directly from the plug-in joint and marginal entropies here
    (rather than from a separate per-x-value conditional-distribution
    formula) specifically so the chain-rule identity itself can be
    checked as an internal consistency property of this module, not
    merely assumed.
    """
    xs = [p[0] for p in pairs]
    h_xy = joint_entropy_from_pairs(pairs, base)
    h_x = entropy_from_counts(xs, base)
    return h_xy - h_x


def max_entropy(n_symbols: int, base: float = 2.0) -> float:
    """The maximum possible entropy for a discrete random variable
    with n_symbols possible outcomes, log(n_symbols), attained
    uniquely by the uniform distribution. This is a direct consequence
    of Gibbs' inequality (see divergence.py): the KL divergence between
    any distribution P and the uniform distribution U is always
    non-negative, D(P||U) = log(n_symbols) - H(P) >= 0, with equality
    exactly when P=U, so log(n_symbols) - H(P) >= 0 for every P, i.e.
    H(P) <= log(n_symbols) always, with the uniform distribution
    alone achieving the bound.
    """
    return math.log(n_symbols, base)
