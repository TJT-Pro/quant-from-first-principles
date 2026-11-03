import math
from typing import List

from grid_approximation import log_sum_exp


def log_bayes_factor(log_evidence_1: float, log_evidence_2: float) -> float:
    """ln(P(D|M1)/P(D|M2)): the log ratio of two models' marginal
    likelihoods. Working in log-space throughout (rather than
    exponentiating each evidence separately and dividing) avoids
    underflow, since a marginal likelihood integrated over even a
    few dozen binomial or Poisson observations is routinely far
    smaller than the smallest representable positive float.
    """
    return log_evidence_1 - log_evidence_2


def bayes_factor(log_evidence_1: float, log_evidence_2: float) -> float:
    return math.exp(log_bayes_factor(log_evidence_1, log_evidence_2))


def interpret_bayes_factor(bf: float) -> str:
    """Jeffreys' (1961) descriptive scale for interpreting a Bayes
    factor's strength of evidence for model 1 over model 2. This is a
    convention for communicating a continuous quantity in words, not a
    formal decision rule; the boundaries themselves (1, 3, 10, 30, 100)
    are traditional rather than derived from any optimality criterion.
    """
    if bf < 1:
        return f"evidence favours model 2 (BF={bf:.3g}, i.e. 1/BF={1/bf:.3g} in its favour)"
    if bf < 3:
        return f"barely worth mentioning evidence for model 1 (BF={bf:.3g})"
    if bf < 10:
        return f"substantial evidence for model 1 (BF={bf:.3g})"
    if bf < 30:
        return f"strong evidence for model 1 (BF={bf:.3g})"
    if bf < 100:
        return f"very strong evidence for model 1 (BF={bf:.3g})"
    return f"decisive evidence for model 1 (BF={bf:.3g})"


def harmonic_mean_log_evidence(log_likelihoods: List[float]) -> float:
    """The harmonic mean estimator of the log marginal likelihood:

        P(D) ~ [ (1/N) * sum_i 1/L(theta_i) ]^{-1},  theta_i ~ posterior

    which rearranges, in log-space, to

        ln P(D) ~ ln(N) - logsumexp(-ln L(theta_i))

    This estimator requires only posterior *samples* (e.g. straight
    from a Metropolis-Hastings run) and no separate integration step,
    which is what makes it tempting; it is included here specifically
    to demonstrate, against a known closed-form answer, why it is
    nonetheless a poor choice (Radford Neal called it "the worst Monte
    Carlo method ever devised" for good reason): the harmonic mean is
    dominated by whichever sampled theta_i happens to have the
    *smallest* likelihood, since a small L(theta_i) makes 1/L(theta_i)
    enormous, and posterior samples occasionally do land in
    low-likelihood tail regions purely by chance. A single unlucky
    draw can swing the entire estimate, giving it infinite variance in
    principle and very high, unstable variance in any finite sample in
    practice.
    """
    if not log_likelihoods:
        raise ValueError("log_likelihoods must be non-empty")
    n = len(log_likelihoods)
    return math.log(n) - log_sum_exp([-ll for ll in log_likelihoods])
