import math
import random
import statistics
from typing import Callable, List, Optional, Dict


def metropolis_hastings(log_posterior: Callable[[float], float], initial: float,
                         proposal_sd: float, n_samples: int, burn_in: int = 1000,
                         seed: Optional[int] = None) -> Dict[str, object]:
    """Sample from a distribution known only up to a normalising
    constant, via the Metropolis-Hastings algorithm with a symmetric
    (Gaussian random-walk) proposal.

    At each step, propose theta' ~ N(theta, proposal_sd^2), and accept
    it with probability min(1, p(theta')/p(theta)) where p is the
    (possibly unnormalised) target density. Because the proposal is
    symmetric (the probability of proposing theta'->theta equals that
    of theta->theta'), the proposal densities cancel out of the
    Metropolis-Hastings acceptance ratio entirely, leaving just the
    target-density ratio; the crucial point is that this ratio never
    requires the normalising constant, since it cancels between
    numerator and denominator. Working in log-space
    (log_posterior(theta') - log_posterior(theta), compared against
    log(uniform(0,1))) avoids ever exponentiating a potentially very
    large or very small log-density directly.

    This construction, in the limit of infinitely many steps, produces
    samples from the target distribution regardless of where the chain
    started, because the resulting Markov chain (an object from
    `stochastic-processes`' own subject matter, though not imported
    from there) can be shown to have the target distribution as its
    unique stationary distribution and to be both irreducible and
    aperiodic under mild conditions on the proposal and target.
    """
    rng = random.Random(seed)
    current = initial
    current_log_p = log_posterior(current)
    samples = []
    accepted = 0
    total_steps = n_samples + burn_in

    for step in range(total_steps):
        proposal = current + rng.gauss(0, proposal_sd)
        proposal_log_p = log_posterior(proposal)
        log_accept_ratio = proposal_log_p - current_log_p
        if log_accept_ratio >= 0 or math.log(rng.random()) < log_accept_ratio:
            current = proposal
            current_log_p = proposal_log_p
            if step >= burn_in:
                accepted += 1
        if step >= burn_in:
            samples.append(current)

    return {"samples": samples, "acceptance_rate": accepted / n_samples}


def metropolis_hastings_vector(log_posterior: Callable[[List[float]], float], initial: List[float],
                                proposal_sds: List[float], n_samples: int, burn_in: int = 1000,
                                seed: Optional[int] = None) -> Dict[str, object]:
    """The multi-parameter generalisation of `metropolis_hastings`:
    each coordinate gets its own independent Gaussian random-walk
    proposal (a diagonal proposal covariance), and the whole proposed
    vector is accepted or rejected together based on the joint
    log-posterior ratio. This is needed for genuinely non-conjugate
    problems with more than one unknown parameter, such as jointly
    estimating a Student-t distribution's degrees of freedom and scale
    (see the EUR/USD application), where no conjugate prior exists for
    either parameter individually, let alone jointly.
    """
    rng = random.Random(seed)
    dim = len(initial)
    current = list(initial)
    current_log_p = log_posterior(current)
    samples = []
    accepted = 0
    total_steps = n_samples + burn_in

    for step in range(total_steps):
        proposal = [current[i] + rng.gauss(0, proposal_sds[i]) for i in range(dim)]
        proposal_log_p = log_posterior(proposal)
        log_accept_ratio = proposal_log_p - current_log_p
        if log_accept_ratio >= 0 or math.log(rng.random()) < log_accept_ratio:
            current = proposal
            current_log_p = proposal_log_p
            if step >= burn_in:
                accepted += 1
        if step >= burn_in:
            samples.append(list(current))

    return {"samples": samples, "acceptance_rate": accepted / n_samples}


def autocorrelation_of_chain(samples: List[float], lag: int) -> float:
    n = len(samples)
    mean = statistics.mean(samples)
    denom = sum((s - mean) ** 2 for s in samples)
    if denom == 0:
        return 0.0
    numer = sum((samples[t] - mean) * (samples[t + lag] - mean) for t in range(n - lag))
    return numer / denom


def effective_sample_size(samples: List[float]) -> float:
    """The effective number of independent samples represented by a
    correlated MCMC chain of length n:

        ESS = n / (1 + 2 * sum_{k=1}^{K} rho_k)

    where rho_k is the chain's own lag-k autocorrelation. Positively
    autocorrelated samples (the usual case for a random-walk proposal,
    since consecutive draws are close together by construction) waste
    information relative to truly independent draws, and this formula
    quantifies exactly how much: a chain with rho_k decaying slowly
    has a large sum and a correspondingly small ESS relative to its
    raw length n. The sum is truncated using Geyer's initial positive
    sequence rule: stop as soon as a pair of consecutive
    autocorrelations sums to a negative value, since autocorrelation
    estimates from a finite chain become unreliable (and can spuriously
    alternate in sign) far out in the lag, and including that noise
    would make the ESS estimate itself noisy or even negative.
    """
    n = len(samples)
    total = 0.0
    k = 1
    while k + 1 < n:
        rho_k = autocorrelation_of_chain(samples, k)
        rho_k1 = autocorrelation_of_chain(samples, k + 1)
        if rho_k + rho_k1 < 0:
            break
        total += rho_k + rho_k1
        k += 2
    denom = 1 + 2 * total
    return n / denom if denom > 0 else float(n)


def gelman_rubin(chains: List[List[float]]) -> float:
    """The Gelman-Rubin R-hat convergence diagnostic: run several
    independent chains from different (ideally overdispersed) starting
    points, and compare the variance *within* each chain to the
    variance *between* the chains' means. If the chains have all
    converged to the same target distribution, the two should roughly
    agree (R-hat close to 1); if they have not (stuck in different
    regions, or one still drifting toward the true target), the
    between-chain variance will be inflated relative to the within-
    chain variance, pushing R-hat above 1. Values noticeably above 1.1
    are the standard rule-of-thumb warning sign that more burn-in or
    more samples are needed before trusting the chains.

        W = mean of the m chains' individual sample variances
        B/n = variance of the m chains' individual sample means
        R-hat = sqrt( ((n-1)/n * W + B/n) / W )
    """
    m = len(chains)
    if m < 2:
        raise ValueError("need at least two chains to compute R-hat")
    n = len(chains[0])
    if any(len(c) != n for c in chains):
        raise ValueError("all chains must have the same length")

    chain_means = [statistics.mean(c) for c in chains]
    chain_vars = [statistics.variance(c) for c in chains]
    w = statistics.mean(chain_vars)
    grand_mean = statistics.mean(chain_means)
    b_over_n = sum((m_i - grand_mean) ** 2 for m_i in chain_means) / (m - 1)

    var_estimate = ((n - 1) / n) * w + b_over_n
    if w == 0:
        return float("inf") if var_estimate > 0 else 1.0
    return math.sqrt(var_estimate / w)
