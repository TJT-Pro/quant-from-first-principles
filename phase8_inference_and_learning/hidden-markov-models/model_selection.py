import math
from typing import List, Dict

import hmm as hmm_module


def n_free_parameters(n_states: int) -> int:
    """The number of free parameters in a K-state Gaussian HMM:

        (K-1)      initial distribution (one constraint: sums to 1)
      + K(K-1)     transition matrix (K rows, each summing to 1)
      + K          means
      + K          variances
      = K^2 + 2K - 1

    This is a genuinely different parameter count from an ARMA model's
    or a mixture model's, because the transition matrix contributes
    K*(K-1) free parameters on its own (K separate row-sum-to-1
    constraints, one per row, not a single global constraint the way
    a single categorical distribution like the initial distribution
    has): doubling the number of states roughly quadruples the
    transition matrix's parameter count, which is what makes AIC/BIC
    penalise additional hidden states so much more steeply here than
    in most other model families.
    """
    return n_states ** 2 + 2 * n_states - 1


def fit_best_of_n_restarts(observations: List[float], n_states: int, n_restarts: int = 5,
                            n_iter: int = 200, seed: int = 0) -> Dict:
    """Baum-Welch is only guaranteed to find a *local* maximum of the
    likelihood, and which local maximum it finds depends on the
    (essentially arbitrary) random initialisation; unlike a convex
    problem, there is no shortcut that guarantees the global optimum
    directly. The standard, simple mitigation is exactly what is done
    here: run EM to convergence from several different random starting
    points and keep whichever run reached the highest log-likelihood,
    since a higher achieved likelihood is never worse (for the same
    number of parameters) and different random seeds have a good
    chance of avoiding the same poor local optimum together.
    """
    best_result = None
    for restart in range(n_restarts):
        init_model = hmm_module.random_init(observations, n_states, seed=seed * 1000 + restart)
        result = init_model.fit(observations, n_iter=n_iter)
        if best_result is None or result["log_likelihoods"][-1] > best_result["log_likelihoods"][-1]:
            best_result = result
    return best_result


def aic(log_likelihood: float, n_states: int) -> float:
    return 2 * n_free_parameters(n_states) - 2 * log_likelihood


def bic(log_likelihood: float, n_states: int, n_obs: int) -> float:
    return n_free_parameters(n_states) * math.log(n_obs) - 2 * log_likelihood


def select_n_states(observations: List[float], k_range: List[int], n_restarts: int = 5,
                     n_iter: int = 200, seed: int = 0) -> Dict:
    """Fit a Gaussian HMM for every K in `k_range` (each with several
    random restarts, keeping the best), and score each by AIC/BIC.
    Because adding a state always weakly increases the achievable
    log-likelihood (a K-state HMM can always mimic a (K-1)-state one
    by making an extra state a near-duplicate of an existing one), the
    raw log-likelihood alone would trivially favour the largest K
    tried; AIC/BIC's parameter penalty is what makes the comparison
    meaningful.
    """
    results = []
    n_obs = len(observations)
    for k in k_range:
        fit = fit_best_of_n_restarts(observations, k, n_restarts, n_iter=n_iter, seed=seed)
        ll = fit["log_likelihoods"][-1]
        results.append({
            "n_states": k,
            "log_likelihood": ll,
            "aic": aic(ll, k),
            "bic": bic(ll, k, n_obs),
            "model": fit["model"],
        })
    best_aic = min(results, key=lambda r: r["aic"])
    best_bic = min(results, key=lambda r: r["bic"])
    return {"results": results, "best_aic_k": best_aic["n_states"], "best_bic_k": best_bic["n_states"]}
