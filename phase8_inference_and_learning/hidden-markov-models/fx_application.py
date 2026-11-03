import csv
import statistics
from typing import List, Dict

import hmm as hmm_module
import model_selection as ms


def load_log_returns(csv_path: str, column: str = "log_return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def fit_regime_model(returns: List[float], n_states: int = 2, n_restarts: int = 5,
                      seed: int = 0) -> Dict:
    """Fit a Gaussian HMM to daily EUR/USD log returns and label the
    resulting states by volatility (ascending variance), since the
    raw state indices from Baum-Welch carry no inherent order or
    meaning of their own (the well-known "label switching" issue in
    any latent-class model: two runs that converge to the same
    underlying fit can assign states 0 and 1 in either order, or a
    different random seed's initialisation can systematically produce
    the opposite labelling).
    """
    result = ms.fit_best_of_n_restarts(returns, n_states, n_restarts, seed=seed)
    model = result["model"]
    order = sorted(range(n_states), key=lambda i: model.variances[i])
    return {
        "model": model,
        "log_likelihoods": result["log_likelihoods"],
        "state_order_low_to_high_vol": order,
    }


def regime_summary(model: hmm_module.GaussianHMM, viterbi_path: List[int],
                    order: List[int]) -> List[Dict]:
    """Summarise each (volatility-ordered) regime: its fitted mean and
    standard deviation, its long-run (stationary) probability implied
    by the fitted transition matrix, and the fraction of the Viterbi-
    decoded path actually assigned to it, so the fitted model's own
    implied long-run behaviour can be compared against what the
    decoded path actually shows.
    """
    n = model.n_states
    stationary = _stationary_distribution(model.transition)
    path_counts = [0] * n
    for s in viterbi_path:
        path_counts[s] += 1

    summary = []
    for rank, state in enumerate(order):
        summary.append({
            "rank": rank,
            "mean": model.means[state],
            "sd": model.variances[state] ** 0.5,
            "stationary_probability": stationary[state],
            "viterbi_fraction": path_counts[state] / len(viterbi_path),
            "expected_duration_days": 1.0 / (1.0 - model.transition[state][state]),
        })
    return summary


def _stationary_distribution(transition: List[List[float]]) -> List[float]:
    """The stationary distribution of the fitted transition matrix,
    via power iteration (the same underlying idea used for this exact
    purpose in `stochastic-processes/project2_markov_chains`,
    reimplemented here rather than imported, since this project shares
    no code with any sibling repository): repeatedly applying
    pi -> pi*A from a uniform start converges to the stationary
    distribution for an irreducible, aperiodic chain, which a fitted
    regime-switching transition matrix with strictly positive
    self-transition probabilities always is.
    """
    n = len(transition)
    pi = [1.0 / n] * n
    for _ in range(10000):
        new_pi = [sum(pi[i] * transition[i][j] for i in range(n)) for j in range(n)]
        if sum(abs(new_pi[j] - pi[j]) for j in range(n)) < 1e-14:
            pi = new_pi
            break
        pi = new_pi
    return pi


def regime_duration_statistics(viterbi_path: List[int], order: List[int]) -> Dict[int, Dict]:
    """The empirical distribution of how many consecutive days the
    Viterbi-decoded path spends in each (volatility-ordered) regime
    before switching, compared against the fitted model's own implied
    expected duration (1/(1-self-transition-probability), the mean of
    a geometric distribution): a large discrepancy between the two
    would suggest the fitted transition matrix does not actually
    describe the decoded path's real switching behaviour well.
    """
    runs = {state: [] for state in set(viterbi_path)}
    current_state = viterbi_path[0]
    current_length = 1
    for s in viterbi_path[1:]:
        if s == current_state:
            current_length += 1
        else:
            runs[current_state].append(current_length)
            current_state = s
            current_length = 1
    runs[current_state].append(current_length)

    rank_of_state = {state: rank for rank, state in enumerate(order)}
    result = {}
    for state, lengths in runs.items():
        result[rank_of_state[state]] = {
            "n_episodes": len(lengths),
            "mean_duration": statistics.mean(lengths),
            "max_duration": max(lengths),
        }
    return result
