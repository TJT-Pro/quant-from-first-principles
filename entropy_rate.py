import math
from typing import Dict, List, Sequence

import entropy as ent


def stationary_distribution(transition_matrix: List[List[float]], tol: float = 1e-14,
                             max_iter: int = 100000) -> List[float]:
    """The stationary distribution of a Markov chain's transition
    matrix via power iteration: repeatedly applying pi -> pi*P from a
    uniform starting distribution converges to the unique stationary
    distribution for any irreducible, aperiodic chain, since P's
    largest eigenvalue is exactly 1 (rows sum to 1) with the
    stationary distribution as its corresponding left eigenvector, and
    every other eigenvalue has magnitude strictly less than 1 for such
    a chain, so repeated multiplication damps out every other
    component of an arbitrary starting vector.
    """
    n = len(transition_matrix)
    pi = [1.0 / n] * n
    for _ in range(max_iter):
        new_pi = [sum(pi[i] * transition_matrix[i][j] for i in range(n)) for j in range(n)]
        if sum(abs(new_pi[j] - pi[j]) for j in range(n)) < tol:
            return new_pi
        pi = new_pi
    return pi


def exact_markov_entropy_rate(transition_matrix: List[List[float]], base: float = 2.0) -> float:
    """The exact entropy rate of a stationary Markov chain,
    H_rate = sum_i pi_i * H(P_i.), where H(P_i.) is the entropy of row
    i of the transition matrix (the conditional entropy of the next
    state given that the chain is currently in state i) and pi is the
    chain's own stationary distribution. This equals H(X_2|X_1) for
    the chain run in its stationary regime, and (by the general fact
    that a Markov chain's entropy rate equals its first-order
    conditional entropy, a consequence of the Markov property making
    every earlier state irrelevant once the immediately preceding one
    is known) it is also exactly the limiting rate of growth of block
    entropy H(X_1,...,X_n)/n as n grows, giving an exact target to
    validate the empirical block-entropy estimator against.
    """
    pi = stationary_distribution(transition_matrix)
    total = 0.0
    for i, row in enumerate(transition_matrix):
        row_entropy = ent.entropy({j: p for j, p in enumerate(row)}, base)
        total += pi[i] * row_entropy
    return total


def simulate_markov_chain(transition_matrix: List[List[float]], n_steps: int, seed: int = None,
                           start_state: int = 0) -> List[int]:
    import random
    rng = random.Random(seed)
    state = start_state
    sequence = [state]
    for _ in range(n_steps):
        row = transition_matrix[state]
        state = rng.choices(range(len(row)), weights=row)[0]
        sequence.append(state)
    return sequence


def block_entropy_rate_estimate(sequence: Sequence, block_length: int, base: float = 2.0) -> float:
    """Estimate the entropy rate as H(X_1,...,X_k)/k: the plug-in
    empirical entropy of overlapping length-k blocks from the
    sequence, divided by k. As k grows, this converges to the true
    entropy rate (the Shannon-McMillan-Breiman theorem's rate-of-growth
    statement), but the plug-in estimator's own finite-sample bias
    (see entropy.entropy_from_counts) grows with the block alphabet
    size k^(alphabet size), so in practice block_length cannot be
    pushed arbitrarily high without a correspondingly large sample;
    this project instead uses `conditional_entropy_rate_estimate`
    below (relying on the Markov property to jump straight to the
    n->infinity limit at order 1) as the more sample-efficient
    estimator whenever the underlying process is actually first-order
    Markov, keeping this direct block-entropy estimator mainly for
    validating that convergence trend directly rather than as the
    primary estimation tool.
    """
    n = len(sequence)
    blocks = [tuple(sequence[i:i + block_length]) for i in range(n - block_length + 1)]
    return ent.entropy_from_counts(blocks, base) / block_length


def conditional_entropy_rate_estimate(sequence: Sequence, order: int = 1, base: float = 2.0) -> float:
    """Estimate the entropy rate as H(X_n | X_{n-order},...,X_{n-1}),
    the empirical conditional entropy of each symbol given the
    preceding `order` symbols. For a genuinely order-th-order Markov
    process, this converges to the exact entropy rate directly (no
    need to grow the block length toward infinity the way
    block_entropy_rate_estimate does), since conditioning on exactly
    the process's own memory length already captures everything
    relevant to predicting the next symbol; for a process with longer
    memory than `order` captures, this instead estimates an upper
    bound on the true entropy rate (conditioning on less information
    than the process actually depends on can only ever make the
    remaining uncertainty look larger, never smaller, by the general
    fact that conditioning on more information never increases
    entropy).
    """
    pairs = [(tuple(sequence[i:i + order]), sequence[i + order]) for i in range(len(sequence) - order)]
    return ent.conditional_entropy(pairs, base)
