import csv
import math
import random
import statistics
from typing import List, Tuple, Optional, Dict


def simulate_walk_1d(n_steps: int, p: float = 0.5, seed: Optional[int] = None) -> List[int]:
    """Simulate a 1D simple random walk with step +1 w.p. p and -1 w.p. 1-p.

    Returns the path including the starting position S_0 = 0, so the
    returned list has length n_steps + 1.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")

    rng = random.Random(seed)
    path = [0]
    position = 0
    for _ in range(n_steps):
        step = 1 if rng.random() < p else -1
        position += step
        path.append(position)
    return path


def simulate_walk_2d(n_steps: int, seed: Optional[int] = None) -> List[Tuple[int, int]]:
    """Simulate a 2D simple random walk on Z^2 with four equally likely
    nearest-neighbour moves. Returns the path including the origin.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    rng = random.Random(seed)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    path = [(0, 0)]
    x, y = 0, 0
    for _ in range(n_steps):
        dx, dy = rng.choice(moves)
        x += dx
        y += dy
        path.append((x, y))
    return path


def theoretical_mean_1d(n: int, p: float = 0.5) -> float:
    """E[S_n] for the 1D walk with step +1 w.p. p, -1 w.p. 1-p."""
    return n * (2 * p - 1)


def theoretical_variance_1d(n: int, p: float = 0.5) -> float:
    """Var(S_n) for the 1D walk. Each step has variance 4p(1-p); steps
    are independent so variances add.
    """
    return 4 * n * p * (1 - p)


def theoretical_std_1d(n: int, p: float = 0.5) -> float:
    return math.sqrt(theoretical_variance_1d(n, p))


def prob_Sn_equals(n: int, m: int, p: float = 0.5) -> float:
    """Exact probability that S_n = m for the 1D walk.

    S_n = m requires k up-steps and n-k down-steps with m = 2k - n, so
    k = (n + m) / 2. This must be a non-negative integer no larger than n,
    otherwise the event has probability zero.
    """
    if (n + m) % 2 != 0:
        return 0.0
    k = (n + m) // 2
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def prob_Sn_at_least(n: int, m: int, p: float = 0.5) -> float:
    """P(S_n >= m), summed exactly over the reachable lattice of values."""
    start = m
    if (n + start) % 2 != 0:
        start += 1
    total = 0.0
    value = start
    while value <= n:
        total += prob_Sn_equals(n, value, p)
        value += 2
    return total


def prob_Sn_at_most(n: int, m: int, p: float = 0.5) -> float:
    """P(S_n <= m)."""
    return 1.0 - prob_Sn_at_least(n, m + 1, p)


def reflection_principle_max_at_least(n: int, a: int, p: float = 0.5) -> float:
    """P(max_{0<=k<=n} S_k >= a) for the symmetric simple random walk,
    derived via the reflection principle.

    For a path that touches level a at some time k <= n, reflecting the
    portion of the path after the first hitting time about the line
    y = a maps bijectively onto paths ending strictly above a, when
    a > 0. This bijection gives:

        P(M_n >= a) = P(S_n >= a) + P(S_n > a)

    which only holds for the symmetric walk (p = 0.5), because the
    reflection preserves probability only when up- and down-steps are
    equally likely.
    """
    if a <= 0:
        raise ValueError("reflection principle formula requires a > 0")
    if p != 0.5:
        raise ValueError("closed-form reflection principle requires p = 0.5")
    return prob_Sn_at_least(n, a, p) + prob_Sn_at_least(n, a + 1, p)


def reflection_principle_max_at_least_bruteforce(n: int, a: int) -> float:
    """Exact brute-force computation of P(max_{0<=k<=n} S_k >= a) for the
    symmetric walk by enumerating all 2^n sign sequences. Intended for
    verification against small n only.
    """
    if a <= 0:
        raise ValueError("a must be positive")
    if n > 22:
        raise ValueError("brute force enumeration is impractical for n > 22")

    count = 0
    total = 1 << n
    for mask in range(total):
        position = 0
        running_max = 0
        for bit in range(n):
            step = 1 if (mask >> bit) & 1 else -1
            position += step
            if position > running_max:
                running_max = position
        if running_max >= a:
            count += 1
    return count / total


def first_passage_pmf(n: int, a: int, p: float = 0.5) -> float:
    """P(T_a = n): probability that the first time the walk hits level
    a > 0 is exactly step n.

    This is the Hitting Time Theorem (a consequence of the cycle lemma,
    sometimes attributed to Otter or Kemperman): for iid +-1 increments,

        P(T_a = n) = (a / n) * P(S_n = a),   n >= a, n - a even.

    The result holds for any p, not only p = 0.5, because it follows
    from a cyclic symmetry argument on the increments rather than from
    reflection.
    """
    if a <= 0:
        raise ValueError("a must be positive")
    if n < a:
        return 0.0
    if (n - a) % 2 != 0:
        return 0.0
    return (a / n) * prob_Sn_equals(n, a, p)


def hitting_probability_ever(a: int, p: float = 0.5) -> float:
    """P(the walk ever reaches level a > 0), started at 0.

    For p >= 0.5 the walk has non-negative drift and reaches any level
    with probability 1. For p < 0.5 the classical gambler's-ruin
    argument with an infinite far boundary gives P = (p / (1-p))^a.
    """
    if a <= 0:
        raise ValueError("a must be positive")
    if p >= 0.5:
        return 1.0
    return (p / (1 - p)) ** a


def expected_hitting_time_biased(a: int, p: float) -> float:
    """E[T_a] for a biased walk hitting level a > 0.

    By Wald's identity, E[S_T] = E[T] * E[step] whenever T is a stopping
    time with finite expectation. Setting E[S_T] = a and E[step] = 2p-1
    gives E[T_a] = a / (2p - 1) when p > 0.5. For p <= 0.5 the walk
    either never reaches a (p < 0.5) or reaches it only in infinite
    expected time (p = 0.5, a null-recurrent walk), so we return
    infinity in both cases.
    """
    if a <= 0:
        raise ValueError("a must be positive")
    mu = 2 * p - 1
    if mu <= 0:
        return math.inf
    return a / mu


def simulate_first_passage_time(a: int, p: float = 0.5, max_steps: int = 100000,
                                 seed: Optional[int] = None) -> Optional[int]:
    """Simulate a single walk and return the first n with S_n = a, or
    None if level a is not reached within max_steps.
    """
    if a <= 0:
        raise ValueError("a must be positive")
    rng = random.Random(seed)
    position = 0
    for step_index in range(1, max_steps + 1):
        position += 1 if rng.random() < p else -1
        if position == a:
            return step_index
    return None


def estimate_return_probability_2d(n_steps: int, n_trials: int, seed: Optional[int] = None) -> float:
    """Estimate P(the 2D walk returns to the origin at least once within
    n_steps), by Monte Carlo. Polya's recurrence theorem states this
    probability tends to 1 as n_steps -> infinity for the 2D simple
    random walk (recurrent), in contrast to 3D and higher (transient).
    """
    rng = random.Random(seed)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    returns = 0
    for _ in range(n_trials):
        x, y = 0, 0
        returned = False
        for _ in range(n_steps):
            dx, dy = moves[rng.randrange(4)]
            x += dx
            y += dy
            if x == 0 and y == 0:
                returned = True
                break
        if returned:
            returns += 1
    return returns / n_trials


def displacement_2d(path: List[Tuple[int, int]]) -> List[float]:
    """Euclidean distance from the origin at each step of a 2D path."""
    return [math.hypot(x, y) for x, y in path]


def load_returns(csv_path: str, column: str = "return") -> List[float]:
    """Load a numeric column from the shared EUR/USD returns CSV."""
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def sign_sequence(returns: List[float], seed: Optional[int] = None) -> List[int]:
    """Map a sequence of returns to +-1 steps by sign. A return of
    exactly zero (never observed in this dataset, but handled for
    robustness) is broken by an unbiased coin flip.
    """
    rng = random.Random(seed)
    signs = []
    for r in returns:
        if r > 0:
            signs.append(1)
        elif r < 0:
            signs.append(-1)
        else:
            signs.append(1 if rng.random() < 0.5 else -1)
    return signs


def empirical_up_probability(returns: List[float], seed: Optional[int] = None) -> float:
    signs = sign_sequence(returns, seed=seed)
    return sum(1 for s in signs if s == 1) / len(signs)


def cumulative_walk(signs: List[int]) -> List[int]:
    """Cumulative sum of a +-1 sequence, including the starting 0."""
    path = [0]
    total = 0
    for s in signs:
        total += s
        path.append(total)
    return path


def fx_windowed_analysis(returns: List[float], window: int = 252,
                          seed: Optional[int] = None) -> Dict[str, float]:
    """Treat the sign of each daily EUR/USD return as a +-1 random-walk
    step, split the series into non-overlapping windows, and compare
    the empirical behaviour of the resulting walk against the theory
    of a simple random walk.

    The reflection-principle comparison uses p = 0.5 as the null
    hypothesis of a fair, memoryless up/down process (consistent with
    the weak-form efficient-market view that daily FX direction is
    unpredictable), regardless of the small empirical drift in p_hat.
    """
    if window <= 0 or window > len(returns):
        raise ValueError("window must be positive and no larger than the data")

    signs = sign_sequence(returns, seed=seed)
    p_hat = sum(1 for s in signs if s == 1) / len(signs)

    n_windows = len(signs) // window
    finals = []
    maxima = []
    for i in range(n_windows):
        segment = signs[i * window:(i + 1) * window]
        path = cumulative_walk(segment)
        finals.append(path[-1])
        maxima.append(max(path))

    theoretical_mean = theoretical_mean_1d(window, p_hat)
    theoretical_var = theoretical_variance_1d(window, p_hat)
    empirical_mean = statistics.mean(finals)
    empirical_var = statistics.variance(finals) if len(finals) > 1 else 0.0

    threshold = max(1, round(2 * theoretical_std_1d(window, 0.5)))
    theoretical_p_exceed = reflection_principle_max_at_least(window, threshold, 0.5)
    empirical_p_exceed = sum(1 for m in maxima if m >= threshold) / n_windows

    return {
        "n_observations": len(returns),
        "window": window,
        "n_windows": n_windows,
        "p_hat": p_hat,
        "theoretical_mean": theoretical_mean,
        "empirical_mean": empirical_mean,
        "theoretical_var": theoretical_var,
        "empirical_var": empirical_var,
        "threshold": threshold,
        "theoretical_p_exceed": theoretical_p_exceed,
        "empirical_p_exceed": empirical_p_exceed,
    }
