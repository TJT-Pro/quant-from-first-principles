import csv
import math
import random
import statistics
from typing import List, Dict, Optional, Tuple, Set

Matrix = List[List[float]]
Vector = List[float]


# Minimal linear algebra, only what the chain analysis needs.

def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    n, k1 = len(a), len(a[0])
    k2, m = len(b), len(b[0])
    if k1 != k2:
        raise ValueError("incompatible matrix dimensions")
    result = [[0.0] * m for _ in range(n)]
    for i in range(n):
        row = a[i]
        for k in range(k1):
            r = row[k]
            if r == 0.0:
                continue
            brow = b[k]
            for j in range(m):
                result[i][j] += r * brow[j]
    return result


def vector_matrix_multiply(v: Vector, a: Matrix) -> Vector:
    n = len(a)
    m = len(a[0])
    if len(v) != n:
        raise ValueError("incompatible dimensions")
    result = [0.0] * m
    for i in range(n):
        vi = v[i]
        if vi == 0.0:
            continue
        row = a[i]
        for j in range(m):
            result[j] += vi * row[j]
    return result


def identity_matrix(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(a: Matrix) -> Matrix:
    n, m = len(a), len(a[0])
    return [[a[i][j] for i in range(n)] for j in range(m)]


def solve_linear_system(a: Matrix, b: Vector) -> Vector:
    """Solve Ax = b by Gaussian elimination with partial pivoting."""
    n = len(a)
    if len(a[0]) != n or len(b) != n:
        raise ValueError("solve_linear_system requires a square system")

    # Build an augmented matrix and work on a copy.
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-14:
            raise ValueError("matrix is singular or nearly singular")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def matrix_power(p: Matrix, n: int) -> Matrix:
    """P^n by exponentiation by squaring."""
    if n < 0:
        raise ValueError("n must be non-negative")
    size = len(p)
    result = identity_matrix(size)
    base = [row[:] for row in p]
    while n > 0:
        if n & 1:
            result = matrix_multiply(result, base)
        base = matrix_multiply(base, base)
        n >>= 1
    return result


# Chain validation and simulation.

def validate_transition_matrix(p: Matrix, tol: float = 1e-8) -> None:
    n = len(p)
    if n == 0:
        raise ValueError("transition matrix must be non-empty")
    for row in p:
        if len(row) != n:
            raise ValueError("transition matrix must be square")
        if any(x < -tol for x in row):
            raise ValueError("transition matrix has negative entries")
        if abs(sum(row) - 1.0) > tol:
            raise ValueError("each row of a transition matrix must sum to 1")


def simulate_chain(p: Matrix, initial_state: int, n_steps: int,
                    seed: Optional[int] = None) -> List[int]:
    validate_transition_matrix(p)
    n = len(p)
    if not 0 <= initial_state < n:
        raise ValueError("initial_state out of range")
    rng = random.Random(seed)
    states = [initial_state]
    current = initial_state
    for _ in range(n_steps):
        r = rng.random()
        cumulative = 0.0
        next_state = n - 1
        for j in range(n):
            cumulative += p[current][j]
            if r < cumulative:
                next_state = j
                break
        states.append(next_state)
        current = next_state
    return states


# Structural properties: reachability, irreducibility, periodicity.

def reachable_states(p: Matrix, start: int) -> Set[int]:
    n = len(p)
    visited = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in range(n):
            if p[u][v] > 0 and v not in visited:
                visited.add(v)
                stack.append(v)
    return visited


def is_irreducible(p: Matrix) -> bool:
    n = len(p)
    for s in range(n):
        if len(reachable_states(p, s)) != n:
            return False
    return True


def communicating_classes(p: Matrix) -> List[Set[int]]:
    n = len(p)
    forward = [reachable_states(p, s) for s in range(n)]
    classes: List[Set[int]] = []
    assigned: Set[int] = set()
    for s in range(n):
        if s in assigned:
            continue
        cls = {t for t in forward[s] if s in forward[t]}
        classes.append(cls)
        assigned |= cls
    return classes


def period_of_state(p: Matrix, state: int) -> int:
    """Period of the communicating class containing `state`, via the gcd
    of (distance(state,u) + 1 - distance(state,v)) over all edges u->v
    with both endpoints reachable from `state`.
    """
    n = len(p)
    dist: Dict[int, int] = {state: 0}
    order = [state]
    head = 0
    while head < len(order):
        u = order[head]
        head += 1
        for v in range(n):
            if p[u][v] > 0 and v not in dist:
                dist[v] = dist[u] + 1
                order.append(v)

    period = 0
    for u in dist:
        for v in range(n):
            if p[u][v] > 0 and v in dist:
                candidate = dist[u] + 1 - dist[v]
                period = math.gcd(period, abs(candidate))
    return period if period > 0 else 1


def is_aperiodic(p: Matrix) -> bool:
    n = len(p)
    return all(period_of_state(p, s) == 1 for s in range(n))


# Stationary distribution.

def stationary_distribution_linear(p: Matrix) -> Vector:
    """Solve pi P = pi, sum(pi) = 1 by replacing one balance equation
    with the normalisation constraint and solving the resulting linear
    system directly.
    """
    validate_transition_matrix(p)
    n = len(p)
    pt = transpose(p)
    a = [[pt[i][j] - (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    a[-1] = [1.0] * n
    b = [0.0] * (n - 1) + [1.0]
    return solve_linear_system(a, b)


def stationary_distribution_power_iteration(p: Matrix, tol: float = 1e-12,
                                             max_iter: int = 100000) -> Vector:
    validate_transition_matrix(p)
    n = len(p)
    dist = [1.0 / n] * n
    for _ in range(max_iter):
        nxt = vector_matrix_multiply(dist, p)
        diff = sum(abs(nxt[i] - dist[i]) for i in range(n))
        dist = nxt
        if diff < tol:
            break
    return dist


# Convergence to stationarity.

def total_variation_distance(p: Vector, q: Vector) -> float:
    if len(p) != len(q):
        raise ValueError("distributions must have the same length")
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))


def convergence_trace(p: Matrix, initial_dist: Vector, n_max: int) -> List[float]:
    stationary = stationary_distribution_linear(p)
    dist = initial_dist[:]
    trace = []
    for _ in range(n_max):
        dist = vector_matrix_multiply(dist, p)
        trace.append(total_variation_distance(dist, stationary))
    return trace


def estimate_convergence_rate(trace: List[float]) -> Optional[float]:
    """Fit log(TV distance) ~ slope * n by least squares on the points
    where the distance is still numerically meaningful, returning the
    slope (negative for genuine convergence). Returns None if fewer
    than two usable points remain.
    """
    points = [(n, math.log(d)) for n, d in enumerate(trace, start=1) if d > 1e-12]
    if len(points) < 2:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in points)
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


# First-passage and return times.

def mean_first_passage_times(p: Matrix, target: int) -> Vector:
    """h_i = expected number of steps to reach `target` starting from i.

    Solves h_i = 1 + sum_{k != target} P_ik h_k for i != target, with
    h_target = 0, by Gaussian elimination on the reduced (n-1)x(n-1)
    system.
    """
    n = len(p)
    others = [i for i in range(n) if i != target]
    idx = {state: pos for pos, state in enumerate(others)}
    m = len(others)

    a = [[0.0] * m for _ in range(m)]
    b = [1.0] * m
    for row_pos, i in enumerate(others):
        a[row_pos][row_pos] = 1.0
        for k in others:
            a[row_pos][idx[k]] -= p[i][k]

    h_reduced = solve_linear_system(a, b)
    h = [0.0] * n
    for pos, i in enumerate(others):
        h[i] = h_reduced[pos]
    h[target] = 0.0
    return h


def expected_return_time(stationary: Vector, state: int) -> float:
    """Kac's formula: for an irreducible positive-recurrent chain, the
    expected return time to state i is exactly 1 / pi_i.
    """
    if stationary[state] <= 0:
        raise ValueError("state has zero stationary probability")
    return 1.0 / stationary[state]


# Regularised incomplete gamma function, for the chi-square test used in
# the Markov-property application below. Implemented from the classic
# series/continued-fraction split (Numerical Recipes algorithm); math.lgamma
# supplies the log-gamma term.

def _gamma_series(a: float, x: float, itmax: int = 500, eps: float = 3e-14) -> float:
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(itmax):
        ap += 1
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_continued_fraction(a: float, x: float, itmax: int = 500,
                               eps: float = 3e-14, fpmin: float = 1e-300) -> float:
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def regularized_gamma_q(a: float, x: float) -> float:
    """Upper regularised incomplete gamma function Q(a, x)."""
    if x < 0 or a <= 0:
        raise ValueError("require a > 0 and x >= 0")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_continued_fraction(a, x)


def chi2_sf(x: float, df: int) -> float:
    """Survival function P(X > x) for a chi-square distribution with
    df degrees of freedom.
    """
    if x <= 0:
        return 1.0
    return regularized_gamma_q(df / 2.0, x / 2.0)


# Application: discretised EUR/USD returns as a Markov chain.

def load_returns(csv_path: str, column: str = "return") -> List[float]:
    values = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    return values


def sign_states(returns: List[float], seed: Optional[int] = None) -> List[int]:
    """Map returns to a 2-state chain: 0 = down, 1 = up."""
    rng = random.Random(seed)
    states = []
    for r in returns:
        if r > 0:
            states.append(1)
        elif r < 0:
            states.append(0)
        else:
            states.append(1 if rng.random() < 0.5 else 0)
    return states


def quantile_breakpoints(data: List[float], n_bins: int) -> List[float]:
    """n_bins - 1 breakpoints splitting `data` into equal-frequency bins,
    using linear interpolation between order statistics (the same
    convention as the default method used by most statistical
    software).
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    ordered = sorted(data)
    n = len(ordered)
    breakpoints = []
    for k in range(1, n_bins):
        q = k / n_bins
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        breakpoints.append(ordered[lo] + frac * (ordered[hi] - ordered[lo]))
    return breakpoints


def assign_bins(data: List[float], breakpoints: List[float]) -> List[int]:
    states = []
    for x in data:
        bin_index = 0
        for bp in breakpoints:
            if x > bp:
                bin_index += 1
            else:
                break
        states.append(bin_index)
    return states


def empirical_transition_matrix(states: List[int], n_states: int) -> Matrix:
    counts = [[0] * n_states for _ in range(n_states)]
    for i in range(len(states) - 1):
        counts[states[i]][states[i + 1]] += 1
    matrix = []
    for row in counts:
        total = sum(row)
        if total == 0:
            matrix.append([1.0 / n_states] * n_states)
        else:
            matrix.append([c / total for c in row])
    return matrix


def markov_independence_test(states: List[int], n_states: int) -> Dict[str, float]:
    """Chi-square test of independence on the contingency table of
    (current state, next state) pairs. The null hypothesis is that the
    next state is independent of the current one, i.e. the process is
    not first-order Markov-dependent but drawn i.i.d. from its marginal
    distribution.
    """
    counts = [[0] * n_states for _ in range(n_states)]
    for i in range(len(states) - 1):
        counts[states[i]][states[i + 1]] += 1

    row_totals = [sum(row) for row in counts]
    col_totals = [sum(counts[i][j] for i in range(n_states)) for j in range(n_states)]
    grand_total = sum(row_totals)

    statistic = 0.0
    for i in range(n_states):
        for j in range(n_states):
            expected = row_totals[i] * col_totals[j] / grand_total
            if expected > 0:
                statistic += (counts[i][j] - expected) ** 2 / expected

    df = (n_states - 1) ** 2
    p_value = chi2_sf(statistic, df)
    return {"statistic": statistic, "df": df, "p_value": p_value}


def analyze_fx_markov_chain(returns: List[float], n_bins: int = 5,
                             seed: Optional[int] = None) -> Dict:
    """Fit a discretised Markov chain to EUR/USD returns and compare it
    against the empirical marginal distribution, plus a formal test of
    the Markov (first-order dependence) hypothesis.
    """
    breakpoints = quantile_breakpoints(returns, n_bins)
    states = assign_bins(returns, breakpoints)

    transition_matrix = empirical_transition_matrix(states, n_bins)
    stationary = stationary_distribution_linear(transition_matrix)

    empirical_freq = [0.0] * n_bins
    for s in states:
        empirical_freq[s] += 1
    empirical_freq = [c / len(states) for c in empirical_freq]

    independence_test = markov_independence_test(states, n_bins)

    passage_low_to_high = mean_first_passage_times(transition_matrix, n_bins - 1)[0]
    passage_high_to_low = mean_first_passage_times(transition_matrix, 0)[n_bins - 1]

    return {
        "n_bins": n_bins,
        "n_observations": len(returns),
        "breakpoints": breakpoints,
        "transition_matrix": transition_matrix,
        "stationary_distribution": stationary,
        "empirical_frequency": empirical_freq,
        "independence_test": independence_test,
        "mean_passage_low_to_high": passage_low_to_high,
        "mean_passage_high_to_low": passage_high_to_low,
    }
