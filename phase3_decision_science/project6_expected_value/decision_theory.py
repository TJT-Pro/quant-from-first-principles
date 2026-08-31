import math


def _check_probabilities(probabilities):
    for p in probabilities:
        if p < 0.0:
            raise ValueError("probabilities cannot be negative")
    total = sum(probabilities)
    if abs(total - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to 1, they sum to {:.10f}".format(total))


def expected_value(outcomes, probabilities):
    """Probability-weighted average of the outcomes, E[X]."""
    if len(outcomes) != len(probabilities):
        raise ValueError("outcomes and probabilities must match in length")
    if not outcomes:
        raise ValueError("need at least one outcome")
    _check_probabilities(probabilities)
    return sum(p * x for p, x in zip(probabilities, outcomes))


def expected_function(outcomes, probabilities, g):
    """E[g(X)] for a function g, by definition."""
    if len(outcomes) != len(probabilities):
        raise ValueError("outcomes and probabilities must match in length")
    if not outcomes:
        raise ValueError("need at least one outcome")
    _check_probabilities(probabilities)
    return sum(p * g(x) for p, x in zip(probabilities, outcomes))


def variance(outcomes, probabilities):
    """Variance of a discrete distribution, E[(X - E[X])^2]."""
    mu = expected_value(outcomes, probabilities)
    return sum(p * (x - mu) ** 2 for p, x in zip(probabilities, outcomes))


def stddev(outcomes, probabilities):
    """Standard deviation of a discrete distribution."""
    return math.sqrt(variance(outcomes, probabilities))


# ---------------------------------------------------------------------------
# Decision rules for a payoff matrix
# ---------------------------------------------------------------------------

def _check_payoffs(payoffs):
    if not payoffs:
        raise ValueError("need at least one action")
    cols = len(payoffs[0])
    if cols < 1:
        raise ValueError("need at least one state")
    for row in payoffs:
        if len(row) != cols:
            raise ValueError("all actions must have the same number of states")


def regret_matrix(payoffs):
    """For each state, how much is lost by choosing each action."""
    _check_payoffs(payoffs)
    cols = len(payoffs[0])
    best = [max(payoffs[i][j] for i in range(len(payoffs))) for j in range(cols)]
    return [[best[j] - payoffs[i][j] for j in range(cols)]
            for i in range(len(payoffs))]


def decision_scores(payoffs, rule="expected-value", probabilities=None,
                    alpha=0.5):
    """Score every action under the chosen decision rule.

    rules: expected-value, maximax, maximin, minimax-regret, laplace,
    hurwicz.  For minimax-regret the score is the maximum regret, which
    is minimized; every other score is maximized.  expected-value needs
    the state probabilities.
    """
    _check_payoffs(payoffs)
    if rule == "expected-value":
        if probabilities is None:
            raise ValueError("expected-value rule needs state probabilities")
        if len(probabilities) != len(payoffs[0]):
            raise ValueError("probabilities must match the number of states")
        _check_probabilities(probabilities)
        return [sum(p * payoff for p, payoff in zip(probabilities, row))
                for row in payoffs]
    if rule == "maximax":
        return [max(row) for row in payoffs]
    if rule == "maximin":
        return [min(row) for row in payoffs]
    if rule == "laplace":
        return [sum(row) / len(row) for row in payoffs]
    if rule == "hurwicz":
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        return [alpha * max(row) + (1.0 - alpha) * min(row) for row in payoffs]
    if rule == "minimax-regret":
        regrets = regret_matrix(payoffs)
        return [max(row) for row in regrets]
    raise ValueError("unknown rule {!r}".format(rule))


def best_action(payoffs, rule="expected-value", probabilities=None, alpha=0.5):
    """Index of the action the rule recommends.

    Returns (index, score).  For minimax-regret the best action is the
    one with the smallest maximum regret; for the other rules the one
    with the largest score.
    """
    scores = decision_scores(payoffs, rule, probabilities, alpha)
    if rule == "minimax-regret":
        best = min(range(len(scores)), key=lambda i: scores[i])
    else:
        best = max(range(len(scores)), key=lambda i: scores[i])
    return best, scores[best]


def dominates(action_a, action_b, payoffs):
    """True if action a beats action b in every state and ties nowhere.

    A dominated action can be discarded before any rule is applied,
    because no reasonable rule ever picks it.
    """
    _check_payoffs(payoffs)
    if action_a >= len(payoffs) or action_b >= len(payoffs):
        raise ValueError("action index out of range")
    a = payoffs[action_a]
    b = payoffs[action_b]
    better = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            better = True
    return better


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class Utility:
    """A utility function with its exact inverse and risk measures."""

    def __init__(self, name):
        self.name = name

    def u(self, x):
        raise NotImplementedError

    def inverse(self, y):
        raise NotImplementedError

    def absolute_risk(self, x):
        raise NotImplementedError

    def relative_risk(self, x):
        raise NotImplementedError


class _Linear(Utility):
    def __init__(self):
        super().__init__("linear")

    def u(self, x):
        return x

    def inverse(self, y):
        return y

    def absolute_risk(self, x):
        return 0.0

    def relative_risk(self, x):
        return 0.0


class _Log(Utility):
    def __init__(self):
        super().__init__("log")

    def u(self, x):
        if x <= 0.0:
            raise ValueError("log utility needs positive outcomes")
        return math.log(x)

    def inverse(self, y):
        return math.exp(y)

    def absolute_risk(self, x):
        return 1.0 / x

    def relative_risk(self, x):
        return 1.0


class _Power(Utility):
    def __init__(self, gamma):
        if not 0.0 < gamma <= 1.0:
            raise ValueError("power utility needs gamma strictly between 0 and 1")
        super().__init__("power(gamma={})".format(gamma))
        self.gamma = gamma

    def u(self, x):
        if x <= 0.0:
            raise ValueError("power utility needs positive outcomes")
        return x ** self.gamma

    def inverse(self, y):
        return y ** (1.0 / self.gamma)

    def absolute_risk(self, x):
        return (1.0 - self.gamma) / x

    def relative_risk(self, x):
        return 1.0 - self.gamma


class _Exponential(Utility):
    def __init__(self, alpha):
        if alpha <= 0.0:
            raise ValueError("exponential utility needs positive alpha")
        super().__init__("exponential(alpha={})".format(alpha))
        self.alpha = alpha

    def u(self, x):
        return 1.0 - math.exp(-self.alpha * x)

    def inverse(self, y):
        if y >= 1.0:
            raise ValueError("utility value outside the domain of the inverse")
        return -math.log(1.0 - y) / self.alpha

    def absolute_risk(self, x):
        return self.alpha

    def relative_risk(self, x):
        return self.alpha * x


class _Quadratic(Utility):
    def __init__(self, b):
        if b <= 0.0:
            raise ValueError("quadratic utility needs positive b")
        super().__init__("quadratic(b={})".format(b))
        self.b = b

    def u(self, x):
        if x >= 1.0 / (2.0 * self.b):
            raise ValueError("quadratic utility is only increasing below 1/(2b)")
        return x - self.b * x * x

    def inverse(self, y):
        disc = 1.0 - 4.0 * self.b * y
        if disc < 0.0:
            raise ValueError("utility value outside the domain of the inverse")
        return (1.0 - math.sqrt(disc)) / (2.0 * self.b)

    def absolute_risk(self, x):
        return 2.0 * self.b / (1.0 - 2.0 * self.b * x)

    def relative_risk(self, x):
        return 2.0 * self.b * x / (1.0 - 2.0 * self.b * x)


def make_utility(kind, param=None):
    """Build one of the built-in utility functions.

    kind is linear, log, power, exponential or quadratic.  power takes
    gamma, exponential takes alpha, quadratic takes b; the others ignore
    param.
    """
    if kind == "linear":
        return _Linear()
    if kind == "log":
        return _Log()
    if kind == "power":
        return _Power(param if param is not None else 0.5)
    if kind == "exponential":
        return _Exponential(param if param is not None else 1.0)
    if kind == "quadratic":
        return _Quadratic(param if param is not None else 0.01)
    raise ValueError("unknown utility {!r}".format(kind))


# ---------------------------------------------------------------------------
# Expected utility, certainty equivalent, risk premium
# ---------------------------------------------------------------------------

def expected_utility(outcomes, probabilities, utility):
    """E[u(X)] for a utility function."""
    if len(outcomes) != len(probabilities):
        raise ValueError("outcomes and probabilities must match in length")
    if not outcomes:
        raise ValueError("need at least one outcome")
    _check_probabilities(probabilities)
    return sum(p * utility.u(x) for p, x in zip(probabilities, outcomes))


def certainty_equivalent(outcomes, probabilities, utility):
    """The amount c with u(c) = E[u(X)], found by exact inversion.

    This is the guaranteed sum an agent with this utility considers as
    good as the lottery.  It is lower than the expected value for any
    concave utility.
    """
    eu = expected_utility(outcomes, probabilities, utility)
    return utility.inverse(eu)


def risk_premium(outcomes, probabilities, utility):
    """Expected value minus certainty equivalent.

    The amount the agent is willing to give up, on average, to swap the
    lottery for a guaranteed payment.  Zero for the risk neutral linear
    utility, positive for any strictly concave one.
    """
    ev = expected_value(outcomes, probabilities)
    return ev - certainty_equivalent(outcomes, probabilities, utility)


# ---------------------------------------------------------------------------
# Prospect theory
# ---------------------------------------------------------------------------

def probability_weight(p, gamma=0.61):
    """The decision weight given to a probability p.

    Low probabilities get overweighted and high ones underweighted,
    which is one of the empirical regularities in prospect theory.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be between 0 and 1")
    numerator = p ** gamma
    return numerator / (numerator + (1.0 - p) ** gamma) ** (1.0 / gamma)


def prospect_value(outcomes, probabilities, reference=0.0, alpha=0.88,
                   beta=0.88, loss_aversion=2.25, gamma=0.61):
    """Prospect theory value of a lottery.

    Gains above the reference point are valued x^alpha, losses below it
    are valued -loss_aversion * (-x)^beta, so losses loom larger than
    gains.  Each outcome probability is distorted through the
    probability weighting function.  The defaults are the parameter
    values estimated by Tversky and Kahneman.
    """
    if len(outcomes) != len(probabilities):
        raise ValueError("outcomes and probabilities must match in length")
    if not outcomes:
        raise ValueError("need at least one outcome")
    _check_probabilities(probabilities)
    total = 0.0
    for x, p in zip(outcomes, probabilities):
        w = probability_weight(p, gamma)
        if x >= reference:
            total += w * (x - reference) ** alpha
        else:
            total += w * (-loss_aversion) * (reference - x) ** beta
    return total


# ---------------------------------------------------------------------------
# St. Petersburg lottery
# ---------------------------------------------------------------------------

def st_petersburg(n):
    """The St. Petersburg game truncated to n tosses, normalized.

    The prize is 2^k dollars when the first tail lands on toss k, which
    has probability 2^-k.  The game is truncated after n tosses, so the
    probabilities are renormalized by their total 1 - 2^-n, which is
    the same as conditioning on a tail appearing in the first n tosses.
    The expected value is n/(1 - 2^-n) and grows without bound, while
    the expected utility under log utility converges to a certainty
    equivalent near 4 dollars, the classic paradox: infinite
    expectation, finite price.
    """
    if n < 1:
        raise ValueError("need at least one toss")
    outcomes = [2.0 ** k for k in range(1, n + 1)]
    probabilities = [2.0 ** (-k) / (1.0 - 2.0 ** (-n)) for k in range(1, n + 1)]
    return outcomes, probabilities


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------

def kelly_fraction_binary(p, odds):
    """Kelly fraction for a bet that pays odds:1 with win probability p.

    The bettor wins odds dollars per dollar staked with probability p
    and loses the stake with probability 1 - p.  The Kelly fraction is
    p - (1 - p)/odds.  It maximizes the expected log growth of wealth;
    betting more than it lowers the growth rate, betting less keeps
    growth but with less variance.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("win probability must lie strictly between 0 and 1")
    if odds <= 0.0:
        raise ValueError("odds must be positive")
    fraction = p - (1.0 - p) / odds
    return max(fraction, 0.0)


def log_growth_rate(returns, fraction):
    """Expected log growth of wealth for a bet size fraction.

    Betting fraction of wealth on each round with return r gives wealth
    growth 1 + fraction * r per round, and the expected log growth is
    the mean of log(1 + fraction * r) over the empirical returns.  Any
    return with 1 + fraction * r <= 0 bankrupts the bettor, so the
    growth rate is negative infinity there.
    """
    if fraction < 0.0:
        raise ValueError("fraction cannot be negative")
    values = []
    for r in returns:
        growth = 1.0 + fraction * r
        if growth <= 0.0:
            return float("-inf")
        values.append(math.log(growth))
    return sum(values) / len(values)


def _log_growth_derivative(returns, fraction):
    """Derivative of expected log growth with respect to the fraction."""
    total = 0.0
    for r in returns:
        growth = 1.0 + fraction * r
        if growth <= 0.0:
            return float("-inf")
        total += r / growth
    return total / len(returns)


def optimal_kelly_fraction(returns):
    """The bet size on [0, 1] that maximizes expected log growth.

    The expected log growth is a concave function of the fraction, so
    its maximum is found by bisecting where its derivative crosses zero.
    The feasible upper bound is the largest fraction that never
    bankrupts the bettor on the given returns.
    """
    if not returns:
        raise ValueError("need at least one return")
    worst = min(returns)
    if worst < 0.0:
        upper = min(1.0, (1.0 - 1e-9) / (-worst))
    else:
        upper = 1.0
    if _log_growth_derivative(returns, 0.0) <= 0.0:
        return 0.0
    if _log_growth_derivative(returns, upper) >= 0.0:
        return upper
    lo, hi = 0.0, upper
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _log_growth_derivative(returns, mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


def expected_log_wealth(returns, fraction, rounds):
    """Expected log wealth after rounds of betting at a fixed fraction.

    The log growth rate per round times the number of rounds, starting
    from wealth 1.
    """
    if rounds < 1:
        raise ValueError("rounds must be at least one")
    return rounds * log_growth_rate(returns, fraction)


def wealth_trajectory(returns, fraction, initial=1.0):
    """Wealth over time betting a fixed fraction on each return.

    Returns the running product of (1 + fraction * r), starting from
    the initial wealth, so the last value is the final wealth.
    """
    if fraction < 0.0:
        raise ValueError("fraction cannot be negative")
    wealth = initial
    trajectory = [wealth]
    for r in returns:
        wealth *= 1.0 + fraction * r
        trajectory.append(wealth)
    return trajectory
