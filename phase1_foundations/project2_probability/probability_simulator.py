import json
import math
import random


# ---------------------------------------------------------------------------
# basic: coins, dice, empirical probability
# ---------------------------------------------------------------------------

def flip_coin(n=1, rng=None):
    """Return n fair coin flips, each 'H' or 'T'."""
    rng = rng or random
    return ['H' if rng.random() < 0.5 else 'T' for _ in range(n)]


def roll_die(sides=6, n=1, rng=None):
    """Return n rolls of a fair sides-sided die, ints from 1 to sides."""
    rng = rng or random
    return [1 + int(rng.random() * sides) for _ in range(n)]


def theoretical_die_probability(sides=6):
    """The probability of any one face of a fair die, 1/sides."""
    if sides < 1:
        raise ValueError("a die needs at least one side, got {}".format(sides))
    return 1.0 / sides


def empirical_probability(outcomes, event):
    """Share of outcomes matching the event.

    ``event`` may be a value to compare against or a callable predicate.
    Raises ValueError on an empty outcome list.
    """
    if not outcomes:
        raise ValueError("empirical probability needs at least one outcome")
    if not callable(event):
        target = event
        event = lambda x: x == target
    hits = sum(1 for outcome in outcomes if event(outcome))
    return hits / len(outcomes)


def running_mean(values):
    """Cumulative averages: mean(values[:k]) for every k."""
    total = 0.0
    means = []
    for i, value in enumerate(values, start=1):
        total += value
        means.append(total / i)
    return means


# ---------------------------------------------------------------------------
# condition: joint tables, marginals, P(A|B), simulation check
# ---------------------------------------------------------------------------

def _check_joint_table(table):
    """A joint probability table must sum to 1 (within tolerance)."""
    total = sum(table.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "joint table probabilities sum to {}, not 1".format(total))


def marginals(table, index):
    """P(X = value) from a joint table {(x, y): p}.

    index 0 sums over the second coordinate (first variable marginal),
    index 1 over the first coordinate.
    """
    _check_joint_table(table)
    sums = {}
    for (a, b), p in table.items():
        key = a if index == 0 else b
        sums[key] = sums.get(key, 0.0) + p
    return sums


def conditional_probability(table, value, given):
    """P(X = value | Y = given) from a joint table {(x, y): p}.

    Defined as P(X=value, Y=given) / P(Y=given).  Raises ValueError when
    the conditioning event has probability zero.
    """
    _check_joint_table(table)
    p_given = 0.0
    p_both = 0.0
    for (x, y), p in table.items():
        if y == given:
            p_given += p
            if x == value:
                p_both += p
    if p_given == 0.0:
        raise ValueError(
            "P(Y={!r}) is zero, so the conditional is undefined".format(given))
    return p_both / p_given


def sample_joint(table, n=1, rng=None):
    """Draw (x, y) pairs from a joint distribution table.

    Builds the cumulative distribution of the (x, y) pairs and inverts a
    uniform draw against it, the standard way to sample a discrete
    distribution.
    """
    _check_joint_table(table)
    rng = rng or random
    entries = list(table.items())
    cumulative = []
    acc = 0.0
    for (x, y), p in entries:
        acc += p
        cumulative.append(acc)
    samples = []
    for _ in range(n):
        u = rng.random()
        for (x, y), c in zip((e[0] for e in entries), cumulative):
            if u < c:
                samples.append((x, y))
                break
    return samples


def empirical_conditional(samples, value, given):
    """Empirical P(X = value | Y = given) from (x, y) samples.

    Counts how often the pair appears, divided by how often Y = given.
    """
    p_given = 0.0
    p_both = 0.0
    for x, y in samples:
        if y == given:
            p_given += 1.0
            if x == value:
                p_both += 1.0
    if p_given == 0.0:
        raise ValueError("no samples with Y = {!r}".format(given))
    return p_both / p_given


# ---------------------------------------------------------------------------
# bayes: prior, likelihood, posterior
# ---------------------------------------------------------------------------

def bayes(prior, likelihood):
    """Posterior P(H | E) from a prior and a likelihood.

    ``prior`` maps each hypothesis to P(H).  ``likelihood`` maps each
    hypothesis to P(E | H), the probability of the observed evidence under
    that hypothesis.  Returns a dict of posteriors.

    P(H | E) = P(E | H) * P(H) / P(E), with
    P(E) = sum over h of P(E | h) * P(h).
    """
    if not prior:
        raise ValueError("prior must list at least one hypothesis")
    prior_total = sum(prior.values())
    if abs(prior_total - 1.0) > 1e-9:
        raise ValueError(
            "prior probabilities sum to {}, not 1".format(prior_total))
    for h in prior:
        if h not in likelihood:
            raise ValueError(
                "no likelihood given for hypothesis {!r}".format(h))

    evidence_prob = 0.0
    for h, p_h in prior.items():
        evidence_prob += likelihood[h] * p_h
    if evidence_prob == 0.0:
        raise ValueError(
            "P(evidence) is zero; the evidence is impossible under these hypotheses")

    posterior = {}
    for h, p_h in prior.items():
        posterior[h] = likelihood[h] * p_h / evidence_prob
    return posterior


def medical_test_example():
    """The classic base-rate problem, with every number explicit.

    Disease prevalence is 1%.  The test is positive in 99% of sick people
    and in 5% of healthy people.  Asked: given a positive result, what is
    the chance the person actually has the disease?  Returns a dict with
    the priors, the likelihoods, and the posterior.
    """
    prior = {"disease": 0.01, "healthy": 0.99}
    likelihood = {"disease": 0.99, "healthy": 0.05}
    posterior = bayes(prior, likelihood)
    return {"prior": prior, "likelihood": likelihood, "posterior": posterior}


# ---------------------------------------------------------------------------
# sample: Binomial, Geometric and Poisson from uniform draws
# ---------------------------------------------------------------------------

def _ncr(n, r):
    """Exact binomial coefficient, same recurrence used in Project 1."""
    if r < 0 or r > n:
        return 0
    if r > n - r:
        r = n - r
    result = 1
    for i in range(1, r + 1):
        result = result * (n - r + i) // i
    return result


def bernoulli(p, rng=None):
    """One Bernoulli draw: 1 with probability p, else 0."""
    rng = rng or random
    return 1 if rng.random() < p else 0


def binomial_sample(n, p, size=1, rng=None):
    """size draws from Binomial(n, p).

    Each draw sums n independent Bernoulli(p) draws.  Straight from the
    definition: Binomial(n, p) counts successes in n trials.
    """
    if n < 0:
        raise ValueError("n must be non-negative, got {}".format(n))
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1, got {}".format(p))
    rng = rng or random
    return [sum(1 for _ in range(n) if rng.random() < p) for _ in range(size)]


def geometric_sample(p, size=1, rng=None):
    """size draws from Geometric(p).

    The number of trials until the first success, so the support is
    1, 2, 3, ...  Keeps drawing until a uniform draw lands below p.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1], got {}".format(p))
    rng = rng or random
    samples = []
    for _ in range(size):
        trials = 0
        while True:
            trials += 1
            if rng.random() < p:
                break
        samples.append(trials)
    return samples


def poisson_sample(lam, size=1, rng=None):
    """size draws from Poisson(lam) using Knuth's method.

    The algorithm: L = exp(-lam), then keep multiplying a running product
    by uniform draws until the product drops below L.  The number of
    draws used minus one is the sample.  Exact, and only needs uniforms.
    """
    if lam <= 0:
        raise ValueError("lambda must be positive, got {}".format(lam))
    rng = rng or random
    limit = math.exp(-lam)
    samples = []
    for _ in range(size):
        k = 0
        prod = 1.0
        while True:
            k += 1
            prod *= rng.random()
            if prod <= limit:
                break
        samples.append(k - 1)
    return samples


def binomial_pmf(n, p, k):
    """P(X = k) for X ~ Binomial(n, p): C(n,k) p^k (1-p)^(n-k)."""
    if k < 0 or k > n:
        return 0.0
    return _ncr(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def geometric_pmf(p, k):
    """P(X = k) for X ~ Geometric(p): (1-p)^(k-1) p, k = 1, 2, ..."""
    if k < 1:
        return 0.0
    return ((1.0 - p) ** (k - 1)) * p


def poisson_pmf(lam, k):
    """P(X = k) for X ~ Poisson(lam): exp(-lam) lam^k / k!."""
    if k < 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def empirical_counts(samples):
    """Count how often each value appears, as {value: count}."""
    counts = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# converge: plots for the law of large numbers and the samplers
# ---------------------------------------------------------------------------

def _get_plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def plot_coin_convergence(n=3000, path="plots/coin_convergence.png", rng=None):
    """Plot the running share of heads against the theoretical 0.5."""
    flips = flip_coin(n, rng)
    means = running_mean([1 if f == 'H' else 0 for f in flips])
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0.5, color='red', lw=1, ls='--',
               label='theoretical P(heads) = 0.5')
    ax.plot(range(1, n + 1), means, lw=1, color='black', label='empirical')
    ax.set_xlabel('number of flips')
    ax.set_ylabel('running proportion of heads')
    ax.set_title('Coin: the empirical probability converges to 0.5')
    ax.set_ylim(0, 1)
    ax.legend()
    return _save_figure(fig, path)


def plot_dice_convergence(sides=6, n=3000, path="plots/dice_convergence.png", rng=None):
    """Plot the running average of die rolls against the mean (sides+1)/2."""
    rolls = roll_die(sides, n, rng)
    means = running_mean(rolls)
    expected = (sides + 1) / 2.0
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(expected, color='red', lw=1, ls='--',
               label='theoretical mean = {}'.format(expected))
    ax.plot(range(1, n + 1), means, lw=1, color='black', label='empirical')
    ax.set_xlabel('number of rolls')
    ax.set_ylabel('running average of the rolls')
    ax.set_title('Dice: the sample mean converges to the expected value')
    ax.legend()
    return _save_figure(fig, path)


def plot_distribution_comparison(samples, pmf, name, k_max,
                                 path, expected_label):
    """Bar chart of empirical proportions against the theoretical PMF."""
    counts = empirical_counts(samples)
    total = len(samples)
    ks = list(range(0, k_max + 1))
    empirical = [counts.get(k, 0) / total for k in ks]
    theoretical = [pmf(k) for k in ks]
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.4
    ax.bar([k - width / 2 for k in ks], empirical, width,
           color='black', label='empirical')
    ax.bar([k + width / 2 for k in ks], theoretical, width,
           color='red', label=expected_label)
    ax.set_xlabel('k')
    ax.set_ylabel('probability')
    ax.set_title('{}: empirical distribution vs the PMF'.format(name))
    ax.legend()
    return _save_figure(fig, path)
