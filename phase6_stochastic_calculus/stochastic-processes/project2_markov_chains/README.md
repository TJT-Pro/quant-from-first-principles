# Project 2: Markov Chains

Transition matrices, stationary distributions, convergence to
stationarity, first-passage and return times, and a from-scratch
chi-square test used to check whether discretised EUR/USD returns
actually behave like a first-order Markov chain.

## Mathematical background

### Transition matrices and the Markov property

A discrete-time Markov chain on states $\{0, \dots, n-1\}$ is
specified by a stochastic matrix $P$ with $P_{ij} = P(X_{t+1} = j \mid X_t = i)$,
each row non-negative and summing to 1. The Markov property says the
distribution of $X_{t+1}$ given the whole history depends only on
$X_t$. The $n$-step transition probabilities are given by matrix
powers: $P(X_{t+n} = j \mid X_t = i) = (P^n)_{ij}$, computed here by
exponentiation by squaring rather than $n$ sequential multiplications.

### Stationary distributions

A distribution $\pi$ is stationary if $\pi P = \pi$ with
$\sum_i \pi_i = 1$. This is a homogeneous linear system with a
redundant equation (the rows of $P - I$ transposed are linearly
dependent, since $(P-I)\mathbf{1} = 0$), so one balance equation is
dropped and replaced with the normalisation constraint before solving
by Gaussian elimination with partial pivoting.

`stationary_distribution_power_iteration` finds the same vector a
different way: repeatedly applying $\pi \mapsto \pi P$ from a uniform
start. For an irreducible, aperiodic chain this converges to $\pi$
regardless of the starting point, by the Perron-Frobenius theorem
applied to the stochastic matrix (the eigenvalue 1 is simple and
strictly dominant in modulus). Both routes are implemented and cross-
checked against each other and against numpy's eigenvector of $P^T$
for eigenvalue 1.

### Irreducibility, periodicity, and convergence

A chain is **irreducible** if every state is reachable from every
other state; `is_irreducible` checks this by forward reachability
(depth-first search) from each state. States that mutually reach each
other form a **communicating class**; `communicating_classes` groups
states this way, which is how the absorbing-chain test distinguishes
the class $\{0\}$, the class $\{2\}$, and the transient class $\{1\}$
in an example with two absorbing states.

The **period** of a state is the greatest common divisor of the
lengths of all cycles through it. `period_of_state` computes this by
running a breadth-first search from the state to get distances
$d(u)$, then taking the gcd of $d(u) + 1 - d(v)$ over every edge
$u \to v$ with positive probability. A chain is **aperiodic** if every
state has period 1.

The ergodic theorem for finite chains says: if $P$ is irreducible and
aperiodic, then $P^n$ converges to a matrix with every row equal to
$\pi$, regardless of starting distribution. The rate of that
convergence is governed by the second-largest eigenvalue modulus of
$P$, so it is geometric, not computed directly here since that
requires an eigenvalue solver. Instead, `convergence_trace` tracks the
total variation distance $\mathrm{TV}(\mu_n, \pi) = \tfrac12 \sum_i |\mu_n(i) - \pi(i)|$
between the distribution after $n$ steps and $\pi$, and
`estimate_convergence_rate` fits a straight line to $\log(\mathrm{TV})$
against $n$: a genuinely geometric decay shows up as a constant
negative slope. Feeding a 2-cycle (period 2) into the same trace shows
the point of periodicity directly: the TV distance to $[0.5, 0.5]$
sits at exactly 0.5 forever and never decays, because a periodic chain
oscillates rather than settling down.

### First-passage and return times

Let $h_i$ be the expected number of steps to first reach a fixed
target state from $i$. Conditioning on the first step gives, for
$i \neq \text{target}$,

$$h_i = 1 + \sum_{k \neq \text{target}} P_{ik} h_k, \qquad h_{\text{target}} = 0.$$

This is a linear system in the $n-1$ unknown $h_i$, solved directly
rather than iteratively. **Kac's formula** gives the expected time to
return to a state from itself, for an irreducible positive-recurrent
chain, as exactly $1/\pi_i$: a state visited with long-run frequency
$\pi_i$ is, on average, revisited every $1/\pi_i$ steps. Both are
checked against long simulated trajectories.

### Testing the Markov hypothesis itself

Given a state sequence, the natural question is whether the process
really depends on its immediate past, or whether the observed
transition frequencies are just what an independent (i.i.d.) sequence
with the same marginal distribution would produce by chance.
`markov_independence_test` builds the two-way table of
(current state, next state) counts and runs the standard chi-square
test of independence: expected count under independence is
$\text{row total} \times \text{column total} / \text{grand total}$,
and the test statistic

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

is compared against a chi-square distribution with
$(n-1)^2$ degrees of freedom.

Getting a p-value requires the chi-square survival function, which
requires the regularised incomplete gamma function. This is
implemented from the classical series/continued-fraction split: the
series form converges quickly for $x < a+1$, the continued fraction
for $x \ge a+1$, with `math.lgamma` supplying the log-gamma
normalisation term. `chi2_sf` and the two gamma routines behind it are
checked against `scipy.stats.chi2.sf` across a range of degrees of
freedom, and against the closed-form exponential survival function
that chi-square with 2 degrees of freedom happens to have.

## Application: does EUR/USD actually behave like a Markov chain?

Two discretisations of the same 6,930 daily EUR/USD returns are
tested.

**Two states (up/down), the natural extension of Project 1's sign
walk.** The lag-1 independence test gives $\chi^2 = 0.96$ on 1
degree of freedom, p = 0.33. There is no detectable evidence that
tomorrow's direction depends on today's. This is worth setting
directly against Project 1's finding that the variance of a 252-day
cumulative sign-walk ran noticeably above the i.i.d. prediction: with
only 27 non-overlapping year-long windows in that analysis, a single
elevated-variance year (a real regime shift, a central-bank
intervention period, whatever) can move that estimate substantially
on its own. The two analyses together are a more honest picture than
either alone: a formal test built directly on the (current day, next
day) structure finds no first-order dependence, so the earlier
variance figure is better read as a small-sample observation than as
confirmed momentum.

**Five states, split into equal-frequency bins by the size of the
return** (quantile breakpoints computed from the data itself, not
fixed thresholds). This test comes back significant: $\chi^2 = 42.0$
on 16 degrees of freedom, p = 0.0004. Looking at the fitted transition
matrix, the extreme bins (largest daily moves in either direction) are
somewhat more likely to be followed by another extreme move than the
stationary distribution alone would suggest. That is the standard
signature of **volatility clustering** rather than directional
momentum: the *sign* of tomorrow's return is close to a coin flip, but
the *size* of tomorrow's return is not independent of today's size.
This distinction, sign versus magnitude, is exactly what a two-state
chain cannot see and a five-state chain can.

The stationary distribution of the five-state chain matches the
empirical bin frequencies almost exactly (both essentially uniform at
0.2, since the bins were built from quantiles of the same data), which
is a useful sanity check rather than a discovery: it confirms the
fitted chain reproduces the marginal distribution it was estimated
from, as it must if the estimation is implemented correctly. The mean
first-passage time between the lowest and highest bins came out to
about 5 trading days each way, meaning the chain reaches the opposite
tail of the daily-return distribution roughly once a week on average,
which is a direct, checkable consequence of the fitted transition
matrix, not of any deeper long-range property.

## Limitations

- `period_of_state` and `is_irreducible` are exact only for the finite
  state spaces used throughout this project; they do not extend as
  written to continuous-state chains.
- The convergence-rate slope from `estimate_convergence_rate` is an
  empirical log-linear fit, not a computed spectral gap; it is a
  reasonable diagnostic for a small number of dominant eigenvalues but
  will understate the true rate if the second and third eigenvalues
  are close in modulus.
- The equal-frequency binning in `quantile_breakpoints` uses linear
  interpolation between order statistics; a different quantile
  convention would shift bin edges slightly and could change the
  independence test's p-value at the margin, though not the
  qualitative conclusion given how far the observed statistic sits
  from the 16-degree-of-freedom null.
- The chi-square independence test assumes the (state, next-state)
  pairs are themselves independent across time, which is not strictly
  true for an overlapping sequence of transitions; this is the
  standard practical approximation used for this kind of test and is
  adequate here given how far the five-bin result sits from the null,
  but it means the reported p-value is only approximately correct in
  a technical sense.
- Five bins with 6,930 observations gives comfortably large expected
  cell counts (in the hundreds), so the chi-square approximation to
  the test statistic's null distribution is reliable; this would not
  hold for a much finer discretisation or a much shorter series.

## CLI examples

```bash
# Stationary distribution, irreducibility and periodicity
python3 cli.py stationary --matrix "0.9,0.1;0.05,0.95"

# Convergence to stationarity from a given starting distribution
python3 cli.py convergence --matrix "0.6,0.4;0.3,0.7" --initial "1,0" --n 30

# Mean first-passage times to a target state
python3 cli.py passage --matrix "0.7,0.3;0.4,0.6" --target 1

# Simulate a trajectory
python3 cli.py simulate --matrix "0.5,0.5;0.5,0.5" --initial 0 --steps 500 --seed 1

# Fit a discretised chain to real EUR/USD data and test the Markov hypothesis
python3 cli.py fxchain --bins 5

# Cross-check every formula against scipy/numpy oracles and simulation
python3 cli.py verify
```

## Tests

```bash
python3 -m pytest test_markov_chains.py -v
```

55 tests covering: linear algebra primitives, transition matrix
validation, chain simulation, reachability and communicating classes,
periodicity on known cycle structures, stationary distributions
against closed-form two-state results and against power iteration,
convergence tracking including the non-convergent periodic case, mean
first-passage times against a closed-form two-state result, the
regularised incomplete gamma function against known chi-square values,
and the EUR/USD application.
