# Project 1: Random Walks

A from-scratch treatment of the simple random walk on Z and Z^2: exact
finite-n distributions, the reflection principle, first-passage times,
gambler's-ruin-type hitting probabilities, and an application to the
sign of daily EUR/USD returns.

## Mathematical background

### The 1D simple random walk

Let $\xi_1, \xi_2, \dots$ be i.i.d. with $P(\xi_i = 1) = p$,
$P(\xi_i = -1) = 1-p$. The walk is

$$S_0 = 0, \qquad S_n = \sum_{i=1}^n \xi_i.$$

**Mean and variance.** Since $E[\xi_i] = 2p - 1$ and
$\mathrm{Var}(\xi_i) = E[\xi_i^2] - E[\xi_i]^2 = 1 - (2p-1)^2 = 4p(1-p)$,
independence gives

$$E[S_n] = n(2p-1), \qquad \mathrm{Var}(S_n) = 4np(1-p).$$

**Exact distribution.** $S_n = m$ requires $k$ up-steps and $n-k$
down-steps with $m = 2k - n$, so $k = (n+m)/2$ must be an integer in
$[0, n]$:

$$P(S_n = m) = \binom{n}{k} p^k (1-p)^{n-k}, \qquad k = \frac{n+m}{2}.$$

This is implemented directly with `math.comb`. No normal
approximation and no scipy appear anywhere in the core code.

### The reflection principle

For the **symmetric** walk ($p = 1/2$) and $a > 0$, let
$M_n = \max_{0 \le k \le n} S_k$. Any path that touches level $a$ at
some time $\tau \le n$ can be reflected about $y = a$ from $\tau$
onward: up-steps become down-steps and vice versa. Because up and down
steps are equally likely, this reflection is measure-preserving and
maps paths with $S_n = m < a$ that touched $a$ bijectively onto paths
ending at $2a - m > a$. Summing over all paths that end above $a$ (all
of which must have touched $a$) and all paths that end exactly at $a$
gives the classical identity

$$P(M_n \ge a) = P(S_n \ge a) + P(S_n > a).$$

The two-sided nature of this argument is why `reflection_principle_max_at_least`
raises for $p \neq 0.5$: the reflected path has a different probability
than the original once steps are not equally likely, so the bijection
no longer preserves probability. `reflection_principle_max_at_least_bruteforce`
checks the formula against exhaustive enumeration of all $2^n$ sign
sequences for small $n$.

### First-passage times and the hitting time theorem

Let $T_a = \min\{n : S_n = a\}$ for $a > 0$. The **hitting time
theorem** (a cycle-lemma argument, sometimes attributed to Otter or
Kemperman, related to the ballot problem) states that for i.i.d.
$\pm 1$ increments with *any* $p$,

$$P(T_a = n) = \frac{a}{n} P(S_n = a), \qquad n \ge a,\ n - a \text{ even}.$$

Unlike the reflection principle above, this identity holds for biased
walks too, because the cycle lemma is a purely combinatorial statement
about cyclic rotations of a fixed sequence of $\pm 1$ steps and does
not depend on steps being equiprobable.

### Hitting probabilities and gambler's ruin with one boundary

For $a > 0$, if $p \ge 1/2$ the walk has non-negative drift and
reaches any level with probability 1. If $p < 1/2$, the standard
gambler's-ruin argument with the far boundary sent to infinity gives

$$P(\text{ever reach } a) = \left(\frac{p}{1-p}\right)^{a}.$$

**Expected hitting time.** When $p > 1/2$, $T_a$ has finite mean and
Wald's identity ($E[S_T] = E[T] \cdot E[\xi]$ for a stopping time with
finite expectation) gives

$$E[T_a] = \frac{a}{2p - 1}.$$

For $p \le 1/2$ this is set to infinity: at $p = 1/2$ the walk is
null-recurrent (it hits $a$ almost surely but $E[T_a] = \infty$), and
for $p < 1/2$ it may never hit $a$ at all.

### The 2D simple random walk and Polya's recurrence theorem

On $\mathbb{Z}^2$ with four equally likely nearest-neighbour moves,
Polya's theorem states the walk is **recurrent**: it returns to the
origin with probability 1 as the time horizon grows, in contrast to
$d \ge 3$ where the walk is transient. `estimate_return_probability_2d`
demonstrates this by showing the empirical return probability climbing
towards 1 as the horizon lengthens; a formal proof is outside the
scope of this repository and left as a documented, cited result.

## Application: is EUR/USD direction a fair coin flip?

`fx_windowed_analysis` treats the **sign** of each daily EUR/USD log
return as a $\pm 1$ step and splits the 6,930-observation series
(1999-01-05 to 2026-08-21, FRED series DEXUSEU) into non-overlapping
windows. Two comparisons are made:

1. **Mean and variance** of the window-ending position against the
   theoretical $n(2p-1)$ and $4np(1-p)$ using the *empirical* $\hat p$
   (the fraction of up days, which sits close to 0.5).
2. **Reflection-principle exceedance probability**, using $p = 0.5$ as
   an explicit null hypothesis (the weak-form-efficient-market view
   that daily direction is an unpredictable, memoryless coin flip),
   compared against the empirical fraction of windows whose running
   maximum exceeds a threshold.

Running `python3 cli.py fxwalk --window 252` (one trading year) gives
$\hat p \approx 0.495$, consistent with no strong directional bias.
But the **empirical variance of the 252-day walk is roughly 1.6-1.8x
the theoretical i.i.d. variance**, and the empirical exceedance
frequency is noticeably higher than the reflection-principle
prediction. A pure i.i.d. coin-flip model of sign would predict a
variance of exactly $4 \cdot 252 \cdot p(1-p) \approx 252$; the
realised variance runs well above that. This is the standard
signature of positive short-run autocorrelation in the sign process:
trending and momentum runs are longer than an i.i.d. model would
produce, even though the unconditional up/down split is close to fair. It does
not by itself imply exploitable predictability once transaction costs
and risk are taken into account, but it does mean the "EUR/USD
direction as fair-coin random walk" model understates tail risk in the
cumulative sign process.

## Limitations

- The reflection-principle closed form is only derived for $p = 0.5$.
  A general-$p$ maximum distribution exists but requires a different
  (generating-function) argument, out of scope here.
- `estimate_return_probability_2d` is a Monte Carlo demonstration of
  Polya recurrence, not a proof; convergence to 1 is slow (logarithmic
  in the number of steps) so the empirical probability at finite
  horizons stays well below 1.
- The FX sign-walk analysis uses non-overlapping windows for
  independence between windows; this discards some data relative to a
  rolling-window design and limits the number of independent replicas
  available (27 windows of 252 days each in the current dataset).
- Zero-return days are broken by an unbiased coin flip in
  `sign_sequence`; none occur in the DEXUSEU dataset at the precision
  provided, so this branch is untested against real data but is
  covered by a unit test on synthetic input.

## CLI examples

```bash
# 1D walk with drift
python3 cli.py simulate1d --steps 200 --p 0.55 --seed 7

# 2D walk displacement statistics
python3 cli.py simulate2d --steps 5000 --seed 3

# Reflection principle: P(max_{k<=n} S_k >= a)
python3 cli.py reflection --n 500 --a 20

# Hitting probability and expected hitting time
python3 cli.py hitting --a 10 --p 0.45

# Simulate first-passage times and compare to theory
python3 cli.py gambler --a 5 --p 0.55 --trials 10000 --max-steps 5000

# Apply to real EUR/USD data
python3 cli.py fxwalk --window 252

# Cross-check every formula against scipy/numpy oracles and brute force
python3 cli.py verify
```

## Tests

```bash
python3 -m pytest test_random_walks.py -v
```

58 tests covering: path generation and reproducibility, exact
finite-sample distributions, the reflection principle against
brute-force enumeration, the hitting time theorem against simulation,
gambler's ruin hitting probabilities, Wald's-identity expected hitting
times, 2D displacement and recurrence trend, and the EUR/USD
application.
