# Project 5: Martingales

Martingale constructions from a random walk, the optional stopping
theorem applied to gambler's ruin (cross-checked against Project 1's
combinatorial answers from a completely different route), Doob's
maximal inequality, Doob's martingale convergence theorem demonstrated
concretely through the Polya urn, and a from-scratch investigation of
the classic doubling betting system that turned up a genuine,
non-obvious result about why naive simulation can be deeply misleading
even when the underlying mathematics is exactly, provably correct.

## Mathematical background

### What makes a process a martingale

$X_n$ is a martingale with respect to its own history if
$E[X_{n+1} \mid X_0, \dots, X_n] = X_n$: knowing the past gives no
information about the direction of the next move, only its
uncertainty. `empirical_conditional_drift` tests this computationally
for any constructed process: bucket many simulated transitions by the
current value, average the next increment within each bucket, and
check whether that average sits at zero. A symmetric random walk
passes this test trivially; a biased one fails it, showing an average
increment of $2p-1$ regardless of the current position, exactly the
constant drift the definition says a martingale cannot have.

### Restoring the martingale property in a biased walk

**Compensated walk.** $M_n = S_n - n(2p-1)$ subtracts the drift out.
$E[M_{n+1} - M_n \mid F_n] = E[S_{n+1}-S_n] - (2p-1) = (2p-1)-(2p-1) = 0$
exactly, so `compensated_walk` restores the martingale property for
any $p$, which `empirical_conditional_drift` confirms even at $p=0.65$.

**The exponential (Wald) martingale.** $M_n = \theta^{S_n}$ is a
martingale exactly when $\theta$ solves $E[\theta^{\xi}] = 1$ for the
step distribution: $p\theta + (1-p)\theta^{-1} = 1$, which factors to
give $\theta = (1-p)/p$ (besides the trivial root $\theta=1$, which is
why the construction is degenerate at $p=0.5$, where the only
solution is $\theta=1$ and the sequence never moves). This is the
tool the rest of the project needs, because it is the martingale that
makes the *two-boundary* gambler's ruin problem solvable in closed
form.

### Gambler's ruin via optional stopping, cross-checked against Project 1

For the walk started at 0, stopped the first time it reaches $a>0$ or
$-b<0$, let $T$ be that stopping time (finite almost surely, since the
walk is recurrent). The **optional stopping theorem** says that for a
martingale $M_n$ and a stopping time $T$ satisfying the theorem's
regularity conditions (here, $T$ has finite expectation and the
martingale increments are bounded, both true for a walk trapped
between two finite boundaries), $E[M_T] = E[M_0]$.

**Symmetric case ($p=0.5$).** $S_n$ itself is the martingale, so
$E[S_T] = 0$. Writing $\alpha = P(\text{hit } a \text{ first})$, this
is $\alpha a - (1-\alpha)b = 0$, giving $\alpha = b/(a+b)$ directly, the same closed form Project 1 derived from the gambler's-ruin
difference equation, now reached through martingale theory instead of
combinatorics. `expected_duration` uses a second martingale for this
case, $S_n^2 - n$ (a martingale because $E[S_{n+1}^2 \mid F_n] =
S_n^2 + 1$ for a unit-variance step), giving $E[T] = E[S_T^2] = \alpha
a^2 + (1-\alpha)b^2 = ab$ after substituting $\alpha$.

**Biased case.** The exponential martingale gives $E[\theta^{S_T}]=1$,
i.e. $\alpha\theta^a + (1-\alpha)\theta^{-b} = 1$, solved directly for
$\alpha$ in `ruin_probability`. The compensated walk then gives
$E[S_T] = (2p-1)E[T]$, so $E[T] = (\alpha a - (1-\alpha)b)/(2p-1)$ in
`expected_duration`. Both formulas are checked against direct
simulation of the two-boundary walk (`simulate_ruin`), and the
symmetric case is checked against Project 1's independently-derived
formulas: the same numbers, reached by two unrelated arguments, are
about as strong a correctness signal as a from-scratch implementation
can get without an external oracle.

### Doob's maximal inequality

For a non-negative submartingale $M_n$ (here, $M_n = S_n^2$, which is
a submartingale because $x \mapsto x^2$ is convex and Jensen's
inequality applied conditionally to the martingale $S_n$ gives
$E[S_{n+1}^2 \mid F_n] \ge S_n^2$), Doob's inequality bounds the
running maximum:

$$P\left(\max_{k \le n} M_k \ge \lambda\right) \le \frac{E[M_n]}{\lambda}.$$

With $E[S_n^2] = n$ exactly, `doob_maximal_bound` returns $n/\lambda$,
and `doob_maximal_inequality_check` confirms by simulation that the
empirical probability never exceeds this bound across several
parameter combinations, while also, deliberately, being loose: the
empirical probability sits well under the bound in every case tested,
which is expected, since Doob's inequality is a general-purpose bound
that does not exploit any structure specific to the random walk.

### Doob's convergence theorem via the Polya urn

Doob's martingale convergence theorem states that an $L^1$-bounded
martingale converges almost surely. The **Polya urn** is a clean,
fully verifiable example: start with $r_0$ red and $b_0$ blue balls;
each draw returns the ball plus one more of the same colour. The
fraction of red balls, $X_n = R_n/(R_n+B_n)$, is a bounded martingale
(bounded between 0 and 1, hence $L^1$-bounded), since

$$E[X_{n+1} \mid F_n] = X_n \cdot \frac{R_n+1}{R_n+B_n+1} + (1-X_n)\cdot\frac{R_n}{R_n+B_n+1} = X_n$$

simplifies exactly (the algebra is a direct consequence of how drawing
red changes both the red count and the total by amounts that preserve
the expected next fraction). Doob's theorem guarantees this converges
almost surely to *some* random limit; the classical extra fact, not
implied by the convergence theorem itself but verifiable
independently, is that the limit is exactly $\mathrm{Beta}(r_0, b_0)$
distributed. `polya_urn_terminal_moments` gives the Beta distribution's
known mean and variance, matched here against simulated terminal
fractions both by moments (in the test suite) and, in `verify`, by a
full Kolmogorov-Smirnov test against `scipy.stats.beta`, which is the
strongest check available: the entire terminal distribution shape, not
just its first two moments, matches the theoretical Beta law.

### Testing the martingale difference hypothesis with OLS

Market efficiency, stated in martingale language, is
$E[r_{t+1} \mid F_t] = 0$: the sequence of returns is a **martingale
difference sequence** (an MDS), even though the returns themselves
need not be independent or identically distributed (their variance,
in particular, is allowed to depend on the past, which is exactly
what Projects 2 through 4 already found for EUR/USD). `ols_fit`
implements ordinary least squares from scratch: slope, intercept,
$R^2$, and two different standard errors for the slope, and
`martingale_difference_test` regresses $r_t$ on $r_{t-1}$: under the
MDS hypothesis the slope should not differ significantly from zero.

Two standard errors are computed deliberately. The **homoskedastic**
formula, $\sqrt{\hat\sigma^2/S_{xx}}$, assumes constant residual
variance. The **White/HC0 robust** formula,
$\sqrt{\sum(x_i-\bar x)^2 e_i^2}/S_{xx}$, does not, and is the
appropriate choice here precisely because Projects 2 through 4
established that EUR/USD return variance is not constant over time.
Both are reported side by side (`cli.py fxmartingale`) rather than
silently picking one, since the gap between them is itself informative
about how much the heteroskedasticity distorts naive inference.

### The doubling strategy: what optional stopping actually requires

The classical "martingale" betting system, double the stake after
every loss and reset to the base stake after every win, is often
described as a way to guarantee a profit from a fair game. It is worth
working through carefully, because the naive version of that claim is
false in a specific, instructive way, and the corrected version is a
sharp illustration of exactly which hypotheses the optional stopping
theorem needs.

**The exact result.** At every round, the current stake is determined
entirely by the outcomes so far (it is *predictable*), and the coin
outcome is independent of that history with mean $2p-1$. So the P&L
process $P_n$ satisfies $E[P_n - P_{n-1} \mid F_{n-1}] =
\mathrm{stake}_{n-1}\cdot(2p-1)$, which is exactly zero for a fair
coin ($p=0.5$) regardless of how the stake is sized: doubling,
capped doubling, or anything else predictable. This means $E[P_n] = 0$
**exactly**, for every fixed $n$, whether or not a cap on the number
of consecutive doublings is imposed. `exact_expected_pnl` computes
this directly via forward propagation of the exact probability
distribution over the current losing-streak length (a small Markov
chain: win resets the streak to 0, loss extends it by one, or forces
a reset if a cap is in effect), rather than by simulation, and it
matches exhaustive enumeration over all $2^n$ outcome sequences
exactly for every $n$ tested up to 12, and stays exactly zero at
$n=2000$ once the stake-scaling arithmetic is done via `math.ldexp`
instead of literally forming the (rapidly overflowing) integer
$2^{2000}$.

**Where the naive intuition breaks down.** Feeding a completely fair,
genuinely i.i.d. coin into 5,000 independent 1,000-round Monte Carlo
trials of the *uncapped* strategy gives an average P&L in the
neighbourhood of +495, nowhere near the exact value of 0, and not
because of a coding error. Every completed win-loss cycle nets exactly
+1 stake unit (a losing streak of length $k$ costs $2^k - 1$ in total,
and the recovering win pays exactly $2^k$), so almost every simulated
1,000-round path, having no way to end mid-streak with high
probability, shows a P&L close to the number of wins in the sequence:
the result is visibly, reliably positive. The exact zero mean is only restored by
a compensating, catastrophically large loss on the rare paths that hit
a very long losing streak: `probability_max_streak_at_least` shows the
probability of a streak of 30 or more consecutive losses within 1,000
rounds is on the order of $10^{-9}$, but the loss on such a path is on
the order of $2^{30} \approx 10^9$. The product of a probability near
$10^{-9}$ and a loss near $10^9$ is exactly the missing few hundred
units of negative expectation needed to bring the true mean back to 0,
and no simulation with a few thousand trials will ever sample that
event. This is a completely general phenomenon for any exactly
zero-mean random variable whose distribution has enough skew: Monte
Carlo estimates converge to the true mean eventually, but "eventually"
can require vastly more trials than are practical whenever the
variable's variance (here, effectively infinite once stakes are
allowed to grow without bound relative to a fixed number of rounds)
is dominated by an event drawn from the extreme tail.

**Capping restores sane Monte Carlo behaviour, and this matters for
optional stopping specifically.** With a cap on the number of
consecutive doublings (a forced loss realised once the cap is
reached, rather than continuing to double), the worst-case loss on
any single path is bounded, and the probability of hitting that
bound within a realistic horizon is no longer astronomically small
(with a cap of 10 doublings, the exact probability of at least one
forced stop within 2,000 rounds is about 62%, computed exactly by
`probability_max_streak_at_least`). Monte Carlo simulation of the
capped strategy converges to something close to its true (still
exactly zero, for a fair coin) mean within a few thousand trials, as
`verify` checks directly. The general lesson this project set out to
demonstrate: the optional stopping theorem needs either a bounded
stopping time or bounded/integrable increments to guarantee
$E[M_T]=E[M_0]$; the doubling strategy over an unbounded, uncapped
horizon does not obviously violate the martingale property itself
(each fixed-$n$ P&L truly has mean zero), but it does produce a
random variable whose *practical, simulation-based* behaviour is
dominated by a tail event so rare that any finite study will
systematically misjudge its expectation: the precise, quantifiable
version of "the free lunch requires infinite capital you'll never
actually need to use."

## Application: does the martingale property hold for EUR/USD, and does a path-dependent strategy see something a simple test misses?

**The direct martingale difference test comes back clean.** Regressing
$r_t$ on $r_{t-1}$ across all 6,929 usable pairs gives a slope of
0.0116, with a homoskedastic p-value of 0.33 and a robust p-value of
0.39, no evidence against the martingale difference hypothesis,
consistent with every prior project's finding of no linear lag-1
predictability in the returns themselves.

**But the doubling strategy tells a different, genuinely informative
story.** Mapping daily returns to a $\pm1$ up/down sequence and
running the capped ($\text{cap}=10$) doubling strategy on 1,000-round
resamples gives two very different answers depending on how the
resample is built. A block bootstrap with 20-day blocks (preserving
real local dependence structure) gives an average P&L around +248,
with a standard error around 23, more than 10 standard errors away
from the exact $-51$ that an i.i.d. Bernoulli model with the same
marginal up-probability would predict. An i.i.d. resample from the
same pool of outcomes (block length 1, deliberately destroying any
local dependence) lands close to that i.i.d. prediction instead. The
same underlying data, resampled two different ways, gives two
statistically distinguishable answers for this specific strategy.

**Why**: comparing the empirical distribution of real down-streak
lengths against the i.i.d. geometric distribution implied by the same
marginal down-probability shows real streaks of moderate length (5-9
consecutive down days) occurring roughly twice as often as an i.i.d.
model predicts, while the very longest streaks (10 or more) are
somewhat rarer than that same model predicts, and no streak longer
than 10 was observed anywhere in the 6,930-day sample. A strategy
whose payoff is dominated by exactly the tail of the streak-length
distribution is far more sensitive to this kind of run-length
structure than a simple lag-1 contingency-table test is designed to
detect: Project 2's aggregate independence test asks "does yesterday's
direction predict today's, on average," and correctly answers "not
much" (p=0.33); it was never built to notice that the *distribution
of run lengths itself* departs mildly from what independence would
imply. This is a genuinely useful, and genuinely humbling, result: a
statistically clean pass on a standard dependence test does not mean
every strategy built on that data is exposed to the same risk profile
an i.i.d. model would suggest, particularly for a strategy whose
payoff is convex in the streak length the way doubling's is.

## Limitations

- The optional stopping theorem's regularity conditions are stated
  informally above; the two-boundary gambler's ruin problem satisfies
  them because both the stopping time and the martingale's per-step
  increments are bounded, but this is not verified programmatically
  for arbitrary constructions fed into the framework.
- `exact_expected_pnl`'s Markov-chain state (the current losing-streak
  length) assumes i.i.d. $\pm1$ outcomes; it is an exact tool for the
  i.i.d. baseline used as a null hypothesis, not a general-purpose
  solver for arbitrary dependent processes, which is exactly why the
  EUR/USD application compares simulation against this i.i.d.
  benchmark rather than trying to extend the DP to the real dependence
  structure directly.
- The 6-10 day threshold used to characterise the "moderate excess,
  long-tail deficit" streak pattern is drawn from a single 6,930-day
  historical sample; with only 1,753 down-streaks total and very few
  observations at the extreme tail (streak lengths above 10 essentially
  never occurred), the deficit at the far tail specifically should be
  read as suggestive rather than statistically conclusive on its own.
- The White/HC0 robust standard error corrects for heteroskedasticity
  but not for serial correlation in the residuals; given the volatility
  clustering documented elsewhere, a full heteroskedasticity-and-
  autocorrelation-consistent (Newey-West) standard error would be a
  more complete choice for a production analysis, though it would not
  change the qualitative conclusion here given how far from
  significance the slope already sits under both simpler estimators.

## CLI examples

```bash
# Gambler's ruin via optional stopping
python3 cli.py ruin --a 10 --b 15 --p 0.5

# Doob's maximal inequality
python3 cli.py doob --n 50 --level 100 --trials 10000

# Polya urn and its Beta-distributed limit
python3 cli.py polya --r0 3 --b0 5 --draws 2000

# The doubling strategy, with its exact (DP) expected P&L
python3 cli.py doubling --steps 1000 --p 0.5 --cap 10

# Martingale difference test and the streak-tail diagnostic on EUR/USD
python3 cli.py fxmartingale

# Cross-check every formula against scipy/numpy oracles, exhaustive
# enumeration, and exact dynamic programming
python3 cli.py verify
```

## Tests

```bash
python3 -m pytest test_martingales.py -v
```

56 tests covering: the empirical martingale-property checker itself,
compensated and exponential martingale constructions, gambler's ruin
against both closed forms and simulation, Doob's maximal inequality
across parameter sets, the Polya urn's martingale property and its
Beta-distributed limit, OLS against a perfect-fit edge case that would
otherwise divide by zero, exhaustive enumeration proving the doubling
strategy's exact zero mean for both capped and uncapped variants, the
exact dynamic-programming expectation matching that enumeration and
remaining numerically stable at n=2000, and the EUR/USD application
including the block-bootstrap-versus-i.i.d.-resample contrast that
is this project's central finding.
