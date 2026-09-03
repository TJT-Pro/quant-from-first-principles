# Project 4: Brownian Motion

Simulation, scaling, and the two path properties that define Brownian
motion mathematically (finite quadratic variation, infinite total
variation), plus a variance ratio test that checks whether EUR/USD
log-prices actually scale the way a Brownian motion should.

## Mathematical background

### Constructing paths from first principles

Standard Brownian motion $W_t$ satisfies $W_0 = 0$, has independent
increments, and $W_{t+s} - W_t \sim N(0, s)$. Simulating a path means
drawing Gaussian increments, which means the first thing this project
needs is a Gaussian random number generator built only from
`random.random()`. The **Box-Muller transform** does this: two
independent uniforms $U_1, U_2$ on $(0,1)$ produce two independent
standard normals

$$Z_0 = \sqrt{-2\ln U_1}\cos(2\pi U_2), \qquad Z_1 = \sqrt{-2\ln U_1}\sin(2\pi U_2),$$

because $(Z_0, Z_1)$ written in polar form has an angle uniform on
$[0, 2\pi)$ and a radius $R = \sqrt{-2\ln U_1}$ whose distribution is
exactly the Rayleigh distribution that a pair of independent standard
normals has in polar coordinates ($R^2 = -2\ln U_1$ is Exponential(1/2)
distributed, matching the sum of two squared independent standard
normals, which is chi-square with 2 degrees of freedom = Exponential(1/2)).
`generate_standard_normals` uses both $Z_0$ and $Z_1$ from each pair
of uniforms rather than discarding one, for efficiency.

A path is then built by accumulating increments: $W(t + \Delta t) =
W(t) + Z\sqrt{\Delta t}$ for $Z \sim N(0,1)$, since scaling a standard
normal by $\sqrt{\Delta t}$ gives exactly the required $N(0, \Delta t)$
increment.

The standard normal CDF needed everywhere else in this project (the
reflection principle, the variance ratio test's p-value) comes from
the identity $\Phi(x) = \tfrac12(1 + \mathrm{erf}(x/\sqrt2))$, using
`math.erf`, which is a special function in the standard library, not
a statistics routine, so this is still a from-scratch construction of
the normal CDF rather than a borrowed one.

### Quadratic variation vs. total variation

These two path functionals are the mathematical core of why Brownian
motion behaves so differently from a differentiable function, and the
project treats them as an experiment rather than just a formula.

**Quadratic variation** sums squared increments,
$\sum_i (W(t_{i+1}) - W(t_i))^2$. Each term has expectation exactly
$\Delta t_i$ regardless of the partition, so the sum's expectation is
always the total elapsed time $t$, and by the law of large numbers its
variance shrinks as the partition gets finer (more, smaller terms
average out). `quadratic_variation` computes this directly, and the
test suite checks both halves of the claim: the mean converges to $t$
at every partition size, and the spread around that mean shrinks
monotonically as `n_steps` increases from 20 to 200 to 2000.

**Total variation** sums absolute increments,
$\sum_i |W(t_{i+1}) - W(t_i)|$. For a fixed horizon, refining the
partition does not make this converge; instead it grows. This is
because $E[|Z|\sqrt{\Delta t}\,] = \sqrt{2\Delta t/\pi}$ per step, and
summing $n$ steps of size $\Delta t = t/n$ gives an expected total
variation of $\sqrt{2nt/\pi}$, which grows with $\sqrt n$ without
bound. This is the standard demonstration that Brownian paths have
infinite total variation (and are therefore nowhere differentiable in
the classical sense): the two functionals are computed by nearly
identical code, and it is the exponent on the increment (squared vs.
absolute) that produces completely different limiting behaviour.

### The reflection principle for continuous paths

$$P\left(\max_{0 \le s \le t} W_s \ge a\right) = 2\left(1 - \Phi\!\left(\frac{a}{\sqrt t}\right)\right), \qquad a > 0.$$

The argument mirrors Project 1's random-walk reflection principle, but
here the symmetry is exact rather than combinatorial: once a
continuous path first touches level $a$ at some time $\tau$, negating
the path after $\tau$ about the line $y=a$ produces another Brownian
path (by the strong Markov property and the symmetry of Gaussian
increments), and this reflected path ends above $a$ exactly when the
original ends below. Because the event $\{M_t \ge a\}$ requires the
path to have touched $a$ at all, and a continuous path cannot jump
over a level, $\{M_t \ge a\} = \{T_a \le t\}$ exactly: the running-
maximum question and the first-passage-time question are the same
event, so `reflection_principle_bm` and `first_passage_cdf_bm` compute
the identical formula. Differentiating this CDF with respect to $t$
gives the first-passage-time density

$$f_{T_a}(t) = \frac{a}{\sqrt{2\pi t^3}} \exp\!\left(-\frac{a^2}{2t}\right),$$

an inverse-Gaussian-type density, checked here by confirming its
numerical integral reproduces the CDF above.

### Donsker's invariance principle

A simple symmetric random walk (Project 1's building block), rescaled
so that each of $n$ steps has size $\sqrt{t_{\max}/n}$, has variance
$k \cdot (t_{\max}/n) = (k/n) t_{\max}$ after $k$ steps: exactly the
variance a Brownian motion should have at the corresponding time. This
matches the first two moments at every step by construction, but
Donsker's invariance principle is the stronger statement that the
*entire path* converges in distribution to Brownian motion as $n \to
\infty$, not just the endpoint's variance. Since a full distributional
convergence test is out of scope here, the project checks a proxy:
excess kurtosis (which is exactly 0 for a true Gaussian) of the
walk's terminal value should shrink towards 0 as $n$ grows, since a
coarse walk (few steps) is visibly non-Gaussian (a 4-step walk can
only land on 5 discrete values) while a fine one is not.

### The Brownian bridge

$$B(t) = W(t) - \frac{t}{t_{\max}} W(t_{\max})$$

subtracts off exactly enough of the free path's randomness to force
$B(0) = B(t_{\max}) = 0$ while keeping $B$ Gaussian (a linear
transformation of jointly Gaussian variables stays Gaussian). Its
variance, $\mathrm{Var}(B(t)) = t(t_{\max}-t)/t_{\max}$, is zero at
both pinned endpoints and largest at the midpoint, which the tests
confirm at several checkpoints against simulated variance.

## Application: does EUR/USD log-price scale like Brownian motion?

If EUR/USD log-prices behaved like a Brownian motion with drift, the
variance of a $q$-day cumulative log return would have to equal
exactly $q$ times the variance of a single day's log return, since
that is what Brownian scaling means at the level of increments. This
is testable directly: split the 6,930 log returns into non-overlapping
blocks of $q$ days, compare the variance of the block sums to $q$
times the one-day variance.

**Building the test statistic's exact standard error required
correcting a first-pass error.** The commonly cited asymptotic result
for this kind of variance ratio test, $\sqrt{m}(\mathrm{VR}(q)-1) \to
N(0, 2(q-1))$, gives a standard error of $\sqrt{2(q-1)/m}$. Plugging
that into a Monte Carlo calibration check (simulate many independent
i.i.d. Gaussian samples, confirm the test rejects a true null at close
to its nominal 5% rate) failed badly: the observed rejection rate was
0% instead of about 5%, and the z-statistics had roughly a fifth of
the variance a standard normal should have. Re-deriving the variance
by hand resolved this. The one-period variance estimator $S_1$ and the
$q$-period block variance estimator $S_q$ are **not independent**,
since they are computed from overlapping information (the same $n =
mq$ observations feed both): expanding $Y_j^2$ into within-block
squared terms plus within-block cross terms shows $S_q = qS_1 + C$
where $C$ is uncorrelated with $S_1$ for Gaussian data (its terms are
odd-order cross moments, which vanish), giving
$\mathrm{Cov}(S_1, S_q) = q\,\mathrm{Var}(S_1)$. Feeding
$\mathrm{Var}(S_1) = 2/n$, $\mathrm{Var}(S_q) = 2q^2/m$, and this
covariance through the delta method for $\mathrm{VR} = S_q/(qS_1)$
gives

$$\mathrm{Var}(\mathrm{VR}(q)) = \frac{2(q-1)}{mq},$$

a factor of $q$ smaller than the commonly quoted formula. This
corrected formula, not the textbook one, is what is implemented in
`variance_ratio_test`, and it is the one that passes the Monte Carlo
calibration check (rejection rate lands between 2% and 9% across 500
independent replications at a nominal 5% level, and the z-statistics
have unit variance to within the Monte Carlo noise). This is a useful
reminder that "standard" asymptotic formulas often come with implicit
assumptions (here, independence between numerator and denominator)
that do not automatically hold for a specific estimator construction,
and that a from-scratch implementation is only as trustworthy as the
calibration check behind it, not the citation.

**The result on real EUR/USD data**: VR(q) sits close to 1 for every
horizon tested (2, 5, 10, 20, and 50 trading days), with p-values
between 0.33 and 0.88, none rejecting the null. This is consistent
with, and reinforces, Project 2's finding that the day-to-day sign
process shows no detectable lag-1 dependence: a variance ratio test is
fundamentally a test for linear (covariance) dependence across
horizons, and it is finding none here either. Put together with
Projects 2 and 3's findings of volatility clustering (dependence in
the *magnitude* and *timing* of large moves, not in returns
themselves), the overall picture is coherent: EUR/USD log-returns
behave close to uncorrelated at the mean, consistent with Brownian
scaling, while the variance of those returns is itself
time-dependent, which is exactly the kind of process a driftless
Brownian motion with constant volatility does not capture and a
stochastic-volatility model would.

## Limitations

- Box-Muller occasionally produces extreme values when $U_1$ is very
  close to 0 (large $\sqrt{-2\ln U_1}$); the implementation floors
  $U_1$ at $10^{-300}$ to avoid a literal `math.log(0)` crash, which
  has no practical effect on the sampled distribution but is worth
  noting as a numerical safeguard rather than a mathematical one.
- The Donsker convergence check uses excess kurtosis as a proxy for
  distributional convergence; it is suggestive, not a full
  Kolmogorov-Smirnov-style test of the entire path distribution.
- The variance ratio test assumes $q$ divides evenly enough into the
  sample that non-overlapping blocking does not waste much data (the
  implementation simply truncates any remainder); for very large $q$
  relative to $n$, the number of usable blocks $m$ shrinks and the
  test loses power, which is why `variance_ratio_test` refuses to run
  when $m < 30$.
- The corrected asymptotic variance formula above assumes Gaussian
  increments under the null (used to zero out the odd-moment cross
  terms); real returns have fatter tails than Gaussian, so the formula
  is an approximation for real data, though the Monte Carlo
  calibration check uses genuinely Gaussian synthetic data specifically
  to test the formula on the terms it was derived under.

## CLI examples

```bash
# Simulate a path and inspect its quadratic/total variation
python3 cli.py simulate --t 5 --steps 2000 --seed 1

# Reflection principle / first-passage probability
python3 cli.py reflection --a 1.0 --t 2.0

# First-passage time density and CDF
python3 cli.py passage --a 1.0 --t 2.0

# Donsker-scaled random walk approximating Brownian motion
python3 cli.py donsker --steps 5000 --t 3 --seed 1

# Brownian bridge pinned at both ends
python3 cli.py bridge --t 4 --steps 400 --seed 1

# Test EUR/USD log returns against Brownian scaling
python3 cli.py fxscaling

# Cross-check every formula against scipy/numpy oracles and Monte Carlo calibration
python3 cli.py verify
```

## Tests

```bash
python3 -m pytest test_brownian_motion.py -v
```

42 tests covering: Box-Muller normal generation and the normal CDF,
path simulation with and without drift, quadratic and total variation
including their opposite behaviour under partition refinement, the
reflection principle against both Monte Carlo and its equivalent
scipy statement, the first-passage density's numerical integral, the
Donsker approximation's convergence toward Gaussian, the Brownian
bridge's variance formula, and the EUR/USD variance ratio application
including a synthetic check that the test correctly detects induced
autocorrelation.
