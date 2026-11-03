# Project 6: First-Passage Problems

The same two-boundary hitting problem Projects 1 and 5 already solved,
now solved a third way (as an absorbing Markov chain) purely as a
cross-check, then extended to continuous time (drifted Brownian
motion) and to a single moving barrier, and finally applied to a
concrete, trader-relevant question: how well does a calibrated
constant-volatility diffusion model actually predict whether a
stop-loss or take-profit gets hit first on real EUR/USD data.

## Mathematical background

### Triangulating the discrete two-boundary problem three ways

The problem is always the same: a walk starts at 0, takes steps of
+1 (probability $p$) or -1 (probability $1-p$), and stops the first
time it reaches $a>0$ or $-b<0$. Three completely different arguments
answer the same two questions (the probability of hitting $a$ first,
and the expected time to stop), and getting the identical number from
each is a much stronger correctness signal than trusting any one of
them alone.

**Combinatorial (Project 1).** A difference equation on the hitting
probability as a function of starting position, solved directly,
gives $\alpha = b/(a+b)$ for $p=0.5$.

**Martingale / optional stopping (Project 5).** The exponential
martingale $\theta^{S_n}$ with $\theta=(1-p)/p$, combined with the
optional stopping theorem, gives the general-$p$ formula
$\alpha = (1-\theta^{-b})/(\theta^a - \theta^{-b})$.

**Absorbing Markov chain (this project).** The states
$\{-b+1, \dots, a-1\}$ are transient, and $a$, $-b$ are absorbing.
Writing $h_i$ for the probability of eventual absorption at $a$
starting from transient state $i$, conditioning on the first step
gives $h = Qh + r$, where $Q$ is the transition matrix restricted to
transient states and $r$ is the probability of moving directly to $a$
from each transient state. This is $(I-Q)h = r$, a linear system
solved directly by `solve_linear_system` (Gaussian elimination with
partial pivoting) rather than iteratively, with no need to invert $I-Q$
explicitly, since only the single starting state's value is needed.
The same linear-algebra machinery gives the expected time to
absorption from $(I-Q)t = \mathbf{1}$ (each transient state's expected
remaining time is one step plus the expectation from its next state),
which is exactly the classical **fundamental matrix** result for
absorbing Markov chains, applied here to a problem this project has
already solved two other ways.

`verify` confirms all three methods agree on both the symmetric
closed form and Project 5's general-$p$ martingale formula, across
several parameter sets, and confirms all of them agree with direct
Monte Carlo simulation.

### From discrete to continuous: the drifted Brownian motion analogue

Project 5 solved the discrete two-boundary problem for a biased coin.
The direct continuous-time analogue is $dX_t = \mu\,dt + \sigma\,dW_t$
started at 0, absorbed at $a>0$ or $-b<0$. The construction mirrors
the discrete exponential martingale exactly: $M_t = e^{\theta X_t}$ is
a martingale precisely when Ito's lemma's drift term vanishes,

$$\theta\mu + \tfrac12\theta^2\sigma^2 = 0 \implies \theta = -\frac{2\mu}{\sigma^2}$$

(besides the trivial root $\theta=0$, which is why $\mu=0$ needs a
different martingale, $X_t$ itself, giving back the familiar
$b/(a+b)$). Optional stopping gives $E[M_T]=1$, i.e.
$\alpha e^{\theta a} + (1-\alpha)e^{-\theta b} = 1$, solved directly
in `bm_ruin_probability`. Expected duration follows the same pattern
as the discrete case: $X_t - \mu t$ is a martingale when $\mu\ne0$,
giving $E[T] = (\alpha a - (1-\alpha)b)/\mu$; at $\mu=0$,
$X_t^2 - \sigma^2 t$ is used instead, giving $E[T] = (\alpha a^2 +
(1-\alpha)b^2)/\sigma^2$ (the direct continuous analogue of the
discrete $ab$ formula). Both are checked against Euler-Maruyama
simulation of the SDE.

### A single moving barrier: the drift-adjusted reflection principle

Project 4 derived $P(\max_{s\le t}W_s \ge a) = 2(1-\Phi(a/\sqrt t))$
for driftless Brownian motion. With drift, Girsanov's theorem (a
change of probability measure that removes the drift term at the cost
of reweighting the probability of each path by
$\exp(\theta W_t - \tfrac12\theta^2 t)$ for the right $\theta$) gives

$$P\left(\max_{0\le s\le t} X_s \ge a\right) = \Phi\!\left(\frac{\mu t - a}{\sigma\sqrt t}\right) + e^{2\mu a/\sigma^2}\,\Phi\!\left(\frac{-\mu t - a}{\sigma\sqrt t}\right).$$

Setting $\mu=0$ collapses the two terms into
$2(1-\Phi(a/\sigma\sqrt t))$ exactly, recovering Project 4's formula
as a special case, checked directly in `verify`, alongside a Monte
Carlo cross-check that surfaced a real numerical subtlety worth
recording: simulating the path at only a few hundred discrete time
steps systematically *underestimates* the true continuous-time
crossing probability, because the path can cross the barrier and
return between two monitoring points without being detected at either
one. This is the well-documented discrete-monitoring bias for barrier
crossings (Broadie, Glasserman & Kou, 1997, in the context of
discretely-monitored barrier options), and it shrinks slowly, at rate
$O(1/\sqrt{n})$ in the number of monitoring steps $n$: the first
Monte Carlo check attempted here, at 500 steps, missed the formula by
about 0.02 (empirical 0.4996 against a formula value of 0.5192);
raising the step count to 5,000 brought the gap down to about 0.005,
consistent with the $O(1/\sqrt n)$ rate. The fix was not to loosen the
tolerance arbitrarily but to simulate finely enough that a known,
named, rate-quantified bias falls inside a reasonable tolerance.

## Application: does a calibrated diffusion model correctly predict stop-loss / take-profit outcomes on real EUR/USD data?

$\mu$ and $\sigma$ are calibrated directly from the sample mean and
population standard deviation of the 6,930 daily log returns:
$\hat\mu \approx -1.6\times10^{-6}$ per day (indistinguishable from
zero, consistent with every earlier project's finding of no
meaningful drift) and $\hat\sigma \approx 0.00575$ per day.

For a symmetric bracket set at 10 daily sigmas each way
(target = stop = $10\hat\sigma \approx 0.0575$ in log-return units),
the calibrated model's infinite-horizon formula predicts almost
exactly a 50/50 split (49.9% / 50.1%) and an expected time to
resolution of about 100 days, both sensible given the near-zero
estimated drift.

`historical_stop_target_backtest` answers the same question a
different way: instead of trusting the model, walk forward along
every one of the 6,730 possible 200-day windows in the *actual*
historical log-return series and record what really happened. Over a
matching 200-day horizon, this genuine historical exercise gives
40.6% target-first, 37.8% stop-first, and **21.6% unresolved**:
neither bracket hit within 200 days. Simulating the calibrated model
over the identical finite 200-day horizon (rather than using the
infinite-horizon formula, for a fair comparison) gives an unresolved
fraction of only about 14%. Real EUR/USD price paths sit inside a
symmetric 10-sigma bracket without resolving noticeably more often
than a constant-volatility random walk with the same overall mean and
variance would predict.

This is a coherent capstone to everything the earlier projects found.
Projects 2 through 4 established that EUR/USD daily returns behave
close to a fair, unpredictable coin flip in their mean and roughly
Brownian in their variance scaling, but with time-varying (clustered)
volatility rather than the constant volatility a plain diffusion model
assumes. A period of unusually low realised volatility, exactly the
kind of episode volatility clustering produces, is precisely what
would cause a real price path to wander less than a constant-volatility
model expects and leave a fixed bracket unresolved for longer than
that model predicts. The stop-loss/take-profit backtest is, in effect,
a very concrete, practically-motivated way of re-discovering the same
volatility-clustering conclusion Projects 2 and 3 reached through a
transition matrix and an arrival-process dispersion test respectively:
three unrelated diagnostics, three different mathematical tools, one
consistent picture of the data.

## Limitations

- The historical backtest uses overlapping 200-day windows (each
  window shares 199 of its 200 days with its neighbour), so the
  6,730 nominal trials are far from independent; the reported
  percentages are honest point estimates of what actually happened
  historically, but they should not be treated as having the
  precision that 6,730 independent trials would imply, and no formal
  significance test is attempted on the gap between the historical and
  model unresolved fractions for this reason.
- The absorbing-chain method's state space grows linearly with $a+b$;
  it is exact and fast for the boundary sizes used in this project's
  tests (single digits to low tens), but would need a sparse or
  iterative linear solver for much larger boundaries.
- The continuous double-barrier and single-barrier formulas assume
  constant $\mu$ and $\sigma$ over the entire horizon, which is
  exactly the assumption the application section shows real EUR/USD
  data violates; the formulas are the correct null-hypothesis
  benchmark, not a claim that they describe the real process.
- The discrete-monitoring bias discussed above is a property of
  *simulating* a continuous barrier-crossing probability at finitely
  many time steps, not a property of the closed-form formulas
  themselves, which are exact for genuinely continuous monitoring;
  this project does not implement the Broadie-Glasserman-Kou
  continuity-correction adjustment itself, only verifies that the bias
  behaves at the documented rate.

## CLI examples

```bash
# Discrete two-boundary ruin via the absorbing-chain method
python3 cli.py ruin --a 8 --b 12 --p 0.5

# Continuous two-boundary ruin for drifted Brownian motion
python3 cli.py bmruin --a 1.0 --b 1.5 --mu 0.3 --sigma 1.0

# Drift-adjusted single-barrier exceedance probability
python3 cli.py barrier --a 1.2 --t 2.0 --mu 0.25 --sigma 1.0

# Stop-loss/take-profit: calibrated model vs real historical backtest
python3 cli.py fxstop --multiple 10 --horizon 200

# Cross-check every formula: three-way triangulation, scipy/numpy
# oracles, and Monte Carlo simulation
python3 cli.py verify
```

## Tests

```bash
python3 -m pytest test_first_passage.py -v
```

27 tests covering: the absorbing-chain method against both the
closed-form symmetric result and Monte Carlo simulation, the linear
solver itself, the continuous double-barrier formulas against
Euler-Maruyama simulation, the drift-adjusted reflection principle
against its own zero-drift special case and against Monte Carlo, and
the EUR/USD application including the central finding that real price
paths leave a symmetric stop/target bracket unresolved more often than
the calibrated model predicts.

---

## Repository-wide synthesis

Across all six projects, one coherent statistical picture of EUR/USD
built up, each project probing it with a different tool that had no
knowledge of what the others found:

- **Project 1** found the raw sign sequence's 252-day windowed
  variance running above the i.i.d. random-walk prediction, but
  flagged this as thin evidence from only 27 windows.
- **Project 2** tested that suspicion formally: no significant lag-1
  dependence in the 2-state sign chain (p=0.33), but strong dependence
  in a 5-state magnitude-based chain (p=0.0004): the sign is close to
  fair, the size of moves is not independent of the recent past.
- **Project 3** confirmed the same magnitude-clustering conclusion
  from a third angle: large moves arrive far more clustered in time
  than a homogeneous Poisson process would produce (index of
  dispersion around 12.6, p on the order of $10^{-40}$).
- **Project 4** confirmed no linear (variance-ratio) predictability in
  returns at any horizon tested, consistent with Project 2's sign
  finding, while deriving and correcting the exact asymptotic variance
  of its own test statistic along the way.
- **Project 5** confirmed the mean process is a clean martingale
  difference sequence (no lag-1 predictability in a direct regression
  test), but found that a convex, streak-sensitive strategy (the
  doubling system) is sensitive to real run-length structure a simple
  lag-1 test cannot see.
- **Project 6** closed the loop with a concrete trading construct: a
  calibrated constant-volatility diffusion model misjudges how often a
  real stop-loss/take-profit bracket resolves, in exactly the
  direction volatility clustering would predict.

No single test told the whole story. Sign, magnitude, arrival timing,
linear dependence, path-dependent strategy payoff, and a concrete
risk-management construct all needed to be checked separately, because
each one is sensitive to a different feature of the same underlying
process, and a model that passes one of these checks cleanly (as a
plain driftless Brownian motion does, for the mean) can still fail
badly at another (as the same model does, for volatility structure and
its practical consequences).
