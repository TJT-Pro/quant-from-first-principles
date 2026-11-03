# black-scholes

A Black-Scholes / Garman-Kohlhagen derivatives engine built from
scratch: the closed-form pricing formula and its Greeks, two implied
volatility solvers, a Cox-Ross-Rubinstein binomial tree (European and
American exercise), a Monte Carlo pricer with antithetic variance
reduction, and a delta-hedging P&L simulator applied to real EUR/USD
data. Every pricing and Greek formula is checked against at least one
independent route: a different closed-form recomputation, a
finite-difference numerical derivative, an unrelated pricing method
(binomial or Monte Carlo), or an external scipy/numpy oracle.

This repository is independent of `math-implementations` and
`stochastic-processes` but follows the same standards and reuses none
of their code.

## Project structure

```
black-scholes/
    README.md
    data/DEXUSEU_returns.csv
    black_scholes.py          core pricing formula, Greeks, GBM, calibration
    implied_volatility.py     Newton-Raphson and bisection solvers
    binomial_tree.py          Cox-Ross-Rubinstein tree, European and American
    monte_carlo.py            GBM Monte Carlo pricer, antithetic variates
    hedging_simulation.py     delta-hedging P&L on real historical data
    test_black_scholes.py
    test_implied_volatility.py
    test_binomial_tree.py
    test_monte_carlo.py
    test_hedging_simulation.py
    cli.py
```

## Status

84 tests, 17/17 verify checks, all passing.

## Setup

Python 3.11+. All pricing and simulation code uses only the standard
library. `cli.py verify` needs numpy and scipy as independent oracles:

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest test_black_scholes.py test_implied_volatility.py \
    test_binomial_tree.py test_monte_carlo.py test_hedging_simulation.py -v
```

## Running the oracle

```bash
python3 cli.py verify
```

## The data

`data/DEXUSEU_returns.csv` holds daily EUR/USD exchange rates sourced
from the Federal Reserve (FRED series DEXUSEU), 1999-01-04 to
2026-08-21 (6,931 price levels, 6,930 daily returns), the same file
used in `stochastic-processes`. Columns: `date`, `price` (U.S. dollars
per one euro), `return` (simple daily return), `log_return` (daily log
return). The file is committed as-is; no external fetch is needed.

## Mathematical background

### Geometric Brownian motion and the lognormal solution

The underlying's price is modelled as $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$.
Applying Ito's lemma to $\ln S_t$ removes the multiplicative structure:

$$d(\ln S_t) = \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma\,dW_t,$$

an ordinary drift-diffusion in $\ln S$, which integrates directly to

$$S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma\sqrt{t}\,Z\right], \qquad Z\sim N(0,1).$$

The $\sigma^2/2$ term is the Ito correction: ignoring it would give the
right answer for $E[S_t] = S_0 e^{\mu t}$ but the wrong distribution
for $S_t$ itself, since $\ln S_t$ is what is actually normally
distributed, not $S_t$. `simulate_gbm_terminal` implements this
directly, and the Monte Carlo pricer in `monte_carlo.py` uses it under
the risk-neutral measure (drift $r-q$ in place of $\mu$) to jump
straight to the terminal price without simulating every intermediate
step, since only $S_T$ matters for a European payoff.

### Deriving the Black-Scholes PDE

Consider a portfolio $\Pi = V - \Delta S$, long one option and short
$\Delta$ shares. Ito's lemma on $V(S,t)$ gives

$$dV = V_t\,dt + V_S\,dS + \frac{1}{2}\sigma^2 S^2 V_{SS}\,dt.$$

Choosing $\Delta = V_S$ eliminates the random $dS$ term from $d\Pi$:

$$d\Pi = \left(V_t + \frac{1}{2}\sigma^2 S^2 V_{SS}\right)dt.$$

Since $\Pi$ is now instantaneously riskless, no-arbitrage requires it
to earn exactly the risk-free rate, $d\Pi = r\Pi\,dt = r(V - SV_S)\,dt$.
Equating the two expressions for $d\Pi$ gives the **Black-Scholes
PDE**:

$$V_t + \frac{1}{2}\sigma^2 S^2 V_{SS} + rSV_S - rV = 0.$$

With a continuous yield $q$ on the underlying (a dividend, or for FX
options the foreign risk-free rate), the risk-neutral drift of $S$
becomes $r-q$ instead of $r$, giving

$$V_t + \frac{1}{2}\sigma^2 S^2 V_{SS} + (r-q)SV_S - rV = 0,$$

the version this project's closed-form solution actually solves. A
full finite-difference numerical solution of this PDE (explicit,
implicit, Crank-Nicolson) is deliberately out of scope here and
reserved for the separate `numerical-pde` project, which applies the
same machinery to this exact PDE as one of its extensions; this
project solves it analytically and validates that solution against two
independent numerical pricing methods instead.

### Solving the PDE: risk-neutral valuation and the closed form

The Feynman-Kac theorem connects the PDE above to a risk-neutral
expectation: $V = e^{-rT}E^Q[\text{payoff}(S_T)]$, where under $Q$,
$S_T$ follows the lognormal law above with drift $r-q$ instead of the
real-world $\mu$ (the real-world drift drops out of the pricing
formula entirely, which is the celebrated, initially counterintuitive
result that a stock's expected return has no bearing on its option
price once volatility is fixed). Evaluating this expectation for the
call payoff $\max(S_T - K, 0)$ by splitting the integral at $S_T = K$
and working through the resulting lognormal integrals gives

$$C = Se^{-qT}N(d_1) - Ke^{-rT}N(d_2), \qquad
d_{1,2} = \frac{\ln(S/K) + (r-q\pm\sigma^2/2)T}{\sigma\sqrt{T}},$$

implemented in `d1_d2` and `call_price`. The put follows either by the
same argument applied to $\max(K-S_T,0)$, or directly from **put-call
parity**: a forward contract (long the underlying, financed at $r$,
net of the yield $q$) can be statically replicated by a long call and
a short put at the same strike, giving $C - P = Se^{-qT} - Ke^{-rT}$
regardless of which stochastic model prices either leg. Both routes
are implemented, and `put_call_parity_gap` checks the identity holds
to numerical precision (see `verify`) across a full grid of spot,
strike, maturity, rate, volatility, and yield combinations.

For FX options specifically, this dividend-adjusted formula **is**
the standard model (Garman & Kohlhagen, 1983): $r$ is the domestic
risk-free rate, $q$ is the foreign risk-free rate, since holding a
unit of foreign currency earns interest exactly the way a
dividend-paying stock does.

### Greeks: differentiating the closed form

Every Greek here is a direct partial derivative of the pricing
formula, cross-checked against a completely independent
finite-difference computation (`numerical_delta`, `numerical_gamma`,
`numerical_vega`, `numerical_theta`) that does not rely on having
differentiated anything correctly by hand. The one identity worth
naming explicitly: differentiating $C$ with respect to $S$ appears at
first to produce four terms (the derivative of $N(d_1)$ and of
$N(d_2)$ each contribute one), but two of them cancel exactly because

$$S\phi(d_1) = Ke^{-(r-q)T}\phi(d_2)$$

(a consequence of how $d_1$ and $d_2$ are related through $\phi$'s
Gaussian form), leaving the clean result $\Delta_{\text{call}} =
e^{-qT}N(d_1)$. Gamma and vega turn out identical for calls and puts
at the same strike, because put-call parity's right-hand side (a
forward) is linear in $S$ and does not depend on $\sigma$ at all, so
neither the second derivative in $S$ nor the derivative in $\sigma$
can tell a call from a put once the model parameters match.

### Implied volatility: two solvers, and a pitfall worth knowing about

**Bisection** needs only that the price is monotone in $\sigma$ (true,
since vega is strictly positive whenever an option has time value) and
a bracket containing the root; it halves the bracket every iteration
and is unconditionally reliable but slow (dozens of iterations for
high precision). **Newton-Raphson** uses
$\sigma_{n+1} = \sigma_n - (\text{price}(\sigma_n) - \text{market price})/\text{vega}(\sigma_n)$
and converges quadratically, typically in 3-6 iterations from a
reasonable starting guess, but can fail outright when vega is very
small (deep in/out-of-the-money, very short-dated options), which is
why `implied_vol_newton_with_bisection_fallback` exists: try the fast
method, fall back to the slow-but-reliable one on failure.

A subtler issue surfaced while testing this pair against each other,
and is worth stating plainly rather than glossing over: **for options
whose true price is already extremely small, an absolute residual
tolerance is a poor stopping criterion for either solver.** A concrete
example, verified directly in the test suite: $S=100$, $K=140$,
$T=0.1$, $r=0.02$, true $\sigma=0.15$ gives a call price on the order
of $10^{-13}$. Both solvers report success with a tiny residual (price
matched to within $10^{-8}$), but the recovered $\sigma$ lands around
0.16-0.18, not 0.15, because the pricing function is nearly flat near
zero in that regime and a wide band of volatilities all reprice the
option to within the same absolute tolerance. This is a genuine,
well-known limitation of absolute-tolerance implied-volatility solvers
in general, not a bug specific to this implementation; production
systems typically use a relative price tolerance, a minimum vega
threshold below which they decline to quote an implied volatility at
all, or both. Neither safeguard is implemented here, deliberately, so
that the test suite can demonstrate the failure mode directly rather
than hide it behind a fix that would make the pitfall invisible.

### The Cox-Ross-Rubinstein binomial tree

The tree parameters $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$, and
risk-neutral probability $p = (e^{(r-q)\Delta t} - d)/(u-d)$ are chosen
so that, at every single step (not merely in the limit), the tree's
expected growth matches the risk-neutral drift exactly:
$e^{(r-q)\Delta t} = pu + (1-p)d$ by construction, while $u$ and $d$
are chosen to match the *variance* of the underlying GBM process to
leading order as $\Delta t \to 0$. Backward induction then computes
each node's value as the discounted expectation of its two children;
for an **American** option, that continuation value is compared
against the immediate exercise payoff at every node, and the larger of
the two is kept, since the holder is always free to exercise early.
This is the one thing the closed-form Black-Scholes formula cannot do
at all, and the tree demonstrates exactly when early exercise matters:
an American call with **no** dividend/foreign yield is worth precisely
the same as the European call (giving up remaining time value early
is never optimal with nothing to gain by doing so), but with a large
enough yield $q$, the American call becomes strictly more valuable,
since exercising early captures that yield rather than continuing to
hold an option on an asset that keeps paying it out to someone else.
Both facts are verified directly. The tree's price also converges to
the closed-form Black-Scholes price as the number of steps grows,
though not perfectly monotonically (whether the strike happens to fall
exactly on a tree node at a given step count introduces small
oscillations), which `cli.py convergence` displays directly rather
than only asserting a final match.

### Monte Carlo pricing and antithetic variance reduction

The Monte Carlo pricer discounts the average simulated payoff,
$e^{-rT}\cdot\overline{\text{payoff}(S_T)}$, with a 95% confidence
interval from the sample standard error. Its defining, comparatively
slow convergence rate, standard error shrinking as
$O(1/\sqrt{N})$, is demonstrated directly in
`monte_carlo_convergence_series` and in the test suite (quadrupling
$N$ should roughly halve the standard error, and does).
**Antithetic variates** pair each standard normal draw $Z$ with its
mirror $-Z$, generating two paths per random draw instead of one.
Because $S_T$ is a monotone function of $Z$, this pairing induces
negative correlation between the two paths' payoffs for any monotone
payoff (calls and puts both qualify), which reduces the variance of
the *pair's average* without introducing any bias, since $Z$ and $-Z$
share the same marginal distribution. The reduction is verified
directly by comparing standard errors from matched-seed plain and
antithetic runs at the same path count.

## Application: delta-hedging a short option through real EUR/USD history

The self-financing replication argument behind the Black-Scholes PDE
derivation above is implemented directly, day by day, using the
*actual* historical EUR/USD path rather than a simulated one. Sell a
30-day at-the-money call for its model premium (using a fixed implied
volatility calibrated from the full 6,930-day sample, 9.13%
annualised), buy the model's delta in the underlying to hedge, and
hold the remainder in a cash account earning the domestic rate while
the hedge position earns the foreign yield. Each day, walk forward
along the real historical return for that day, recompute the new
delta from the new spot and the shrinking time to maturity (still
under the same fixed implied volatility), and rebalance. At maturity,
settle the option against its true payoff.

Run across 231 non-overlapping 30-day windows spanning the full
history, the average hedging P&L for the short call comes out slightly
positive (+0.00086, against a strike/premium scale where the
underlying itself is normalised to start at 1.0), because the average
*realized* volatility across windows (8.43%) sits a little below the
9.13% implied volatility used throughout. Far more informative than
that average, though, is the **correlation between each window's
hedging P&L and that window's (realized variance - implied variance)
gap: -0.80.** This is the discrete-hedging analogue of the well-known
continuous-time "gamma P&L" result: a short, delta-hedged option loses
money in windows where the market turns out more volatile than what
was charged for it, and makes money when the market turns out calmer,
in direct proportion to the gap between what actually happened and
what the constant-volatility model assumed. The mechanism was
validated first on synthetic GBM data with a deliberately mismatched
true volatility (Section `verify`, checks 11-12) before being trusted
on the ambiguous real data, and it produces exactly the predicted sign
in both settings.

This result closes the loop with a running theme:
`stochastic-processes` established, through four unrelated
diagnostics (a transition-matrix independence test, a Poisson
dispersion test, a variance ratio test, and a doubling-strategy
run-length analysis), that EUR/USD daily returns cluster in variance
even though they show essentially no linear or directional
predictability. A short-option seller who prices and hedges off a
single fixed volatility number is, in effect, betting that this
clustering will not matter over their specific holding period; the
-0.80 correlation here is the direct, quantified cost of that bet
going the wrong way, using nothing but real historical data and the
same replication argument that derives the Black-Scholes formula in
the first place.

## Limitations

- The delta-hedging simulation rebalances once per day; continuous
  rebalancing (the theoretical ideal behind the PDE derivation) would
  track the model more closely, and the daily-rebalancing P&L reported
  here includes a small amount of genuine discretization noise on top
  of the volatility-mismatch signal the correlation is measuring.
- The 231 windows used in the FX application are non-overlapping to
  keep the observations close to independent, at the cost of using
  less of the available history than an overlapping-window design
  would; this mirrors the same tradeoff made in
  `stochastic-processes/project6_first_passage`.
- The implied-volatility absolute-tolerance pitfall documented above
  is real and demonstrated directly, but no relative-tolerance or
  vega-floor safeguard is implemented; this is a deliberate choice to
  keep the failure mode visible rather than papered over.
- Domestic and foreign interest rates (`r`, `q`) are left as
  user-supplied parameters throughout rather than fetched from any
  live source; the FX hedging application uses `r=q=0` for simplicity,
  which is a reasonable approximation for a short-dated study but
  would matter more for longer-dated options or currency pairs with a
  larger interest-rate differential.
- The binomial tree's early-exercise comparisons (American vs European
  call with/without dividends) are demonstrated at specific parameter
  sets chosen to make the effect clear; the general boundary at which
  early exercise becomes optimal is not derived in closed form here.

## CLI examples

```bash
# Price a call or put
python3 cli.py price --s 100 --k 100 --t 1 --r 0.03 --sigma 0.2 --type call

# All Greeks
python3 cli.py greeks --s 100 --k 100 --t 1 --r 0.03 --sigma 0.2 --type call

# Implied volatility from a market price
python3 cli.py impliedvol --price 9.41 --s 100 --k 100 --t 1 --r 0.03 --type call

# Binomial tree, European or American
python3 cli.py binomial --s 100 --k 110 --t 1 --r 0.05 --sigma 0.2 --q 0.10 \
    --steps 500 --exercise american --type call

# Monte Carlo with antithetic variates
python3 cli.py montecarlo --s 100 --k 100 --t 1 --r 0.03 --sigma 0.2 \
    --paths 200000 --antithetic --type call

# Compare all three pricing methods' convergence side by side
python3 cli.py convergence --s 100 --k 100 --t 1 --r 0.03 --sigma 0.2 --type call

# Delta-hedge backtest on real EUR/USD history
python3 cli.py fxhedge --horizon 30

# Cross-check every formula against scipy/numpy oracles, finite
# differences, and independent pricing methods
python3 cli.py verify
```
