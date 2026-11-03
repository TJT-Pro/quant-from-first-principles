# stochastic-differential-equations

A stochastic differential equation engine built from scratch: Brownian
motion and its defining (and famously counter-intuitive) properties,
the Euler-Maruyama and Milstein numerical schemes, empirical strong
and weak convergence order verification, numerical verification of
Ito's lemma, and exact (discretization-free) simulation of geometric
Brownian motion and the Ornstein-Uhlenbeck process, applied to real
EUR/USD price and volatility data.

This repository is independent of the sibling repositories and reuses
none of their code, though the application connects directly to the
volatility-mean-reversion investigation in ode-engine, this time with
a genuinely stochastic (not deterministic) model.

## Project structure

```
stochastic-differential-equations/
    README.md
    data/DEXUSEU_returns.csv
    brownian_motion.py       simulation and verification of Wiener process properties
    sde_solvers.py           Euler-Maruyama and Milstein schemes
    models.py                exact GBM and OU simulation via closed-form transitions
    convergence_analysis.py  empirical strong/weak convergence order verification
    ito_calculus.py          numerical verification of Ito's lemma
    fx_application.py        GBM/OU calibration to real EUR/USD data
    test_*.py
    cli.py
```

## Status

55 tests, 13/13 verify checks, all passing.

## Setup

Python 3.11+. All simulation code uses only the standard library.
`cli.py verify` uses numpy and scipy as independent oracles.

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest test_brownian_motion.py test_sde_solvers.py test_models.py \
    test_convergence_analysis.py test_ito_calculus.py test_fx_application.py -v
```

Note: several tests estimate Monte Carlo quantities across thousands
of simulated paths and take a noticeable fraction of a minute in
total; none are slow by accident.

## Running the oracle

```bash
python3 cli.py verify
```

## The data

`data/DEXUSEU_returns.csv`: daily EUR/USD exchange rates from the
Federal Reserve (FRED series DEXUSEU), 1999-01-04 to 2026-08-21
(6,931 price levels, 6,930 daily returns), the same file used in the
sibling repositories.

## Mathematical background

### Brownian motion: finite quadratic variation, infinite total variation

Standard Brownian motion has independent increments
W_{t+dt}-W_t ~ N(0,dt). Its quadratic variation,
sum_i (W_{t_{i+1}}-W_{t_i})^2, converges to exactly t as the partition
is refined, a famously counter-intuitive property (verified here
directly on a single path at increasing resolutions: 0.70, 1.12, 1.01,
0.98 as the step count grows from 10 to 10,000, converging toward the
theoretical limit of 1.0): Brownian motion accumulates a finite,
deterministic amount of squared fluctuation over any interval, even
though its ordinary (unsquared) total variation is infinite. This
single fact is the entire reason stochastic calculus needs a
second-order correction term that ordinary calculus does not:
(dW_t)^2 behaves like dt, not like a negligible higher-order term the
way it would for any smooth, differentiable path.

### Euler-Maruyama and Milstein: the stochastic analogues of Euler and its correction

For dX_t = mu(X_t,t)dt + sigma(X_t,t)dW_t, Euler-Maruyama is the
direct stochastic analogue of forward Euler,
X_{n+1} = X_n + mu(X_n,t_n)dt + sigma(X_n,t_n)dW_n. Just as forward
Euler for ODEs has O(h) global error from truncating a Taylor
expansion at first order, Euler-Maruyama's strong error (how close a
single simulated path gets to the true path driven by the same
Brownian motion) is O(h^0.5), one power of h worse than its
deterministic counterpart, because the stochastic Taylor expansion it
truncates has a dW^2 ~ dt term (from Brownian motion's quadratic
variation) that an ordinary Taylor expansion would treat as negligible
but Ito calculus cannot. The Milstein scheme adds that missing term
back in:

    X_{n+1} = X_n + mu*dt + sigma*dW_n + 0.5*sigma*(d sigma/d x)*(dW_n^2 - dt)

raising the strong order to O(h). Verified directly on GBM (known
exact solution) with matched Brownian increments at every resolution:
Euler-Maruyama's measured strong order is 0.470-0.488 across
independent runs, Milstein's is 0.974-0.988, both close to the
theoretical 0.5 and 1.0, and at a fixed step count Milstein's error is
routinely 100-1500x smaller than Euler-Maruyama's on the same path.

### A real design bug in the Ito's-lemma verification, caught by a genuinely nonlinear test case

Applying Ito's lemma to a transform Y_t=f(X_t) gives Y_t's own drift
and diffusion, mu_Y = f'(x)*mu(x,t) + 0.5*f''(x)*sigma(x,t)^2,
sigma_Y = f'(x)*sigma(x,t), written as functions of the original state
x. An earlier version of this project's verification function passed
these functions directly to Euler-Maruyama as if they described Y's
own autonomous process, which silently calls them with Y's own
current value, not X's. This happened to work for the flagship GBM
log-transform example, because mu_Y and sigma_Y there reduce to
constants (mu-0.5*sigma^2 and sigma, independent of x), masking the
bug entirely. It broke immediately and loudly (a ZeroDivisionError) on
a genuinely x-dependent test case, f(x)=x^2 applied to an
Ornstein-Uhlenbeck process, where f'(x)=2x actually depends on the
state. The fix tracks X_t explicitly alongside Y_t and evaluates every
Ito-lemma coefficient at X_t's own current value at each step, which
is correct for any twice-differentiable f, invertible or not; both the
constant-coefficient GBM case and the genuinely x-dependent OU case
are now checked directly in the test suite, specifically so the fix
cannot silently regress back to the version that only worked by
coincidence.

### Weak convergence: a case where Monte Carlo noise had to be worked around, not just tolerated

Weak error, |E[f(X_T^h)] - E[f(X_T)]|, compares distributions rather
than individual paths, and Euler-Maruyama's weak order is O(h), better
than its own strong order, since averaging across many paths cancels
the leading-order pathwise error that dominates the strong-error
comparison. A direct Monte Carlo measurement of this for f(x)=x on
GBM failed outright: with 200,000 paths per step size, the measured
errors (0.0012, 0.00073, 0.00066, 0.000028 across four step sizes)
gave an estimated "order" of 1.65, which was really sampling noise of
the same magnitude as the underlying bias itself, not a real
convergence trend. The fix exploits that for a linear drift
mu(x)=mu*x (as in GBM), E[X_N] under Euler-Maruyama follows the
exactly deterministic recursion E[X_N] = X_0*(1+mu*h)^N, needing no
Monte Carlo sampling at all for this specific test function. This
noise-free measurement gives a clean empirical weak order of 0.998,
matching the theoretical 1.0 almost exactly; the general
Monte-Carlo-based weak_error function remains available (and
necessary) for nonlinear drifts where no such closed-form shortcut
exists.

### Exact simulation via known transition distributions

Both GBM and the Ornstein-Uhlenbeck process have known closed-form
transition distributions, letting them be simulated with zero
discretization error at any step size, however large: GBM's exact
solution X_t = X_0*exp((mu-0.5*sigma^2)*t + sigma*W_t) follows from
applying Ito's lemma to ln(x) (see below); OU's transition
X_{t+dt} | X_t ~ N(X_t*e^{-theta*dt} + mu*(1-e^{-theta*dt}),
sigma^2/(2*theta)*(1-e^{-2*theta*dt})) follows from an
integrating-factor argument applied to d(e^{theta*t}*X_t) combined
with Ito's isometry. Verified directly: simulating OU with a single,
large 5-year time step gives the identical terminal mean and variance
(to within Monte Carlo tolerance) as simulating it with 500 small
steps over the same horizon, a property Euler-Maruyama applied to the
same SDE would not have (its single-large-step approximation would be
badly biased).

### Ito's lemma: the general formula and a fully worked, exactly checkable example

For Y_t=f(X_t), Ito's lemma states
dY_t = [f'*mu + 0.5*f''*sigma^2]dt + f'*sigma*dW_t. For GBM and
f(x)=ln(x): f'(x)=1/x, f''(x)=-1/x^2, so f'(x)*mu(x) = mu*x/x = mu and
0.5*f''(x)*sigma(x)^2 = 0.5*(-1/x^2)*(sigma*x)^2 = -0.5*sigma^2, giving
d(ln X_t) = (mu-0.5*sigma^2)dt + sigma*dW_t with no remaining
x-dependence at all: ln(X_t) is a drifted Brownian motion outright.
Checked against a single explicit Brownian path's realised
ln(X_T)-ln(X_0), matching the closed-form prediction to
floating-point precision (difference exactly 0.0), not merely
approximately, since this specific case is the exact GBM solution
itself.

## Application: calibrating stochastic models to real EUR/USD data

**GBM, calibrated by maximum likelihood.** Daily log returns under GBM
are exactly Normal((mu-0.5*sigma^2)*dt, sigma^2*dt), giving closed-form
MLEs directly from the sample mean and standard deviation. On the
full 6,930-observation EUR/USD series: mu_hat=+0.0038 (essentially
zero annualized drift, consistent with every sibling repository's
"no linear predictability" finding) and sigma_hat=0.0913 (9.1%
annualized volatility, a plausible order of magnitude for a major
currency pair).

**OU, calibrated to realized volatility via a genuinely different
route than ode-engine's.** The 20-day rolling realized volatility
series is fit by exploiting that OU's Euler-Maruyama discretization is
exactly an AR(1) model, X_{t+1} = a + b*X_t + eps, with
theta=(1-b)/dt, mu=a/(1-b), sigma=sqrt(Var(eps)/dt) inverted directly
from an ordinary least-squares fit. This gives theta_hat=3.79
(half-life 46.1 days), mu_hat=0.0052 (long-run daily vol level,
about 8.3% annualized, consistent with the GBM sigma_hat),
sigma_hat_OU=0.0054. This is a completely different calibration method
from ode-engine's ACF-decay-based fit of a deterministic
mean-reversion ODE to the same underlying phenomenon (there, a
roughly 73-day half-life over a 300-lag window); the two independent
projects landing on broadly the same order of magnitude for
volatility's mean-reversion speed, via unrelated methods built without
reference to each other, is a meaningful cross-check neither project
was specifically built to produce.

**The real, quantified finding: GBM under-predicts real dispersion,
and increasingly so at longer horizons.** Comparing the empirical
standard deviation of real k-day-ahead log price changes against the
theoretical GBM prediction sigma*sqrt(k/252) at the same calibrated
sigma_hat:

| horizon | real std | GBM-theoretical std | ratio |
|---|---|---|---|
| 5 days | 0.01291 | 0.01286 | 1.0043 |
| 20 days | 0.02589 | 0.02571 | 1.0067 |
| 60 days | 0.04554 | 0.04454 | 1.0227 |

The ratio is small but consistently above 1, and grows with the
horizon: real dispersion exceeds a constant-volatility model's
prediction by 0.4% at 5 days but 2.3% at 60 days. This is precisely
the signature of volatility clustering (established independently
throughout this entire project family, now confirmed via yet another,
genuinely different method): GBM assumes i.i.d. increments with a
single fixed volatility, so its variance scales exactly with
sqrt(k); a process whose volatility is itself time-varying and
persistent (as the OU-calibrated mean-reversion above directly
measures) accumulates more variance at longer horizons than simple
sqrt(k)-scaling predicts, precisely because periods of elevated
volatility cluster together rather than averaging out independently
period to period.

## Limitations

- `sde_solvers.milstein`'s finite-difference fallback for
  d(sigma)/d(x) uses a fixed step h=1e-5; for a diffusion coefficient
  with very high curvature this fixed step could itself introduce
  meaningful approximation error on top of the scheme's own O(h)
  discretization error, not automatically detected or adapted for.
- `convergence_analysis.weak_error`'s general Monte Carlo route
  remains genuinely noisy for nonlinear drifts with no closed-form
  shortcut analogous to `linear_drift_exact_mean_error`; demonstrating
  a clean weak-order trend for a nonlinear SDE would need substantially
  larger sample sizes than this project's test suite budgets for.
- `fx_application.calibrate_ou_via_ar1` assumes the input series is
  already stationary (genuinely mean-reverting); applied to a series
  that is not (a random walk, for instance) it would still return a
  numerical answer (a very small or even negative theta_hat) rather
  than raising an error, leaving the caller responsible for checking
  that the AR(1) coefficient b is meaningfully below 1 before trusting
  the result.

## CLI examples

```bash
# Verify Brownian motion's defining properties
python3 cli.py brownian --t-end 1.0 --n-steps 100 --n-paths 10000

# Measure strong convergence order for Euler-Maruyama and Milstein on GBM
python3 cli.py convergence --mu 0.05 --sigma 0.3 --x0 1.0 --t-end 1.0

# Verify Ito's lemma on the GBM log-transform
python3 cli.py ito --mu 0.05 --sigma 0.3 --x0 1.0 --t-end 1.0

# Calibrate GBM and OU models to real EUR/USD data
python3 cli.py fxcalibrate

# Cross-check every simulation and convergence claim against
# scipy/numpy oracles and known closed-form theory
python3 cli.py verify
```
