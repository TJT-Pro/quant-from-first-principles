# ode-engine

An ordinary differential equation solver engine built from scratch:
Euler, Improved Euler, Midpoint, and classical RK4, an adaptive
step-doubling controller, empirical convergence-order verification,
and two applications to real EUR/USD data: a calibrated volatility
mean-reversion equation, and the Kolmogorov forward equations for a
continuous-time regime-switching chain.

This repository is independent of the sibling repositories and reuses
none of their code, though the applications extend themes those
repositories established.

## Project structure

```
ode-engine/
    README.md
    data/DEXUSEU_returns.csv
    ode_solvers.py       Euler, Improved Euler, Midpoint, RK4, adaptive stepping
    error_analysis.py    empirical convergence order, Richardson extrapolation
    fx_application.py    mean-reversion calibration, Kolmogorov forward equations
    test_*.py
    cli.py
```

## Status

45 tests, 10/10 verify checks, all passing.

## Setup

Python 3.11+. All solver code uses only the standard library.
`cli.py verify` uses numpy and scipy as independent oracles.

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest test_ode_solvers.py test_error_analysis.py test_fx_application.py -v
```

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

### The initial value problem and the four fixed-step methods

Given $dy/dt = f(t,y)$, $y(t_0)=y_0$, each method here walks forward
in steps of size $h$, differing only in how many times $f$ is
evaluated per step and how those evaluations combine:

- **Euler**: $y_{n+1} = y_n + h f(t_n, y_n)$, one evaluation, following
  the tangent line. Matches the true solution's Taylor expansion
  through the first-order term, leaving an $O(h^2)$ local error per
  step and an $O(h)$ global error.
- **Improved Euler (Heun)**: predict with an Euler step, then average
  the slope at the start and the predicted endpoint. Equivalent to the
  trapezoidal rule for the step's integral, cancelling Euler's leading
  error term: $O(h^2)$ globally.
- **Midpoint**: estimate the slope once, at the interval's midpoint
  (reached via a half-Euler-step), and use that single estimate for
  the whole step. Also $O(h^2)$ globally, via the midpoint quadrature
  rule instead of the trapezoidal one, which is why it does not give
  numerically identical answers to Improved Euler despite sharing the
  same order.
- **Classical RK4**: four evaluations per step ($k_1, k_2$ at the
  midpoint via two different predicted positions, $k_3$, and $k_4$ at
  the endpoint), combined with Simpson's-rule weights $1{:}2{:}2{:}1$.
  $O(h^4)$ globally: halving $h$ should cut the error by roughly 16x.

**Every one of these theoretical orders is verified empirically, not
just stated.** `error_analysis.estimate_convergence_order` fits
$\log(\text{error}) = \log C + p\log h$ by least squares across five
step sizes on $dy/dt=-y$ (exact solution $e^{-t}$) and reads the
fitted slope off as the observed order. The result: Euler 1.011,
Improved Euler 2.051, Midpoint 2.051, RK4 4.054, matching theory to
within about 5% purely from measured numbers, confirmed again on a
second, unrelated ODE ($dy/dt=2y$) to rule out the result being an
artifact of one specific problem.

### A real bug, found by testing at a non-round step size

`solve_ivp` originally computed its number of steps as
`round((t_end - t0) / h)`, silently reaching $t_0 + h\cdot
\text{round}((t_{end}-t_0)/h)$ instead of $t_{end}$ whenever the
interval was not an exact multiple of $h$. This surfaced directly
while testing a harmonic oscillator integrated to $t_{end}=\pi$ with
$h=0.001$: $\pi/0.001 = 3141.59\ldots$ rounds to 3142 steps, landing
at $t=3.142$, about $4\times10^{-4}$ past $\pi$, which looked exactly
like a solver-accuracy failure (the reported state didn't match
$\cos(\pi)$, $-\sin(\pi)$) but was actually the wrong endpoint being
evaluated in the first place, not a numerical error in the integration
itself. The fix takes the largest integer number of full steps that
stay within $t_{end}$, then a single shorter step of exactly the
remaining size to close the gap precisely, so the solver always
reports the value at the $t_{end}$ actually requested. Every solver in
this project is now checked (`verify`) against the value it reports
having reached, confirmed to land exactly on $\pi$ even at a step size
that does not divide it evenly.

### Adaptive step-size control via step-doubling

For a method of order $p$, the global error at step size $H$ scales as
$C H^p$ for a constant $C$ depending on the true solution's higher
derivatives, not on $H$ itself. Advancing by $2h$ two different ways,
one step of size $2h$ or two steps of size $h$, gives two estimates
differing by approximately $C(2h)^p - 2Ch^p = Ch^p(2^p-2)$, which
yields a usable error estimate,
$\text{error} \approx |y_{\text{small}} - y_{\text{big}}|/(2^p-1)$,
without ever needing the true solution. If the estimated error exceeds
the requested tolerance, the step is rejected and retried smaller; if
it comfortably beats tolerance, the next step grows, both via the
standard proportional-control rule $h_{\text{new}} = h \cdot
\text{safety}\cdot(\text{tol}/\text{error})^{1/(p+1)}$. Checked
directly against `scipy.integrate.solve_ivp`'s own adaptive RK45 on
the same problem, and shown to need only 33 steps to reach
$\text{tol}=10^{-8}$ accuracy over $t\in[0,5]$ where a conservative
fixed grid would need thousands.

### Richardson extrapolation

The same two runs used for step-doubling error estimation
(`richardson_extrapolate`) can be recombined algebraically to
eliminate the leading error term entirely: if
$y(h) = y_{\text{true}} + Ch^p + O(h^{p+1})$ and
$y(h/2) = y_{\text{true}} + C(h/2)^p + O(h^{p+1})$, solving the two
equations for $y_{\text{true}}$ gives
$y_{\text{true}} \approx [2^p y(h/2) - y(h)]/(2^p-1)$, accurate to
$O(h^{p+1})$: one order better than either individual run, purely from
combining two runs of the *same* method, with no higher-order method
needed at all. Extrapolating a plain Euler run improves its error by
more than 10x in the test suite; extrapolating RK4 pushes the result
to within $10^{-8}$ of the exact solution from a coarse $h=0.1$.

## Application: two very different ODEs, calibrated to the same market

### A mean-reversion equation for volatility, and where a naive fit falls apart

If squared returns followed a single-timescale mean-reverting process
(the deterministic skeleton of an Ornstein-Uhlenbeck or GARCH(1,1)
variance process), $dV/dt = \kappa(\theta - V)$, the autocorrelation
of squared returns at lag $k$ would decay exactly exponentially,
$\rho_k = \rho_1^k$, so $\ln\rho_k$ would be exactly linear in $k$ with
slope $-\kappa$. `calibrate_mean_reversion` fits that line by
ordinary least squares.

**Fit over the first 30 lags, and the model falls apart immediately**:
$\kappa = -0.00159$/day, negative, meaning the fitted "decay" actually
runs the wrong direction, with $R^2 = 0.003$, essentially no
explanatory power at all. This is not a bug; it is an honest,
important finding. Looking at the raw autocorrelation values directly,
they sit in a noisy band around 0.06-0.14 across lags 1 through 30
with no visible downward trend at all: over a one-month horizon, EUR/
USD's squared-return autocorrelation is close to flat, not decaying.

**Extending the fit to 300 lags recovers a sensible answer, but not a
complete one**: $\kappa = 0.00952$/day, implying a half-life of about
73 days, with $R^2 = 0.64$. A real decay trend is visible once the
window is wide enough to see it (the raw autocorrelations do fall from
roughly 0.10 at short lags to roughly 0.03-0.05 by lag 200, and toward
zero somewhere past lag 400), but $R^2=0.64$ means well over a third of
the variation in $\ln(\text{ACF})$ across lags is *not* explained by a
single exponential. This is the empirical signature of **long-memory
volatility**, a well-documented departure from simple GARCH(1,1)-style
single-timescale mean reversion in real financial time series
(motivating fractionally-integrated models like FIGARCH in the
literature): volatility shocks decay on multiple timescales at once,
not one, so a first-order linear ODE captures the broad trend while
leaving real structure on the table. Solving the fitted equation for a
4x variance shock, all four fixed-step methods and the adaptive solver
agree with the exact solution $V(t) = \theta + (V_0-\theta)e^{-\kappa t}$
to at least 7 significant figures (RK4: error $1.9\times10^{-15}$),
and the fitted model implies such a shock takes about 430 days to
decay to within 5% of the long-run variance, a genuinely interpretable
number, understood as a description of the *dominant* decay timescale
a single-exponential model can see, not the complete dynamics.

### The Kolmogorov forward equations for a continuous-time regime chain

A continuous-time Markov chain's state probabilities evolve by
$dp/dt = pQ$ for a generator matrix $Q$ (rows summing to zero: the
rate of leaving a state must equal the total rate of arriving
elsewhere). `two_state_generator_from_persistence` builds a plausible
$Q$ from daily discrete self-transition probabilities via the small-
interval approximation $P(\Delta t) \approx I + Q\Delta t$ at
$\Delta t=1$ day. Solving the resulting system with RK4 from a
certain starting state and letting it run for 1,000 days recovers
exactly the generator's own analytically-computed stationary
distribution (solved via $\pi Q=0$, $\sum\pi=1$, the direct continuous-
time analogue of the discrete-time stationary-distribution calculation
in `stochastic-processes`), and the total probability across states
stays at exactly 1.00000000 throughout the entire integration, a
genuine conservation-law check, not merely a plausibility check,
verified directly against a numpy null-space eigenvector recomputation
of the same stationary distribution in `verify`.

## Limitations

- `calibrate_mean_reversion`'s single-exponential model is, by its own
  $R^2$, a genuine but incomplete description of EUR/USD's volatility
  persistence; a proper treatment of the long-memory structure the
  fit leaves unexplained would need a fractionally-integrated or
  multi-timescale model, out of scope for a first-order linear ODE.
- `two_state_generator_from_persistence`'s $P(\Delta t) \approx
  I + Q\Delta t$ approximation is exact only as $\Delta t \to 0$; used
  here at $\Delta t = 1$ day (a full trading day, not infinitesimal),
  it is a reasonable but approximate way to obtain a demonstration
  generator, not an exact matrix-logarithm inversion of a specific
  fitted discrete transition matrix.
- The step-doubling adaptive controller re-derives its error estimate
  from two extra function evaluations every accepted step (one at
  $2h$, two at $h$); a proper embedded Runge-Kutta pair (Dormand-Prince,
  for instance) gets an error estimate from the same evaluations
  already needed for the step itself, at lower cost per step than the
  doubling approach used here.

## CLI examples

```bash
# Solve an IVP with a chosen fixed-step method (note the '=' before a
# right-hand side starting with '-', so argparse doesn't treat it as a flag)
python3 cli.py solve --rhs="-y" --y0 1.0 --t-end 2.0 --h 0.01 --method rk4

# Adaptive step-doubling
python3 cli.py adaptive --rhs="-y" --y0 1.0 --t-end 5.0 --tol 1e-8

# Empirically measure each method's convergence order
python3 cli.py convergence --rhs="-y" --exact "math.exp(-t)" --y0 1.0 --t-end 2.0

# Calibrate and simulate the EUR/USD volatility mean-reversion equation
python3 cli.py fxmeanreversion --max-lag 300 --shock-multiple 4 --horizon 200

# Solve the Kolmogorov forward equations for a 2-state regime chain
python3 cli.py fxregimes --p-stay-low 0.99 --p-stay-high 0.99 --horizon 1000

# Cross-check every solver against scipy oracles, conservation laws,
# and empirical convergence-order measurement
python3 cli.py verify
```
