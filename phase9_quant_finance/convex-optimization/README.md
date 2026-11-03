# convex-optimization

A convex optimization laboratory built from scratch: convexity
checking (two independent ways), gradient descent with backtracking
line search, Newton's method, projected gradient descent (with an
exact simplex projection algorithm), KKT/Lagrangian machinery for
equality- and inequality-constrained problems, and Lagrangian duality,
applied to constrained portfolio optimization on real EUR/USD-derived
trading strategies.

This repository is independent of the sibling repositories and reuses
none of their code.

## Project structure

```
convex-optimization/
    README.md
    data/DEXUSEU_returns.csv
    linear_algebra.py            matrix ops, Gaussian elimination, determinants
    convex_functions.py          gradients, Hessians, two convexity checks
    gradient_methods.py          gradient descent with Armijo line search
    newtons_method.py            Newton's method, quadratic convergence
    constrained_optimization.py  simplex/box projection, projected GD, KKT solver/verifier
    duality.py                   Lagrangian dual, weak/strong duality
    fx_application.py            constrained portfolio optimization on real EUR/USD strategies
    test_*.py
    cli.py
```

## Status

92 tests, 10/10 verify checks, all passing.

## Setup

Python 3.11+. All optimization code uses only the standard library.
`cli.py verify` uses numpy and scipy as independent oracles.

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest test_linear_algebra.py test_convex_functions.py test_gradient_methods.py \
    test_newtons_method.py test_constrained_optimization.py test_duality.py test_fx_application.py -v
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

### Convexity: two checks, one exact and one that can only falsify

A function is convex on a convex domain if
f(lambda*x+(1-lambda)*y) <= lambda*f(x)+(1-lambda)*f(y) for every x,y
in the domain and lambda in [0,1]. `check_convexity_by_definition`
tests this inequality directly on many random point pairs, but this is
honestly reported as "convex_on_samples", not "is_convex": it can only
ever falsify convexity (by finding one violation), never prove it,
since the defining property ranges over an infinite set no finite
sample exhausts. `convexity_via_hessian` gives an exact, second-order
alternative wherever the Hessian is available: a twice-differentiable
function is convex on a convex domain if and only if its Hessian is
positive semi-definite everywhere on it, checked here via Sylvester's
criterion (all leading principal minors non-negative) rather than a
full eigendecomposition, since the small Hessians this project works
with need nothing more.

### Gradient descent and the cost of ill-conditioning

Steepest descent moves along -grad f(x), the direction of steepest
local decrease (by Cauchy-Schwarz, the directional derivative
grad f(x).d is minimised exactly when d is anti-parallel to the
gradient). The Armijo backtracking line search picks the step size by
shrinking from a generous initial guess until
f(x+alpha*d) <= f(x) + c*alpha*grad f(x).d holds, removing the need to
hand-tune a fixed step for each problem's curvature. A directly
measured cost of poor conditioning: on an isotropic quadratic,
gradient descent with an exact line search reaches the minimiser in
exactly one step; on the same quadratic scaled to a condition number
of 100 (100*x1^2 + x2^2), it needs 869 iterations, zigzagging its way
to the same answer.

### Newton's method: exact for quadratics, quadratic convergence otherwise

Minimising the local second-order Taylor approximation
f(x+d) ~ f(x) + grad f(x).d + 0.5*d^T*Hess f(x)*d exactly gives the
Newton step d = -Hess f(x)^{-1}*grad f(x). Two properties verified
directly, not merely cited: (1) for a genuinely quadratic f, that
second-order approximation *is* f itself exactly, so Newton's method
with exact derivatives reaches the minimiser in exactly one step
regardless of starting point or conditioning, the same
condition-number-100 problem that took gradient descent 869 iterations
is solved in 1. (With the default finite-difference gradient/Hessian
instead of exact analytic ones, it takes 2 steps: the first lands
extremely close to the true minimiser, and the small residual from the
finite-difference approximation itself needs one corrective step to
clear the tolerance, a real, documented distinction between the two
code paths, not a bug in either.) (2) Near the optimum of a smooth,
strongly convex function, Newton's method converges quadratically: on
f(x) = e^x - x (minimum at x=0, where f''(0)=1, f'''(0)=1, giving a
theoretical convergence constant of |f'''(0)|/(2*f''(0)) = 0.5), the
measured ratio e_{k+1}/e_k^2 converges to exactly 0.5 as k grows:
0.284, 0.354, 0.432, 0.485, 0.499, 0.4999..., matching theory purely
from the observed sequence of errors.

### Exact simplex projection

The Euclidean projection of v onto the probability simplex
{x : x_i>=0, sum(x)=1} has a closed-form, finite algorithm (Held,
Wolfe & Crowder 1974): the KKT stationarity condition for this
projection gives x_i = max(v_i-theta, 0) for a single shared theta,
and sorting v in descending order identifies exactly which theta makes
the result sum to 1, with no iteration needed. Verified directly
against a brute-force grid search over the simplex (no candidate grid
point beats the computed projection's distance to v) and against
scipy.optimize.minimize solving the identical projection problem.

### KKT conditions and the Lagrangian system for equality-constrained QPs

For min 0.5*x^T*Q*x + c^T*x subject to A*x=b, the Lagrangian
stationarity conditions Q*x+c+A^T*lambda=0, A*x-b=0 form a single
linear system in [x;lambda]:

    [[Q, A^T], [A, 0]] * [x; lambda] = [-c; b]

solved here directly (no iteration at all) since this system's
solution *is* the global constrained minimiser for a convex quadratic
Q. The general KKT verifier checks all four conditions (stationarity,
primal feasibility, dual feasibility mu>=0, and complementary
slackness mu_j*g_j(x)=0) as four separately-named booleans rather than
one pass/fail, since a failure in one points at a genuinely different
problem with a candidate solution than a failure in another.

### Lagrangian duality, verified on a fully worked analytical example

For min x^2 s.t. x>=1 (g(x)=1-x<=0), the Lagrangian
L(x,mu) = x^2 + mu*(1-x) is minimised over x at x=mu/2, giving the
dual function g(mu) = mu - mu^2/4, maximised at mu*=2 with g(mu*)=1,
exactly matching the primal optimum f(1)=1 (strong duality, since this
problem is convex and satisfies Slater's condition). This project's
dual-ascent procedure, run from scratch with no knowledge of the
closed-form answer, recovers mu*=2, x*=1, and a duality gap of
3.2e-12: this project's own Newton's method evaluates the inner
minimisation over x for each fixed mu (the dual function's gradient
with respect to mu is, by the envelope theorem, simply the constraint
value at that inner minimiser, so no separate gradient derivation is
needed), and projected gradient ascent (projecting mu onto mu>=0)
finds the best mu.

## Application: constrained portfolio optimization on real, derived trading strategies

**Building genuine covariance structure from a single real series.**
Four daily P&L series, all derived from the same real EUR/USD returns:
buy_hold (the raw return itself), momentum (long or short based on the
trailing 5-day return's sign), mean_reversion (the opposite signal,
but computed over a different, 20-day window, not the literal negative
of momentum, since using the same window for both would make them
exact opposites by construction, correlation exactly -1, a degenerate,
singular covariance matrix caught and fixed during development before
it reached the final design), and vol_scaled (the same return rescaled
by inverse trailing volatility). All four Sharpe ratios are
unremarkable individually (buy_hold +0.012, momentum -0.075,
mean_reversion -0.256, vol_scaled +0.023, none a real trading edge),
which is itself consistent with every sibling repository's "no linear
predictability" finding; what makes this a genuine portfolio problem
is their correlation structure (0.02 to 0.93 between different pairs),
not any individual strategy's quality.

**The long-only constraint genuinely binds, and the cost is
quantified.** The unconstrained minimum-variance solution
(buy_hold +0.77, momentum +0.28, mean_reversion +0.27, vol_scaled
-0.33) wants to short the vol_scaled strategy, since it is highly
correlated with buy_hold (0.93) and carries much higher individual
variance, making it an effective volatility hedge when held short.
Forcing all weights non-negative (via projected gradient descent onto
the simplex) zeros that position entirely (buy_hold 0.24, momentum
0.38, mean_reversion 0.38, vol_scaled exactly 0) and raises the
portfolio's variance from 6.50e-6 to 8.82e-6: the long-only constraint
costs exactly 35.6% more variance on this data. KKT conditions are
verified to hold exactly at both solutions (all four conditions
satisfied, stationarity residual at machine precision for the
unconstrained case).

**Two real numerical pitfalls, both found by testing and both about
the same underlying issue: step-size scale.** First,
`constrained_optimization.projected_gradient_descent` with a naively
"reasonable" step of 0.5 failed to converge within 100,000 iterations
on this real covariance matrix, whose entries are tiny (real
daily-return data, O(1e-5) to O(1e-4)), making the gradient itself
tiny and any fixed step calibrated for unit-scale problems glacially
slow; a step of 10,000, calibrated to the data's actual scale,
converges in under 100 iterations to the identical answer.
`fx_application.py`'s `min_variance_portfolio_long_only` now defaults
to the correctly-scaled step for exactly this reason. Second, and
independently, `scipy.optimize.minimize`'s SLSQP solver, called on the
identical real covariance matrix with its own default tolerances,
falsely reported convergence at its unmoved starting point (all four
weights at 0.25, zero progress at all), the same scale-sensitivity,
this time in someone else's well-tested library, not a homemade one.
Tightening SLSQP's `ftol` to 1e-16 makes it agree with this project's
answer to eight significant figures. Both findings say the same thing:
at real financial-data scale, "reasonable-looking" numerical defaults
calibrated for unit-scale toy problems are not reasonable at all, and
this is true regardless of whose optimizer is being used.

**A self-caught interpretive error, worth stating precisely.** An
early version of this project's own exploratory analysis briefly
concluded the long-only portfolio had lower variance than the
unconstrained one, an impossible result (constraints can only weaken
or match an optimum, never improve it). The cause: comparing the
optimizer's internal objective value, 0.5*w^T*Sigma*w, directly
against the true variance w^T*Sigma*w, without the factor of 2.
`min_variance_portfolio_long_only` now returns the true variance
directly in its result, specifically so this exact confusion cannot
recur for any future caller of this module.

## Limitations

- `projected_gradient_descent` uses a fixed step size rather than a
  line search; the step-size-scale finding above is a direct
  consequence of this choice, and a backtracking line search adapted
  for the projected setting (as used for `gradient_descent`) would
  remove the need to hand-pick a scale-appropriate step at all.
- `solve_equality_constrained_qp` solves the full stacked KKT system
  by dense Gaussian elimination; this is exact and fine for the small
  problems (a handful of assets) this project ever applies it to, but
  would not scale to a realistic institutional portfolio with hundreds
  of assets, where an interior-point or active-set QP method
  exploiting the system's block structure would be used instead.
- `dual_function`'s inner minimisation uses this project's own
  Newton's method with default (finite-difference) derivatives unless
  analytic ones are supplied; for a Lagrangian whose Hessian is
  ill-conditioned or difficult to approximate well by finite
  differences, this inner solve could itself need the same kind of
  careful scaling attention documented above for the portfolio
  application, not automatically handled by `dual_function` itself.

## CLI examples

```bash
# Minimize a function via gradient descent or Newton's method
python3 cli.py minimize --f "(x[0]-3)**2 + (x[1]+2)**2" --x0 "0,0" --method gradient
python3 cli.py minimize --f "(x[0]-3)**2 + (x[1]+2)**2" --x0 "0,0" --method newton

# Constrained portfolio optimization on real EUR/USD-derived strategies
python3 cli.py portfolio

# Cross-check every method against scipy/numpy oracles and known analytical results
python3 cli.py verify
```
