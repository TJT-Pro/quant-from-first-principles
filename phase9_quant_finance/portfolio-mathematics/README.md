# portfolio-mathematics

A classical portfolio theory engine built from scratch: the Markowitz
mean-variance efficient frontier in exact closed form, the tangency
(maximum-Sharpe) portfolio and the Capital Market Line, CAPM beta and
alpha estimation with an exact systematic/idiosyncratic risk
decomposition, marginal and component contributions to portfolio risk
(with Euler's theorem verified directly), and the Kelly criterion for
growth-optimal position sizing, including its genuine mathematical
connection to the tangency portfolio.

This repository is independent of the sibling repositories and reuses
none of their code. It is a different lens on portfolio construction
than the convex-optimization repository: that repository built the
general-purpose optimization machinery (gradient descent, Newton's
method, KKT conditions, duality) and applied it to constrained
minimum-variance portfolios; this repository focuses on the closed-
form financial theory itself (the full efficient frontier, CAPM, risk
attribution, growth-optimal betting) and shares no code with it.

## Project structure

```
portfolio-mathematics/
    README.md
    data/DEXUSEU_returns.csv
    linear_algebra.py       matrix inversion, linear system solver
    mean_variance.py        Markowitz efficient frontier, closed form
    tangency_portfolio.py   maximum-Sharpe portfolio, Capital Market Line
    capm.py                 beta/alpha estimation, risk decomposition
    risk_decomposition.py   marginal/component contributions to risk
    kelly_criterion.py      growth-optimal position sizing
    fx_application.py       full analysis of real EUR/USD-derived strategies
    test_*.py
    cli.py
```

## Status

76 tests, 10/10 verify checks, all passing.

## Setup

Python 3.11+. All portfolio-theory code uses only the standard
library. `cli.py verify` uses numpy as an independent oracle.

```bash
pip install --break-system-packages numpy
```

## Running tests

```bash
python3 -m pytest test_linear_algebra.py test_mean_variance.py test_tangency_portfolio.py \
    test_capm.py test_risk_decomposition.py test_kelly_criterion.py test_fx_application.py -v
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

### The Markowitz efficient frontier, in exact closed form

Minimizing 0.5*w^T*Sigma*w subject to w^T*mu=target and w^T*1=1 via
Lagrange multipliers gives Sigma*w = lambda_1*mu + lambda_2*1, so
w = Sigma^{-1}*(lambda_1*mu+lambda_2*1); substituting back into both
constraints gives a 2x2 linear system in (lambda_1, lambda_2) whose
coefficients are the classical scalars A=1^T*Sigma^{-1}*1,
B=1^T*Sigma^{-1}*mu, C=mu^T*Sigma^{-1}*mu. Solving that system directly
gives every point on the frontier in closed form, with achieved
variance simplifying to sigma_p^2(target) = (A*target^2-2*B*target+C)/D
(D=A*C-B^2): a parabola in (variance, return) space, minimized exactly
at target*=B/A, giving the global minimum-variance portfolio's
weights w_gmv=Sigma^{-1}*1/A. Verified three independent ways: the
achieved variance from computed weights matches the parametric
formula to floating-point precision; the GMV portfolio found by
minimizing the parabola over target matches the GMV portfolio found by
dropping the return constraint entirely and solving directly; and
every frontier point checked against an independent numpy computation
of the same closed form agrees to 1e-8.

### The tangency portfolio and the Capital Market Line

Maximizing (w^T*mu-rf)/sqrt(w^T*Sigma*w) subject only to w^T*1=1 (no
target-return constraint, since the tangency portfolio is not "best
at some particular return," it is the single best risk-return
trade-off among every combination of risky assets) gives, via the
Lagrangian's stationarity condition, w_tan proportional to
Sigma^{-1}*(mu-rf*1), normalized to sum to 1. Verified as a genuine
optimization claim, not merely asserted: across 50,000 randomly
generated portfolios (short sales allowed, only the budget constraint
imposed), none ever beat the closed-form tangency portfolio's Sharpe
ratio. The Capital Market Line, R_p=rf+Sharpe_tan*sigma_p, follows
directly from eliminating the mixing weight alpha between
R_p=rf+alpha*(R_tan-rf) and sigma_p=alpha*sigma_tan (the risk-free
asset contributes zero variance and zero covariance, which is exactly
why mixing it in produces a straight line rather than a curve): every
two-fund combination tested has, to eight decimal places, exactly the
tangency portfolio's own Sharpe ratio, the Two-Fund Separation Theorem
in its most direct, checkable form.

### CAPM: an exact variance decomposition, not an approximate one

Beta (Cov(R_i,R_m)/Var(R_m)) and alpha (Jensen's alpha,
mean(R_i)-beta*mean(R_m)) come from the ordinary least squares fit
R_i=alpha+beta*R_m+epsilon. The variance decomposition
Var(R_i)=beta^2*Var(R_m)+Var(epsilon) holds exactly, not
approximately, as a direct consequence of OLS's own defining
orthogonality condition Cov(R_m,epsilon)=0: expanding
Var(alpha+beta*R_m+epsilon) gives a cross term
2*beta*Cov(R_m,epsilon) that vanishes identically because OLS was
specifically constructed to make it zero. Verified on synthetic data
to agree to 12 decimal places, and the orthogonality itself
(Cov(market,residuals)) checked directly to be indistinguishable from
zero at machine precision (on the order of 1e-20 in a representative
run).

### Marginal and component risk: an exact decomposition via Euler's theorem

MCTR_i=(Sigma*w)_i/sigma_p is the derivative d(sigma_p)/d(w_i),
verified directly against a finite-difference approximation of the
same derivative to 4 decimal places. Because portfolio volatility
sigma_p(w)=sqrt(w^T*Sigma*w) is homogeneous of degree 1 in w
(scaling every weight by t scales sigma_p by exactly t), Euler's
theorem for degree-1-homogeneous functions guarantees
sum_i w_i*MCTR_i = sigma_p exactly, with no leftover or double-counted
risk: a genuinely additive decomposition, checked directly (not
merely cited) to hold across several different weight vectors, and
under an explicit scaling test (doubling every weight exactly doubles
both total volatility and every individual component). This
decomposition can reveal risk concentration invisible to capital
weights alone: a synthetic three-asset example with capital weights
[0.9, 0.05, 0.05] but asset 3 carrying 100x the variance of the other
two shows asset 3 contributing 23.5% of total portfolio risk despite
only 5% of capital, exactly the gap between capital allocation and
risk allocation this decomposition exists to expose.

### The Kelly criterion, and its exact connection to the tangency portfolio

Maximizing expected log wealth growth E[log(1+f*R)] via a second-order
Taylor approximation, f*mu-0.5*f^2*sigma^2, gives the single-asset
Kelly fraction f*=mu/sigma^2; the multi-asset generalization, maximizing
w^T*mu-0.5*w^T*Sigma*w with no budget constraint (Kelly weights are
leverage fractions, not a normalized allocation), gives
w*=Sigma^{-1}*mu directly, by the same stationarity argument used for
mean-variance theory throughout this project. Verified two ways: (1)
full-precision Monte Carlo compounding (no small-return approximation
at all) confirms realized growth genuinely peaks at f=1x the analytic
Kelly fraction, with both 0.5x and 1.5x Kelly giving measurably lower
realized growth in every test run; over-leveraging to 2x Kelly on a
representative single-asset example drives the realized growth rate
negative. (2) At rf=0, Kelly weights (normalized to sum to 1) match
the tangency portfolio's weights exactly: growth-optimal betting
theory and Sharpe-ratio-maximizing portfolio theory, developed from
entirely different first principles, arrive at the same optimal
risky-asset direction, differing only in normalization. This
equivalence is confirmed not only on a clean synthetic three-asset
example but on real EUR/USD-derived strategy data as well.

## Application: portfolio theory applied to real EUR/USD-derived trading strategies

Four strategies rebuilt fresh from the same real EUR/USD series used
throughout this project family (buy_hold, 5-day momentum, 20-day
mean-reversion, and volatility-scaled buy_hold; momentum and
mean-reversion deliberately use different lookback windows, since
using the same one would make them exact opposites by construction, a
real design trap caught early elsewhere in this project family and
confirmed here not to recur: their realized correlation is -0.32, not
-1.0).

**CAPM cleanly separates the "same information, different transform"
strategy from the two genuinely different-information strategies.**
vol_scaled, a rescaled version of the same underlying return series,
has beta=1.81 and R-squared=0.87 against buy_hold: it is
overwhelmingly explained by the market factor, as it should be, since
it uses no information buy_hold does not also use. Momentum
(beta=0.016, R-squared=0.0003) and mean-reversion (beta=0.055,
R-squared=0.003) show essentially none of their variance explained by
the market factor at all, consistent with these signal-based
strategies genuinely trading on different (if not necessarily
profitable) information than simple direction.

**The tangency portfolio's Sharpe ratio is honestly negative, and this
is the correct answer, not a flaw.** With rf=2%, the theoretically-
optimal combination of all four strategies (even with full long-short
flexibility) has Sharpe=-0.74: none of the four strategies individually
has an average annualized return anywhere close to a realistic
risk-free rate (buy_hold: 0.11%; momentum: -0.68%; mean-reversion:
-2.33%; vol_scaled: 0.41%), so no combination of them can either. This
is the portfolio-construction machinery correctly reporting that no
genuine edge exists among these specific inputs, exactly consistent
with the "no linear or nonlinear predictability" finding running
throughout this entire project family via entirely different methods
(autocorrelation tests, mutual information, direct backtesting): the
mathematics does not manufacture an edge where none exists, and
reporting an honest negative Sharpe ratio here is the theory working
correctly, not failing.

**Euler's risk decomposition still holds exactly on real data, and
reveals a genuine hedge.** The tangency portfolio's risk contributions
sum to exactly its own volatility (confirmed to the same precision as
the synthetic examples), with mean-reversion contributing 53.1% of
total risk, buy_hold 33.5%, momentum 26.2%, and vol_scaled
contributing -12.8%: a negative risk contribution, permitted (and
correctly reported) by the Euler decomposition when an asset's
current position is actually reducing total portfolio risk rather than
adding to it, a genuine hedging effect the decomposition surfaces
directly rather than obscuring inside a single aggregate number.

## Limitations

- `mean_variance.py` and `tangency_portfolio.py` allow unrestricted
  short selling (no non-negativity constraint on weights); a long-only
  efficient frontier would need the iterative, constrained approach
  built in the sibling convex-optimization repository instead, since
  the clean closed-form solutions here rely specifically on there
  being no inequality constraints to handle.
- `capm.py`'s single-factor model attributes all non-market-driven
  variance to a single "idiosyncratic" residual term; a genuine
  multi-factor model (Fama-French style) would decompose that residual
  further into additional systematic factors, not implemented here.
- `kelly_criterion.py`'s formulas rely on a second-order (small-return)
  Taylor approximation to expected log growth; simulate_compound_growth
  cross-checks this against full-precision compounding for the
  specific parameter ranges tested here, but the approximation would
  degrade for asset return distributions with much larger typical
  magnitudes or heavier tails than the ones used in this project's
  tests and application.

## CLI examples

```bash
# Construct the efficient frontier for a given (mu, Sigma)
python3 cli.py frontier --mu "0.05,0.10,0.15" --sigma "0.04,0.01,0.0,0.01,0.09,0.02,0.0,0.02,0.16"

# Compute the tangency portfolio and Capital Market Line
python3 cli.py tangency --rf 0.02

# Kelly criterion and fractional Kelly comparison
python3 cli.py kelly --mu 0.0005 --sigma 0.01

# Full portfolio analysis of real EUR/USD-derived strategies
python3 cli.py fxanalysis --rf 0.02

# Cross-check every formula against numpy oracles and known theory
python3 cli.py verify
```
