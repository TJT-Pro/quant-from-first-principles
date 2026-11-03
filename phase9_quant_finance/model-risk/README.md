# model-risk

The capstone of a 16-repository personal mathematical library: a
model risk engine built from scratch. Value-at-Risk backtesting via
Kupiec's proportion-of-failures test, model selection via AIC/BIC and
a general likelihood-ratio test (with a from-scratch regularized
incomplete gamma function), a direct demonstration of backtest
overfitting via parameter grid search, and bootstrap propagation of
parameter estimation uncertainty into model outputs. This repository
does not just build more mathematical machinery: it turns that
machinery back on the findings of every sibling repository, asking
directly what happens when the models built throughout this project
are naively applied to real data without accounting for exactly the
phenomena (fat tails, volatility clustering, no genuine predictive
edge) those other repositories spent so much effort establishing.

This repository is independent of the sibling repositories and reuses
none of their code, including its own from-scratch closed-form
Black-Scholes formula and its own from-scratch inverse normal CDF.

## Project structure

```
model-risk/
    README.md
    data/DEXUSEU_returns.csv
    var_backtesting.py       Gaussian/historical VaR, Kupiec's POF test
    model_comparison.py      AIC/BIC, general chi-squared test, cross-validation
    overfitting_demo.py      backtest overfitting via parameter grid search
    parameter_uncertainty.py bootstrap propagation of parameter uncertainty
    fx_application.py        the capstone analysis on real EUR/USD data
    test_*.py
    cli.py
```

## Status

64 tests, 12/12 verify checks, all passing.

## Setup

Python 3.11+. All model-risk code uses only the standard library.
`cli.py verify` uses numpy and scipy as independent oracles.

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest test_var_backtesting.py test_model_comparison.py test_overfitting_demo.py \
    test_parameter_uncertainty.py test_fx_application.py -v
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

### Value-at-Risk backtesting: Kupiec's proportion-of-failures test

Under a correctly specified VaR model at confidence level c,
exceptions (days a realized loss exceeds the VaR estimate) should
occur independently at exactly rate p=1-c. Kupiec's (1995)
likelihood-ratio statistic,
LR = -2*ln[(1-p)^(n-x)*p^x / ((1-x/n)^(n-x)*(x/n)^x)], compares the
likelihood of the observed exception count x under the model's
claimed rate against the likelihood under the empirically observed
rate, and is chi-squared distributed with 1 degree of freedom under
the null. This project uses the exact closed-form relationship
between chi-squared(1) and the normal distribution
(P(chi2_1<=x)=erf(sqrt(x/2)), since chi-squared-1 is the distribution
of a single squared standard normal) rather than a lookup table.
Calibration verified directly: simulating a genuinely correctly
specified model thousands of times rejects it at a rate close to the
nominal 5% (measured between 5% and 6.4% across several independent
runs and sample sizes from 250 to 20,000), not far more often, as a
correctly calibrated test should.

### Model selection: AIC, BIC, and a from-scratch general chi-squared test

AIC=2k-2ln(L) and BIC=k*ln(n)-2ln(L) both penalize model complexity
against fit quality, with BIC's penalty growing with sample size
(making it asymptotically consistent for identifying the true model,
a property AIC does not share). The general likelihood-ratio test for
models differing by more than one parameter needs a chi-squared CDF
for arbitrary degrees of freedom, computed here via the regularized
lower incomplete gamma function P(a,x) (chi-squared with k degrees of
freedom has CDF P(k/2,x/2)), implemented from scratch via the
standard two-regime numerical approach (a series expansion for
x<a+1, a continued fraction otherwise). Verified against the
chi-squared(2) distribution's own exact closed form,
CDF(x)=1-exp(-x/2), to 8 decimal places, and against known critical
values (chi2_1(3.841)=0.95, chi2_5(11.07)=0.95, chi2_10(18.31)=0.95)
all matching to 3 decimal places. AIC and BIC correctly recover the
true model on synthetic data with known ground truth: fitting
polynomials of degree 0 through 6 to data generated from a genuine
degree-2 polynomial plus Gaussian noise, both AIC (541.18) and BIC
(554.37) are minimized exactly at degree 2, with every higher degree
scoring worse on both criteria despite necessarily fitting the
training data at least as well, precisely the behavior these criteria
exist to produce.

### The overfitting demonstration, and its direct link to information-theory's multiple-testing finding

Searching a grid of parameter combinations for the single best
in-sample Sharpe ratio is itself a multiple-comparisons procedure,
structurally identical to the mechanism the sibling information-
theory repository identified when scanning multiple lags for
"significant" mutual information and finding one (lag=10 on real
sign data) purely by chance, not surviving Bonferroni correction. The
same phenomenon governs backtest overfitting: with enough parameter
combinations tried, some will look good on any fixed training period
purely from noise, whether or not the underlying series has any
genuine exploitable structure. Demonstrated first on pure noise with
known ground truth: 6,000 simulated i.i.d. Gaussian returns (by
construction containing no exploitable structure whatsoever) still
yield, from a 47-combination grid search, an in-sample "winner" whose
in-sample Sharpe comfortably exceeds its own out-of-sample Sharpe,
confirming the mechanism operates exactly as expected even when the
true answer (no genuine edge exists) is known in advance.

### Bootstrap propagation of parameter uncertainty into model outputs

Even when a model's functional form is exactly correct, its
parameters are estimated from a finite historical sample and carry
real uncertainty that should propagate into any output computed from
them. The nonparametric bootstrap (Efron 1979) estimates an
estimator's sampling distribution directly: resample the original
data with replacement many times, recompute the estimator on each
resample, and use the resulting spread as a genuine, assumption-free
measure of estimation uncertainty. Applied here to volatility
estimation feeding a Black-Scholes option price: a smaller historical
sample gives a genuinely wider bootstrap distribution for both the
volatility estimate and, propagated through, the resulting option
price, verified directly (not merely asserted) by comparing bootstrap
spreads from a 200-observation sample against a 5,000-observation one
on identical underlying data-generating parameters.

A real discounting bug, caught by its own test. An earlier version of
the from-scratch Black-Scholes formula's zero-volatility special case
returned the plain (undiscounted) intrinsic value max(S_0-K,0)
instead of the correct discounted-forward limit
max(S_0-K*e^{-rT},0): as sigma->0, the terminal price becomes
deterministic and converges to the risk-free forward S_0*e^{rT}, so
the discounted payoff converges to S_0-K*e^{-rT}, not S_0-K. Caught
directly by a test checking this exact limit (S_0=110, K=100, r=0.05,
T=1: the buggy version returned 10.0 against the correct 14.877, a
difference of nearly 4.9, far too large to be rounding) and fixed to
use the correct discounted formula, which also correctly reduces to
the plain intrinsic value at T=0 (where the discount factor is
exactly 1) without any special-casing needed.

## Application: what happens when naive models meet real EUR/USD data

The flagship finding: Gaussian VaR fails exactly where fat tails
matter most, and passes exactly where they do not. At 95% confidence
(well within the bulk of the return distribution, where a Gaussian
approximation is reasonably adequate), both the Gaussian model (350
exceptions against 346.5 expected, p=0.847) and the historical model
(346 against 346.5, p=0.978) pass Kupiec's test comfortably. At 99%
confidence (deep in the tail, exactly where the fat tails established
throughout this entire project family via kurtosis, KL divergence
from a fitted Gaussian, and direct extreme-value analysis actually
bite), the Gaussian model fails decisively: 122 actual exceptions
against only 69.3 expected, p-value effectively 0.000000, a model
that should have been exceeded roughly 1% of the time was actually
exceeded nearly 1.8% of the time. The historical model, making no
distributional assumption at all, continues to pass cleanly (69
exceptions against 69.3 expected, p=0.971). This is not a subtle
statistical nuance: a risk manager relying on the Gaussian model's 99%
VaR figure would have been caught unprepared by a large loss roughly
76% more often than the model itself claimed to be delivering.

Backtest overfitting, demonstrated on the actual EUR/USD series used
throughout this project. Searching 47 moving-average-crossover window
combinations on a 60% training split, the best in-sample combo
((8,40)) shows an eye-catching Sharpe ratio of +0.524; that exact same
combination's performance on the held-out 40% test period collapses
to -0.037, and the average out-of-sample Sharpe across all 47
combinations tested is -0.184. The apparently impressive in-sample
result was, in expectation, exactly the kind of noise-driven
"discovery" the grid search was most likely to surface, not evidence
of genuine exploitable structure, precisely consistent with the "no
linear or nonlinear predictability" finding established via entirely
different methods (autocorrelation, mutual information, direct
Sharpe-ratio backtesting) throughout this entire project family.

Even a perfectly specified model has a real, quantified uncertainty
budget. Pricing a real, calibrated 3-month EUR/USD at-the-money call,
the volatility estimate itself carries a 90% bootstrap confidence
interval of [8.93%, 9.31%] around its point estimate of 9.13%, purely
from the finite (6,930-observation) historical sample used to
estimate it. Propagated through Black-Scholes, this gives the option
price itself a genuine 90% confidence interval of [0.02449, 0.02537]
around its point estimate of 0.02494, a relative uncertainty of about
1.08% coming entirely from not knowing the true volatility with
certainty, before any consideration of whether Black-Scholes' own
assumptions (constant volatility, no jumps, continuous trading) are
themselves correct.

## Limitations

- Kupiec's POF test checks only the unconditional exception rate, not
  whether exceptions cluster in time; a model could pass this test
  while still failing to account for volatility clustering
  (established throughout this project family) if its exceptions
  bunch up during high-volatility periods rather than occurring
  independently, a distinct failure mode a conditional coverage test
  (Christoffersen's test) would catch and this project does not
  implement.
- overfitting_demo.py's grid search uses a single fixed 60/40
  train/test split; a genuinely more rigorous demonstration would use
  walk-forward (rolling-origin) validation across multiple splits to
  confirm the overfitting pattern is not itself an artifact of one
  particular split's own sampling luck.
- parameter_uncertainty.py's bootstrap treats each daily return as an
  independent, exchangeable observation; given the volatility
  clustering established throughout this project family, a block
  bootstrap (resampling contiguous chunks rather than individual days)
  would more faithfully preserve the data's own temporal dependence
  structure and could give a genuinely different (likely wider)
  uncertainty estimate than the simple i.i.d. bootstrap used here.

## CLI examples

```bash
# Backtest Gaussian vs historical VaR on real EUR/USD data
python3 cli.py varbacktest --confidence 0.99

# Demonstrate backtest overfitting via parameter grid search
python3 cli.py overfit

# Propagate volatility estimation uncertainty to a real option price
python3 cli.py priceuncertainty --r 0.03 --t 0.25

# AIC/BIC model selection sanity check on synthetic data with known ground truth
python3 cli.py modelselect

# Cross-check every formula against scipy/numpy oracles and known theory
python3 cli.py verify
```

## Closing note

This is the sixteenth and final repository in this project. Sixteen
independent mathematical domains, each built from scratch in Python's
standard library, each validated against independent oracles (numpy,
scipy, direct grid-integration, hand-derived closed forms, brute-force
enumeration) and against real EUR/USD market data, finding real,
documented bugs along the way, each caught by the same discipline:
never trust a result until it has been checked against something
else. This final repository's entire purpose is to turn that same
discipline on the rest of the project, and the result is consistent
with what came before it: the mathematics works exactly as derived,
the real data behaves exactly as the fifteen sibling repositories
independently found it to behave, and naive models that ignore those
findings fail in specific, predictable, and now precisely quantified
ways.
