# time-series-mathematics

A classical (Box-Jenkins) time series engine built from scratch:
autocorrelation and partial autocorrelation via the Durbin-Levinson
recursion, AR/MA/ARMA simulation and estimation (Yule-Walker, OLS, and
conditional-sum-of-squares maximum likelihood via a from-scratch
Nelder-Mead optimizer), the Augmented Dickey-Fuller stationarity test
with critical values obtained honestly by Monte Carlo simulation, and
AIC/BIC order selection, applied throughout to real EUR/USD data.

This repository is independent of `math-implementations`,
`stochastic-processes`, `black-scholes`, and `automatic-differentiation`,
and reuses none of their code.

## Project structure

```
time-series-mathematics/
    README.md
    data/DEXUSEU_returns.csv
    autocorrelation.py     ACF, PACF (Durbin-Levinson), Ljung-Box test
    optimization.py        Nelder-Mead simplex optimizer
    ar_models.py           AR(p): simulation, Yule-Walker/OLS, stationarity
    ma_models.py           MA(q): simulation, invertibility, CSS fitting
    arma_models.py         ARMA(p,q): simulation, CSS fitting, likelihood
    stationarity.py        Augmented Dickey-Fuller test, Monte Carlo critical values
    model_selection.py     AIC/BIC order selection
    test_*.py
    cli.py
```

## Status

105 tests, 14/14 verify checks, all passing.

## Setup

Python 3.11+. All estimation and simulation code uses only the
standard library. `cli.py verify` uses numpy and scipy as independent
oracles.

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest test_autocorrelation.py test_optimization.py test_ar_models.py \
    test_ma_models.py test_arma_models.py test_stationarity.py test_model_selection.py -v
```

Note: the stationarity and model-selection test suites run Monte Carlo
simulations inside several tests and take under a minute combined;
none are slow by accident.

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

### Autocorrelation, partial autocorrelation, and the Durbin-Levinson recursion

The sample autocorrelation at lag $k$, $\hat\rho_k = \hat\gamma_k/\hat\gamma_0$,
measures the raw linear association between $X_t$ and $X_{t-k}$. The
**partial** autocorrelation at lag $k$, $\phi_{kk}$, measures that same
association *after* removing the linear effect of everything in
between: it is the coefficient on $X_{t-k}$ in the best linear
predictor of $X_t$ from $X_{t-1}, \dots, X_{t-k}$. Solving a fresh
order-$k$ Yule-Walker system for every $k$ would cost $O(k^3)$ per
lag; the **Durbin-Levinson recursion** instead updates the order-$(k-1)$
solution to order $k$ in $O(k)$ work:

$$\phi_{kk} = \frac{\rho_k - \sum_{j=1}^{k-1}\phi_{k-1,j}\rho_{k-j}}{1 - \sum_{j=1}^{k-1}\phi_{k-1,j}\rho_j}, \qquad
\phi_{kj} = \phi_{k-1,j} - \phi_{kk}\phi_{k-1,k-j}.$$

The final row of this recursion at $k=p$ **is** the Yule-Walker
AR($p$) coefficient vector, which is why `yule_walker_coefficients`
runs the identical recursion rather than a separate algorithm; the
test suite checks this identity directly (the PACF at lag $k$ always
equals the last coefficient of an AR($k$) Yule-Walker fit).

**Two classical model-identification signatures** fall directly out
of this machinery and are verified on simulated data: an AR(1)
process's ACF decays geometrically ($\rho_k = \phi^k$) while its PACF
cuts off sharply after lag 1; an MA(1) process's ACF cuts off after
lag 1 ($\rho_1 = \theta/(1+\theta^2)$, $\rho_k=0$ for $k\ge2$) while
its PACF decays instead. Under the white-noise null, $\hat\rho_k$ is
asymptotically $N(0, 1/n)$ (Bartlett's formula for the special case of
testing against white noise), giving the $\pm z\cdot n^{-1/2}$
significance band `bartlett_confidence_band` computes, using a
from-scratch rational approximation to the inverse normal CDF
(Acklam's algorithm) so no statistics library is needed for it.

### AR(p): Yule-Walker, OLS, and checking stationarity properly

Two estimators are implemented and cross-checked against each other:
**Yule-Walker** (method of moments, via Durbin-Levinson) and
**conditional least squares / OLS** (regressing $X_t$ directly on its
own lags). The two coincide asymptotically and are close at finite
sample sizes; both are implemented rather than trusting either alone.

**Checking stationarity properly, not just for AR(1).** An AR($p$)
process is stationary iff every root of
$1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0$ lies strictly
outside the unit circle. For $p=1$ this is just $|\phi_1|<1$, but no
simple closed form exists for general $p$, so `is_stationary` finds
the roots numerically via the **Durand-Kerner method**: starting from
$p$ points spread around a circle in the complex plane, each is
refined simultaneously by
$z_i \leftarrow z_i - p(z_i) / \prod_{j\ne i}(z_i - z_j)$, using the
*current* guesses for the other roots as a stand-in for the
polynomial's true (unknown) factorisation. This is checked directly
against `numpy.roots` on shared test polynomials.

### MA(q) and ARMA(p,q): conditional sum of squares via Nelder-Mead

Unlike AR estimation, no closed-form solution exists for MA
parameters: the residuals $\varepsilon_t = X_t - \mu - \sum_j \theta_j
\varepsilon_{t-j}$ depend on the parameters recursively and
nonlinearly, since each residual depends on previous residuals which
themselves depend on the same $\theta_j$. `fit_css` minimises the
conditional sum of squared residuals (conditioning on zero pre-sample
innovations, the standard simplification short of the full
innovations algorithm) using a from-scratch **Nelder-Mead simplex
optimizer**, appropriate here because the parameter count is small (a
handful of AR/MA coefficients) and a numerical gradient would cost
$2n$ extra evaluations per step for no real benefit at this scale.
ARMA(p,q) estimation (`arma_models.fit_css`) is the direct combination
of both.

**A genuine bug, found by testing at realistic scale, fixed and
explained rather than patched over.** The Nelder-Mead implementation
initially used a single stopping rule: declare convergence once the
spread of function values across the simplex fell below a tolerance.
Tested against a plain 1D quadratic minimised at $x=5$, it reported
"converged" after only 10 iterations at $x=4.9$. The root cause: with
only two points in a 1-dimensional simplex, two points symmetric about
the true minimum of an even function ($x=4.9$ and $x=5.1$, say) can
have function values that agree to within floating-point noise while
being nontrivially far from the minimum in $x$, since a quadratic's
value is identical at $+d$ and $-d$ from its vertex. The fix requires
**two** independent convergence criteria, not one: the function-value
spread *and* the simplex's spread in parameter space, both below their
own tolerances. This is not a workaround specific to this project;
it is the same two-part stopping rule production implementations
(including SciPy's) actually use, and a regression test locks in the
exact scenario that broke the single-criterion version.

**A second bug, found while fitting ARMA(1,1) to real EUR/USD data.**
Certain Nelder-Mead starting points sent the search transiently
through AR coefficients corresponding to explosive (non-stationary)
dynamics, and the recursive residual computation overflowed to an
uncaught `OverflowError`, crashing the fit outright. The fix follows
a standard optimization pattern: catch the overflow and score that
candidate point as positive infinity rather than letting the exception
propagate. Nelder-Mead's logic (sort by score, move away from the
worst point) handles an infinite score correctly with no special
casing needed anywhere else, and a candidate that blows up is simply
treated as arbitrarily bad, exactly as it should be.

### Testing for a unit root: the Augmented Dickey-Fuller test

The ADF test regression is

$$\Delta X_t = \alpha + \gamma X_{t-1} + \sum_{i=1}^{k}\delta_i \Delta X_{t-i} + e_t,$$

fit by OLS, testing $H_0: \gamma=0$ (a unit root; $X$ is $I(1)$, a
random walk with no tendency to revert) against
$H_1: \gamma<0$ (stationary, reverting at rate $|\gamma|$). The test
statistic $\hat\gamma/\text{SE}(\hat\gamma)$ looks like an ordinary
$t$-statistic but is **not** $t$-distributed under $H_0$: its true
finite-sample distribution is a known, famously non-standard object
with no closed form, which is precisely why the original
Dickey-Fuller critical value tables were themselves built by Monte
Carlo simulation in the first place. `stationarity.py` does the same
thing honestly rather than hard-coding a borrowed table: it simulates
many pure random walks **at the same sample size as the data being
tested** (the distribution depends materially on $n$), runs the
identical regression on each, and reports the fraction of simulated
null statistics at or below the observed one as the p-value. The
resulting 5%-critical-value estimate (about $-2.80$ to $-2.86$
depending on the simulation run, at large $n$) lands squarely in the
neighbourhood of the published Dickey-Fuller tables, and a **calibration
check** (simulate 100 independent random walks, confirm the empirical
false-positive rate stays near the nominal 5% level, exactly the same
kind of check used for the variance ratio test in
`stochastic-processes`) is run directly in `verify` rather than merely
asserted. Note: a single random walk instance can legitimately land in
the rejection region purely by chance under a correctly calibrated
test, which is why the test suite checks the rejection *rate* across
many trials rather than asserting a fixed outcome for one specific
random seed (an earlier version of this test suite made exactly that
mistake and had to be corrected).

### Model selection: AIC and BIC

$$\text{AIC} = 2k - 2\ln\hat L, \qquad \text{BIC} = k\ln n - 2\ln\hat L.$$

AIC's constant per-parameter penalty makes it comparatively permissive
toward larger models as $n$ grows; BIC's $\ln n$ penalty makes it the
more conservative, parsimony-favouring criterion once $n$ is at all
large ($\ln n > 2$ for $n>e^2\approx7.4$). Both are computed from the
conditional Gaussian log-likelihood
$\ln\hat L = -\tfrac{n}{2}\ln(2\pi\hat\sigma^2) - \text{SSE}/(2\hat\sigma^2)$,
and `select_ar_order`/`select_arma_order` grid-search over candidate
orders, scoring every candidate on the *same* usable observation count
so that a higher-order model is not unfairly rewarded or penalised for
starting its regression on a different stretch of data.

## Application: EUR/USD prices, returns, and volatility, through a classical time-series lens

**Prices are $I(1)$; returns are $I(0)$, exactly as the classical
theory predicts.** Running the from-scratch ADF test on the raw price
level gives a statistic of $-1.48$ against a Monte-Carlo-simulated 5%
critical value of $-3.15$ (p = 0.51): the test fails to reject the
unit root, consistent with prices behaving like a random walk. The
same test on daily log returns gives a statistic of $-59.3$ (p $\approx$
0.00): the unit root is rejected overwhelmingly, confirming returns
are stationary. This is the textbook "prices are integrated, returns
are not" result, demonstrated here with an actual from-scratch test
rather than assumed.

**AR order selection on returns lands on order 0.** Fitting AR(p) for
$p=0,\dots,5$ and scoring by AIC and BIC, both criteria select order 0
(pure white noise): every candidate order's coefficients are tiny
(0.01-0.02 in magnitude), consistent with every sibling repository's
finding of no meaningful linear predictability in EUR/USD returns.

**A genuine ARMA parameter-redundancy pathology, found and explained,
not hidden.** Naively fitting ARMA(1,1) to the real returns converges
to $\hat\phi \approx 0.9645$, $\hat\theta \approx -0.9635$, reproducibly
from multiple different starting points. This looks like a strong,
persistent AR and MA structure; it is not. The AR polynomial
$(1-0.9645L)$ and MA polynomial $(1-0.9635L)$ very nearly cancel,
meaning $(1-\phi L)/(1+\theta L) \approx 1$: the fitted model is
numerically almost indistinguishable from pure white noise, and its
sum of squared residuals (0.22894) is only about 0.06% better than the
plain mean-only (0,0) model's (0.22908), a negligible improvement
purchased with two extra, individually uninterpretable parameters
sitting on a ridge of near-equivalent solutions. This is the classical
**common-factor / parameter-redundancy problem** in ARMA modelling:
when the true process has little or no genuine ARMA structure, the
CSS objective's surface can have a ridge along $\phi\approx-\theta$
that fits equally well almost anywhere along it, and naive point
estimation alone cannot tell the difference. **AIC and BIC resolve it
correctly**: a full grid search over ARMA(p,q) for $p,q\in\{0,1\}$
selects $(0,0)$ by both criteria, exactly because they penalise the
two extra, functionally redundant parameters against the negligible
log-likelihood gain. The lesson generalises well beyond this one
dataset: point-estimating an ARMA model without also running a proper
order-selection criterion can manufacture the appearance of strong
persistence out of pure noise, and the fix is not a better optimizer
but a penalty for parameters that are not earning their keep.

**Squared returns show what raw returns do not.** The ACF of raw
returns sits almost entirely inside the $\pm0.0235$ white-noise band
(one borderline exception at lag 5), and a joint Ljung-Box test across
10 lags gives $Q=16.91$, $p=0.076$: not significant at the
conventional 5% level, though close enough to be worth reporting
honestly rather than rounding down to "clearly white noise." The ACF
of **squared** returns is a completely different picture: every one of
the first 10 lags exceeds the significance band, with values from
0.06 to 0.14, and the joint Ljung-Box test gives $Q=833.85$ with a
p-value on the order of $10^{-172}$. This is the same
volatility-clustering conclusion `stochastic-processes` reached
through a transition-matrix independence test and a Poisson dispersion
test, and `black-scholes` reached through a delta-hedging P&L
correlation, now reached a fourth way, through the single most
standard diagnostic in classical Box-Jenkins time series analysis:
look at the ACF of the squared series. Four unrelated methods, one
answer.

## Limitations

- The CSS estimator used throughout for MA and ARMA models conditions
  on zero pre-sample innovations rather than modelling their exact
  stationary distribution (the full innovations-algorithm likelihood);
  this is the standard simplification and is asymptotically
  negligible, but introduces a small finite-sample approximation
  particularly for short series or MA/ARMA coefficients close to the
  invertibility boundary.
- The ADF test implemented here has no drift or trend term option
  (only the no-constant-in-the-differenced-equation, or rather the
  intercept-only case is implemented via the `alpha` term); a full
  implementation would also offer trend-inclusive variants with their
  own, different simulated critical values.
- Durand-Kerner root-finding, like any iterative numerical root
  finder, is not guaranteed to converge for pathological polynomials
  (very close or repeated roots, extreme coefficient scales); the
  tolerance-based stopping rule used here is adequate for the
  low-order (p typically under 10) characteristic polynomials this
  project's AR models produce.
- The ARMA parameter-redundancy finding is specific to conditional
  least squares point estimation without a penalty term; it is not a
  claim that ARMA(1,1) models are never appropriate, only that fitting
  one without also checking AIC/BIC against simpler alternatives can
  be actively misleading on data that does not need it.

## CLI examples

```bash
# ACF and PACF on real EUR/USD returns
python3 cli.py acf --fx --max-lag 10

# Augmented Dickey-Fuller test
python3 cli.py adf --fx --lags 1 --trials 2000

# Fit AR(p) by Yule-Walker and OLS
python3 cli.py fitar --fx --order 1

# Fit MA(q) by conditional sum of squares
python3 cli.py fitma --fx --order 1

# Fit ARMA(p,q) by conditional sum of squares
python3 cli.py fitarma --fx --p 1 --q 1

# AIC/BIC order selection
python3 cli.py selectorder --fx --max-order 5

# Ljung-Box white-noise test
python3 cli.py ljungbox --fx --max-lag 10

# The volatility-clustering demonstration: raw vs squared returns
python3 cli.py volclustering --max-lag 10

# Cross-check every formula against scipy/numpy oracles, calibration
# checks, and synthetic ground-truth recovery
python3 cli.py verify
```
