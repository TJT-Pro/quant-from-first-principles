# Project 7 - Monte Carlo Risk & Drawdown Analyser

A risk analysis library built entirely from the definitions. No numpy,
scipy or pandas inside the implementations. Those libraries appear only
in the verify command as independent oracles that confirm the hand-built
code is correct.

| Idea | What it answers |
|---|---|
| Return simulation | how do synthetic paths behave if the past distribution continues |
| Value at Risk | what loss will not be exceeded in 95% of periods |
| CVaR / Expected Shortfall | what is the average loss when VaR is breached |
| Drawdown | how far has the account fallen from its previous peak |
| Maximum drawdown | the largest peak-to-trough decline over the whole series |
| Drawdown duration | how long the account spent below its previous peak |
| Calmar ratio | annualised return per unit of maximum drawdown |
| Ruin probability | what fraction of simulated paths hit a drawdown limit |
| Bootstrap CI | how certain are we about any of these estimates |

## Return simulation

Two distributions are supported.

The **normal** draws use the Box-Muller transform. If U1 and U2 are
independent uniform(0,1) variables then

    Z = sqrt(-2 ln U1) * cos(2 pi U2)

is exactly standard normal. No approximation is involved. The transform
is applied once per draw, and the only inputs are uniform random numbers
from the standard library.

The **Student-t** is useful when you believe daily returns have fatter
tails than the normal. A t(df) variable is

    Z / sqrt(V / df)

where Z is standard normal and V is chi-squared(df). V is itself the
sum of df squared standard normals, so both pieces come from Box-Muller
and the whole draw uses only uniform random numbers.

For df > 2 the t has finite variance df/(df-2). `fit_t` scales sigma so
that the fitted distribution matches the sample variance exactly.

## Value at Risk

**Historical VaR** makes no distributional assumption. Sort the returns,
find the alpha-quantile by linear interpolation, and negate it. A
negative return in the left tail becomes a positive loss.

    VaR_alpha = -Q_alpha(returns)

**Parametric VaR** fits a normal to the returns and uses the inverse
normal CDF (Acklam's rational approximation, accurate to 1.15e-9).

    VaR_alpha = -(mu + sigma * Phi^{-1}(alpha))

Both methods are reported so you can see how much the normality
assumption changes the number.

## CVaR (Expected Shortfall)

VaR marks the boundary of the tail. CVaR averages what is inside the
tail.

**Historical CVaR**: take all returns at or below the VaR threshold and
compute their mean.

    CVaR_alpha = -mean(returns where return <= Q_alpha)

**Parametric CVaR**: for a normal distribution this has a closed form.

    CVaR_alpha = -(mu - sigma * phi(z_alpha) / alpha)

where phi is the standard normal PDF and z_alpha = Phi^{-1}(alpha). CVaR
is always greater than or equal to VaR at the same confidence level.

## Drawdown

The drawdown at time t is how far the account has fallen from the
highest point it has ever reached up to t.

    dd(t) = (peak(t) - value(t)) / peak(t)

where peak(t) = max(value(0), value(1), ..., value(t)).

Maximum drawdown is max(dd(t)) over the entire series. This is the
number that matters for risk management: it tells you the worst
contiguous loss a strategy has ever produced.

Drawdown duration counts all periods where dd(t) > 0. Maximum drawdown
duration is the length of the longest uninterrupted episode below the
previous peak.

## Calmar ratio

    Calmar = annualised_return / max_drawdown

Higher is better. A strategy that earns 20% per year with a 10% maximum
drawdown has a Calmar of 2.0. One that earns 20% with a 40% maximum
drawdown has a Calmar of 0.5 and is much less attractive on a
risk-adjusted basis.

## Monte Carlo simulation

The simulation fits a distribution to the observed returns, generates
n_paths independent synthetic paths of n_periods each, builds an equity
curve for every path, and computes VaR, CVaR, and drawdown statistics
across the cross-section of paths.

The purpose is to answer questions like: given a strategy with this
return distribution, how often would we expect to see a 20% drawdown
over the next year? The historical series is one realisation; the
simulation explores the full space of outcomes consistent with the
fitted distribution.

## Ruin probability

A path is counted as a ruin event if its maximum drawdown is at or above
the threshold. The ruin probability is the fraction of paths that ruin.

A bootstrap confidence interval is computed by resampling the
max-drawdown list with replacement and recalculating the ruin fraction
on each resample. The percentile method gives the 95% interval.

    CI = [Q_{0.025}(boot_estimates), Q_{0.975}(boot_estimates)]

## Quick start

```
pip install numpy scipy
```

Full risk report on the EUR/USD daily return series:

```
python3 cli.py report data/DEXUSEU_returns.csv --col return
```

95% VaR only:

```
python3 cli.py var data/DEXUSEU_returns.csv --col return --alpha 0.05
```

Ruin probability with a 30% drawdown threshold, t distribution, 20,000 paths:

```
python3 cli.py ruin data/DEXUSEU_returns.csv --col return \
    --threshold 0.3 --dist t --df 5 --paths 20000
```

Drawdown metrics on the observed series:

```
python3 cli.py drawdown data/DEXUSEU_returns.csv --col return
```

Cross-check everything against numpy and scipy:

```
python3 cli.py verify
```

## Running the tests

```
python3 -m pytest test_risk_analyser.py -v
```

88 tests. Every formula is checked against a hand-computed value or a
known mathematical identity. The Monte Carlo tests use fixed seeds so
results are reproducible.

## Design decisions

All implementations use only the standard library (`math` and `random`).
The inverse normal CDF (Acklam's approximation) is implemented here
rather than calling scipy because the whole point of these projects is
to build the tools, not import them.

The equity curve raises an error on a return of -1 or worse because a
-100% return means the account is wiped out and subsequent returns are
meaningless. The simulation does not generate such returns from a normal
or t distribution in practice, but the check is there to catch bad
input data.

The t simulation uses integer df for the chi-squared draw. Strictly
speaking, the chi-squared distribution with non-integer df requires the
gamma function, which goes beyond what this project covers. If you need
non-integer df, df=5 and df=6 bracket most practical cases.
