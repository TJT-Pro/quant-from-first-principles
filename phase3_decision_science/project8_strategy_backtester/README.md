# Project 8 - Strategy Backtester with Statistical Edge Verification

A backtester built from the definitions. The central question is not
"did this strategy make money?" but "is the edge statistically real, or
is it noise?". A strategy that looks profitable in a backtest might just
be getting lucky. This project gives you the tools to tell the
difference.

| Idea | What it answers |
|---|---|
| Expectancy | the average return per trade; the fundamental measure of edge |
| Win rate / loss rate | how often the strategy wins and loses |
| Profit factor | gross profit divided by gross loss |
| Payoff ratio | average win divided by average loss |
| Sharpe ratio | mean return per unit of total volatility, annualised |
| Sortino ratio | mean return per unit of downside volatility, annualised |
| Max drawdown | largest peak-to-trough decline of the equity curve |
| Calmar ratio | annualised return divided by max drawdown |
| t-test | is the mean return statistically different from zero |
| Bootstrap CI | how wide is the uncertainty around any metric |
| Permutation test | could the observed edge be explained by random ordering |

## Expectancy

The mean return per trade. If the strategy takes many trades, expectancy
times the number of trades per year gives the expected annual
contribution before compounding.

It decomposes exactly as:

    expectancy = win_rate * avg_win + loss_rate * avg_loss

This decomposition matters because it shows which component is carrying
the edge. A high payoff ratio strategy (large wins, large losses) behaves
very differently from a high win rate strategy (many small wins), even
if expectancy is identical.

## Sharpe and Sortino

Both measure return per unit of risk, annualised by multiplying by
sqrt(periods_per_year).

    Sharpe  = (mean(r) - r_f) / std(r) * sqrt(T)
    Sortino = (mean(r) - r_f) / downside_std(r) * sqrt(T)

The Sortino's denominator uses only returns below the risk-free rate.
For a strategy with positive skew (small losses, large wins) the Sortino
is higher because the upside volatility is not penalised.

## t-test on returns

The null hypothesis is that mean return equals mu0 (default 0). The
t-statistic is

    t = (x_bar - mu0) / (s / sqrt(n))

where x_bar is the sample mean, s is the sample standard deviation, and
n is the number of trades. Under H0 this follows a t(n-1) distribution.
The p-value is two-sided: P(|T| >= |t_observed|).

A low p-value means the observed mean would be very unlikely if the true
edge were zero. A high p-value does not mean the strategy has no edge --
it might just mean you do not have enough trades to detect it yet.

The confidence interval for the mean uses the same t distribution:

    CI = x_bar +/- t_{alpha/2, n-1} * s / sqrt(n)

## Permutation test

The permutation test addresses a different question. Shuffling the
returns destroys any sequential structure (momentum, mean reversion,
clustering) while preserving the individual return magnitudes. If the
observed statistic (e.g. Sharpe) is no better than the shuffled
versions, the edge is not in the individual trade outcomes -- it is in
their ordering, which the strategy may or may not be capturing.

The p-value is the fraction of shuffled outcomes as extreme as or more
extreme than the observed one.

## Bootstrap

The bootstrap resamples the trade log with replacement and recomputes
any statistic on each resample. This gives the sampling distribution of
the statistic without any distributional assumption.

For the equity curve, resampling the trade returns generates an
ensemble of possible equity curves consistent with the observed return
distribution. The 5th and 95th percentile curves bound 90% of the
probable outcomes.

## Included strategies (demonstration only)

Two simple strategies are included to provide a concrete trade log from
a raw return series. Neither is recommended as a live strategy.

**MA crossover**: goes long when the fast MA crosses above the slow MA,
flat otherwise. Trade return is the next period's return.

**Mean reversion**: goes long when the most recent return is more than
z_threshold standard deviations below the rolling mean. Expects a
bounce.

Running both on EUR/USD daily returns (1999-2024, 6,930 observations):

| | MA crossover (10/30) | Mean reversion (z=1.5) |
|---|---|---|
| Trades | 3,505 | 415 |
| Expectancy | -0.007% | -0.054% |
| p-value | 0.49 | 0.05 |
| Sharpe | -0.19 | -1.53 |

Neither has a statistically significant positive edge. That is the
correct result. EUR/USD buy-and-hold has a near-zero mean return with
high volatility. Any strategy that takes a subset of those returns
inherits the same problem. The backtester is working.

## Quick start

Full report on any CSV with a column of per-trade returns:

```
python3 cli.py report data/DEXUSEU_returns.csv --col return
```

t-test on whether mean return is different from zero:

```
python3 cli.py ttest data/DEXUSEU_returns.csv --col return
```

Bootstrap equity curve with 2000 resamples:

```
python3 cli.py boot data/DEXUSEU_returns.csv --col return --n-boot 2000
```

Permutation test on Sharpe ratio:

```
python3 cli.py permute data/DEXUSEU_returns.csv --col return --stat sharpe
```

Run the MA crossover strategy and backtest it:

```
python3 cli.py simulate data/DEXUSEU_returns.csv --col return --strategy ma
```

Cross-check everything against numpy and scipy:

```
python3 cli.py verify
```

## Running the tests

```
python3 -m pytest test_backtester.py -v
```

100 tests. Every formula is checked against a hand-computed value or a
known identity. Fixed seeds make Monte Carlo tests deterministic.

## Using your own trade log

Supply a CSV where one column holds the per-trade return as a decimal
(0.02 means 2%). Optional columns: date, direction (long or short),
entry, exit.

```
python3 cli.py report my_trades.csv --col return
```

To test only your long trades:

```python
import backtester as bt
import csv

trades = []
with open("my_trades.csv") as f:
    for row in csv.DictReader(f):
        trades.append({"return": float(row["return"]),
                       "direction": row["direction"]})

log   = bt.load_trade_log(trades)
longs = bt.filter_by_direction(log, "long")
r     = bt.returns_from_log(longs)
print(bt.t_test_one_sample(r))
```
