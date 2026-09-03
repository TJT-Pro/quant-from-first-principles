# Phase 3 - Decision Science

Expected Value, Risk Theory, Strategy Backtesting. This is where
probability and statistics stop being about what is and become about
what to do.

Expected value gives the objective yardstick for any uncertain choice.
Decision rules handle the cases where you cannot assign probabilities to
states. Expected utility and risk aversion explain why two people can
look at the same lottery and make different rational choices. Monte
Carlo simulation explores the full distribution of outcomes rather than
just the mean. The strategy backtester then applies all of it to real
trading data and asks the only question that matters: is the edge real?

| Project | Folder | Tests | Status |
|---|---|---|---|
| 6 - Expected Value & Decision Theory | `project6_expected_value/` | 52 | done |
| 7 - Monte Carlo Risk & Drawdown Analyser | `project7_monte_carlo_risk/` | 88 | done |
| 8 - Strategy Backtester | `project8_strategy_backtester/` | 100 | done |

**240 tests total across Phase 3, all passing.**

## What is implemented

**Project 6** builds expected value, variance, all six classical decision
rules (expected-value, maximax, maximin, minimax-regret, Laplace,
Hurwicz), dominance checking, five utility families with exact inverses
and Arrow-Pratt risk measures, certainty equivalents, risk premia,
prospect theory with Tversky-Kahneman parameters, the St. Petersburg
paradox, and the full Kelly criterion with empirical optimal fraction
search.

**Project 7** builds return simulation from scratch via Box-Muller
(normal and Student-t), historical and parametric VaR and CVaR, equity
curves, drawdown series, maximum drawdown, drawdown duration, Calmar
ratio, Monte Carlo simulation across thousands of paths, ruin
probability with bootstrap confidence intervals, and a full risk report.
The inverse normal CDF uses Acklam's rational approximation (accurate to
1.15e-9) -- no scipy.

**Project 8** builds a full strategy backtester: trade log validation,
expectancy decomposition (win rate, payoff ratio, profit factor), Sharpe
ratio, Sortino ratio, Calmar ratio, drawdown, a one-sample t-test with
the incomplete beta CDF implemented from scratch, confidence intervals on
mean return, bootstrap equity curves with element-wise percentile bands,
permutation tests, and two demonstrator strategies (MA crossover and
mean reversion) to generate trade logs from a raw return series.

## The data

All three projects use `data/DEXUSEU_returns.csv`, the daily EUR/USD
return series from FRED (6930 observations, 1999-2026). It is committed
to each project's folder.

## Running

From the repo root:

```bash
python3 -m pytest phase3_decision_science/
```

Or run one project at a time:

```bash
cd phase3_decision_science/project6_expected_value  && python3 -m pytest test_decision_theory.py -v
cd phase3_decision_science/project7_monte_carlo_risk && python3 -m pytest test_risk_analyser.py -v
cd phase3_decision_science/project8_strategy_backtester && python3 -m pytest test_backtester.py -v
```

Verify against numpy/scipy oracles:

```bash
cd phase3_decision_science/project6_expected_value  && python3 cli.py verify
cd phase3_decision_science/project7_monte_carlo_risk && python3 cli.py verify
cd phase3_decision_science/project8_strategy_backtester && python3 cli.py verify
```
