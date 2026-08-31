# math-implementations

Fourteen Python projects, one per mathematical topic, built in sequence
from the Mathematics Mastery Roadmap. Every concept is implemented from
its definition -- numpy and scipy appear only as independent verification
oracles, never inside the implementations themselves.

## Repository layout

```
phase1_foundations/        Projects 1-2   Discrete Math, Probability
phase2_data_inference/     Projects 3-5   Statistics, Sampling Theory
phase3_decision_science/   Projects 6-8   Expected Value, Risk Theory
phase4_calculus/           Projects 9-11  Calculus I, II, III
phase5_advanced_tools/     Projects 12-14 Linear Algebra, Optimization
```

## Project status

| # | Project | Phase | Tests | Status |
|---|---|---|---|---|
| 1 | Logic & Combinatorics Toolkit | 1 Foundations | 57 | done |
| 2 | Probability Simulator & Bayes Engine | 1 Foundations | 37 | done |
| 3 | Descriptive Statistics Library | 2 Data & Inference | 55 | done |
| 4 | Hypothesis Testing Engine | 2 Data & Inference | 68 | done |
| 5 | Bootstrap Sampler & CLT Demonstrator | 2 Data & Inference | 42 | done |
| 6 | Expected Value & Decision Theory | 3 Decision Science | 52 | done |
| 7 | Monte Carlo Risk & Drawdown Analyser | 3 Decision Science | 88 | done |
| 8 | Strategy Backtester | 3 Decision Science | 100 | done |
| 9 | Numerical Differentiation Engine | 4 Calculus | 84 | done |
| 10 | Numerical Integrator & Taylor Series | 4 Calculus | 111 tests · 53 checks | done |
| 11 | Gradient Field Visualiser | 4 Calculus | 70 tests · 27 checks | done |
| 12 | Matrix Engine from Scratch | 5 Advanced Tools | 71 tests · 21 checks | done |
| 13 | PCA from Scratch | 5 Advanced Tools | 42 tests · 18 checks | done |
| 14 | Gradient Descent Optimizer & LP Solver | 5 Advanced Tools | 36 tests · 23 checks | done |

**583 tests passing across 9 completed projects.**

## Setup

Python 3.11+. Each project's core needs only the standard library. The
`verify` command in every project needs the scientific stack:

```bash
pip install --break-system-packages numpy scipy matplotlib
```

## Running all tests

```bash
python3 -m pytest phase1_foundations/project1_logic_combinatorics/test_logic_combinatorics.py
python3 -m pytest phase1_foundations/project2_probability/test_probability_simulator.py
python3 -m pytest phase2_data_inference/project3_statistics/test_descriptive_stats.py
python3 -m pytest phase2_data_inference/project4_hypothesis_testing/test_hypothesis_tests.py
python3 -m pytest phase2_data_inference/project5_bootstrap_clt/test_bootstrap_sampler.py
python3 -m pytest phase3_decision_science/project6_expected_value/test_decision_theory.py
python3 -m pytest phase3_decision_science/project7_monte_carlo_risk/test_risk_analyser.py
python3 -m pytest phase3_decision_science/project8_strategy_backtester/test_backtester.py
python3 -m pytest phase4_calculus/project9_numerical_differentiation/test_differentiator.py
```

Or run every completed project in one shot from the repo root:

```bash
python3 -m pytest phase1_foundations/ phase2_data_inference/ \
                  phase3_decision_science/ phase4_calculus/
```

## Running the oracles

Every project ships a `verify` command that cross-checks the hand-built
code against numpy and scipy. Run each from inside its project folder:

```bash
cd phase1_foundations/project1_logic_combinatorics && python3 cli.py verify
cd phase1_foundations/project2_probability         && python3 cli.py verify
cd phase2_data_inference/project3_statistics       && python3 cli.py verify
cd phase2_data_inference/project4_hypothesis_testing && python3 cli.py verify
cd phase2_data_inference/project5_bootstrap_clt    && python3 cli.py verify
cd phase3_decision_science/project6_expected_value && python3 cli.py verify
cd phase3_decision_science/project7_monte_carlo_risk && python3 cli.py verify
cd phase3_decision_science/project8_strategy_backtester && python3 cli.py verify
cd phase4_calculus/project9_numerical_differentiation && python3 cli.py verify
```

## The data

Projects 3 through 9 all use the same real dataset: daily EUR/USD
exchange rates from the Federal Reserve (FRED series DEXUSEU), from
1999-01-04 to 2026-08-21, giving 6930 daily returns. The CSV lives in
each project's `data/` folder. It was downloaded once and is committed
as-is; no external fetch is needed to run any test.

## Design rules

Every project follows the same rules:

- **No hardcoded values.** Every number that appears in output is
  computed from the data or from the formula.
- **Standard library only in implementations.** numpy, scipy, and
  matplotlib are for checking and plotting, never for the core logic.
- **Every formula is verified.** The `verify` command in each project
  runs the hand-built result against an independent oracle and prints
  PASS or FAIL per check.
- **Order is strict.** Each phase builds on the mathematics of the
  previous one. Running phase 5 without understanding phase 1 would
  produce correct outputs but no understanding.

## What each phase covers

**Phase 1 - Foundations.** Propositional logic, sets, combinatorics,
graph traversal, proof validation. Probability from sample spaces through
Bayes' theorem, all major discrete and continuous distributions, the law
of large numbers and central limit theorem -- all implemented without
scipy's distribution functions.

**Phase 2 - Data and Inference.** Descriptive statistics from scratch,
every hypothesis test built from the distribution CDFs, four styles of
bootstrap confidence interval, jackknife bias correction, permutation
tests, and a CLT demonstrator on real EUR/USD return data.

**Phase 3 - Decision Science.** Expected value, all six classical
decision rules, expected utility with five utility families, prospect
theory, Kelly criterion. Monte Carlo simulation from Box-Muller through
VaR, CVaR, drawdown, and ruin probability. Full strategy backtesting
with t-test edge verification, Sharpe, Sortino, Calmar, bootstrap equity
curves, and permutation tests.

**Phase 4 - Calculus.** Numerical differentiation by four finite
difference methods with error analysis, gradients, partial derivatives,
Jacobian, product and chain rules verified numerically, gradient descent,
and bond duration and convexity as finance applications. (Projects 10
and 11 pending.)

**Phase 5 - Advanced Tools.** Matrix engine, PCA, and a gradient descent
optimizer with linear programming solver. (Pending.)
