# math-implementations

Fourteen Python projects, one per mathematical topic, built in sequence
from a Mathematics Mastery Roadmap, a plan I drafted to aid guide my study. Every concept is implemented from
its definition -- numpy and scipy appear only as independent verification
oracles, never inside the implementations themselves.

## Repository layout

```
phase1_foundations/        Projects 1-2   Discrete Math, Probability
phase2_data_inference/     Projects 3-5   Statistics, Sampling Theory
phase3_decision_science/   Projects 6-8   Expected Value, Risk Theory
phase4_calculus/           Projects 9-11  Differentiation, Integration, Gradients
phase5_advanced_tools/     Projects 12-14 Linear Algebra, PCA, Optimisation
```

## Project status

| # | Project | Phase | Tests | Verify |
|---|---|---|---|---|
| 1 | Logic & Combinatorics Toolkit | 1 Foundations | 57 | 23 checks |
| 2 | Probability Simulator & Bayes Engine | 1 Foundations | 37 | 16 checks |
| 3 | Descriptive Statistics Library | 2 Data & Inference | 55 | 20 checks |
| 4 | Hypothesis Testing Engine | 2 Data & Inference | 68 | 3729 checks |
| 5 | Bootstrap Sampler & CLT Demonstrator | 2 Data & Inference | 42 | 19 checks |
| 6 | Expected Value & Decision Theory | 3 Decision Science | 52 | 34 checks |
| 7 | Monte Carlo Risk & Drawdown Analyser | 3 Decision Science | 88 | 31 checks |
| 8 | Strategy Backtester | 3 Decision Science | 100 | 19 checks |
| 9 | Numerical Differentiation Engine | 4 Calculus | 84 | 44 checks |
| 10 | Numerical Integrator & Taylor Series | 4 Calculus | 111 | 53 checks |
| 11 | Gradient Field Visualiser | 4 Calculus | 70 | 27 checks |
| 12 | Matrix Engine from Scratch | 5 Advanced Tools | 71 | 21 checks |
| 13 | PCA from Scratch | 5 Advanced Tools | 42 | 18 checks |
| 14 | Gradient Descent Optimizer & LP Solver | 5 Advanced Tools | 36 | 23 checks |

**913 tests passing across all 14 projects.**

## Setup

Python 3.11+. Each project's core needs only the standard library. The
`verify` command in every project uses numpy/scipy as independent oracles:

```bash
pip install numpy scipy matplotlib
```

## Running all tests

From the repo root:

```bash
python3 -m pytest phase1_foundations/ phase2_data_inference/ \
                  phase3_decision_science/ phase4_calculus/ \
                  phase5_advanced_tools/
```

Or one project at a time, e.g.:

```bash
python3 -m pytest phase5_advanced_tools/project14_optimizer/test_optimizer.py -v
```

## Running the oracles

Every project ships a `verify` command that cross-checks the hand-built
code against numpy/scipy. Run from inside the project folder:

```bash
cd phase1_foundations/project1_logic_combinatorics  && python3 cli.py verify
cd phase1_foundations/project2_probability           && python3 cli.py verify
cd phase2_data_inference/project3_statistics         && python3 cli.py verify
cd phase2_data_inference/project4_hypothesis_testing && python3 cli.py verify
cd phase2_data_inference/project5_bootstrap_clt      && python3 cli.py verify
cd phase3_decision_science/project6_expected_value   && python3 cli.py verify
cd phase3_decision_science/project7_monte_carlo_risk && python3 cli.py verify
cd phase3_decision_science/project8_strategy_backtester && python3 cli.py verify
cd phase4_calculus/project9_numerical_differentiation && python3 cli.py verify
cd phase4_calculus/project10_numerical_integration   && python3 cli.py verify
cd phase4_calculus/project11_gradient_field          && python3 cli.py verify
cd phase5_advanced_tools/project12_matrix_engine     && python3 cli.py verify
cd phase5_advanced_tools/project13_pca               && python3 cli.py verify
cd phase5_advanced_tools/project14_optimizer         && python3 cli.py verify
```

## The data

Projects 3 through 14 use the same real dataset: daily EUR/USD exchange
rates from the Federal Reserve (FRED series DEXUSEU), from 1999-01-04
to 2026-08-21, giving 6930 daily returns. The CSV lives in each
project's `data/` folder. No external fetch is needed to run any test.

## Design rules

- **No hardcoded values.** Every number in output is computed from the
  data or from the formula.
- **Standard library only in implementations.** numpy, scipy, and
  matplotlib are for checking and plotting, never for the core logic.
- **Every formula is verified.** The `verify` command in each project
  runs the hand-built result against an independent oracle and prints
  PASS or FAIL per check.
- **Order is strict.** Each phase builds on the mathematics of the
  previous one. Phase 5 imports the matrix engine from Phase 5 project
  12; everything else is self-contained.

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
difference methods with Richardson extrapolation, error analysis,
Jacobian and Hessian matrices, product and chain rules verified
numerically, bond duration and convexity. Numerical integration via
Riemann, trapezoid, Simpson, Gaussian quadrature, Monte Carlo, and
adaptive methods. Taylor series expansion and remainder bounds. Gradient
fields, partial derivatives, directional derivatives, Laplacian, critical
point classification, gradient descent with trajectory, divergence and
curl of vector fields.

**Phase 5 - Advanced Tools.** Complete linear algebra engine (LU
decomposition, QR decomposition, eigenvalues, least squares, condition
number). PCA from scratch via the Jacobi eigenvalue algorithm applied to
EUR/USD lag matrices. Gradient descent optimizers (vanilla, momentum,
RMSProp, Adam, Nelder-Mead, backtracking line search), the revised
simplex method for linear programming, logistic regression, and portfolio
optimisation via LP.
