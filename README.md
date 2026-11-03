# Quant From First Principles

Thirty mathematical systems, implemented from their definitions, in Python,
with no NumPy or SciPy inside the implementations themselves. Those libraries
appear only as independent oracles used to verify the code is right.

All 30 live in this one repository: the 14-project foundational
curriculum in phases 1 to 5, and the 16-project quant engine built on top
of it in phases 6 to 9.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-2%2C247%20passing-brightgreen)

## Why this exists

Most quantitative and data-science code hides its mathematics behind NumPy,
SciPy, and statsmodels. That's the right call in production. It's the wrong
call if the goal is to actually understand what those libraries are doing
underneath the API.

This is a personal mathematics curriculum built the other way around: derive
the mathematics, implement it from the definition, verify it against the
library everyone else just calls directly, then apply it to a single real
dataset that runs through almost every project — daily EUR/USD exchange
rates from the Federal Reserve (FRED series `DEXUSEU`), 1999-01-04 to
2026-08-21, 6,930 observations.

The goal was never to replace NumPy or SciPy. It was to earn the right to
use them by first understanding what they compute and why it works.

## How everything is verified

Every implementation follows the same pipeline:

```
mathematical definition  →  from-scratch implementation  →  unit tests
                                                                  ↓
                                                         independent oracle
                                                          (numpy / scipy)
                                                                  ↓
                                                            pass / fail
```

Each project ships a `cli.py verify` command that runs its from-scratch
output against the equivalent NumPy/SciPy/statsmodels call and reports a
pass/fail table. Nothing is "verified" by eyeballing a plot.

## Repository structure

Nine phases. Phases 1 to 5 are the foundational curriculum, 14 projects,
discrete math up through PCA and optimization. Phases 6 to 9 are the 16
more advanced projects that build on that foundation: stochastic
calculus, option pricing, Bayesian inference, and the rest of a working
quant toolkit. Each project folder is still complete on its own, with
its own README, tests, and `cli.py verify` command, and each one arrived
here with its full commit history intact.

The first five phases follow the order things were actually learned in,
not a strict prerequisite chain. Decision science comes before formal
calculus because that's the order curiosity took, not because expected
value depends on derivatives. Phases 6 to 9 are grouped by mathematical
family rather than build order.

```
phase1_foundations/        Projects 1-2   Discrete Math, Probability
                            Nothing else here is possible without these two.

phase2_data_inference/     Projects 3-5   Statistics, Sampling Theory
                            Once you can reason about chance, the next
                            question is what a batch of real data lets
                            you conclude from it.

phase3_decision_science/   Projects 6-8   Expected Value, Risk Theory
                            Applying probability and inference directly to
                            decisions, risk, and whether a trading edge is
                            statistically real -- ahead of calculus, because
                            this is what the data-inference phase led to.

phase4_calculus/           Projects 9-11  Differentiation, Integration, Gradients
                            The continuous-math machinery most of what
                            surrounds it quietly assumes.

phase5_advanced_tools/     Projects 12-14 Linear Algebra, PCA, Optimization
                            Linear algebra as the engine underneath PCA,
                            optimization as the engine underneath modern
                            machine learning -- the natural place to end
                            the foundational curriculum.

phase6_stochastic_calculus/      Stochastic processes, SDEs, Black-Scholes
                                  Randomness evolving through time, and the
                                  pricing built on top of it.

phase7_numerical_methods/        Autodiff, ODEs, Fourier, PDEs, quasi-Monte Carlo
                                  The computational engines everything else
                                  quietly leans on.

phase8_inference_and_learning/   Time series, Bayesian, HMMs, information theory, SVD
                                  Extracting structure and parameters from
                                  data rather than simulating forward.

phase9_quant_finance/            Convex optimization, portfolios, model risk
                                  Where the rest of it gets applied to an
                                  actual portfolio, then stress-tested.

profile/                         Source of the TJT-Pro GitHub profile page
```

### Foundations (phases 1 to 5)

| # | Project | Core mathematics | Tests |
|---|---|---|---|
| 1 | [Logic & Combinatorics Toolkit](phase1_foundations/project1_logic_combinatorics) | Propositional logic, proof rules, set theory, combinatorics | 57 |
| 2 | [Probability Simulator & Bayes Engine](phase1_foundations/project2_probability) | Discrete/continuous distributions, Bayes' theorem | 37 |
| 3 | [Descriptive Statistics Library](phase2_data_inference/project3_statistics) | Moments, quantiles, skewness/kurtosis | 55 |
| 4 | [Hypothesis Testing Engine](phase2_data_inference/project4_hypothesis_testing) | t-tests, chi-square, ANOVA, power analysis | 68 |
| 5 | [Bootstrap Sampler & CLT Demonstrator](phase2_data_inference/project5_bootstrap_clt) | Resampling, confidence intervals, CLT | 42 |
| 6 | [Expected Value & Decision Theory](phase3_decision_science/project6_expected_value) | Utility theory, the Kelly criterion | 52 |
| 7 | [Monte Carlo Risk & Drawdown Analyser](phase3_decision_science/project7_monte_carlo_risk) | Monte Carlo simulation, VaR/CVaR | 88 |
| 8 | [Strategy Backtester](phase3_decision_science/project8_strategy_backtester) | Sharpe/Sortino/Calmar, permutation testing for edge | 100 |
| 9 | [Numerical Differentiation Engine](phase4_calculus/project9_numerical_differentiation) | Finite differences, Richardson extrapolation | 84 |
| 10 | [Numerical Integrator & Taylor Series Expander](phase4_calculus/project10_numerical_integration) | Quadrature rules, Taylor series | 111 |
| 11 | [Gradient Field Visualiser](phase4_calculus/project11_gradient_field) | Partial derivatives, divergence, curl | 70 |
| 12 | [Matrix Engine from Scratch](phase5_advanced_tools/project12_matrix_engine) | LU, QR, least squares, eigenvalues, condition numbers | 71 |
| 13 | [PCA from Scratch](phase5_advanced_tools/project13_pca) | Jacobi eigenvalue algorithm, PCA | 42 |
| 14 | [Gradient Descent Optimizer & LP Solver](phase5_advanced_tools/project14_optimizer) | Adam, RMSProp, Nelder-Mead, simplex LP | 36 |

**913 tests, 14 projects.**

### Quant engine (phases 6 to 9)

Sixteen more advanced projects, grouped below by mathematical family
rather than by build order, since that's how they're actually related to
each other. Each folder is a complete project with its own README, tests,
and `cli.py verify` command.

**Stochastic calculus & derivatives** -- randomness evolving through time,
and the pricing built on top of it.

| # | Project | Core mathematics | Tests |
|---|---|---|---|
| 15 | [stochastic-processes](phase6_stochastic_calculus/stochastic-processes) | Random walks, Markov chains, Poisson processes, Brownian motion, martingales, first-passage times | 285 |
| 24 | [stochastic-differential-equations](phase6_stochastic_calculus/stochastic-differential-equations) | Euler-Maruyama, Milstein scheme, Ito's lemma | 55 |
| 16 | [black-scholes](phase6_stochastic_calculus/black-scholes) | Closed-form pricing, Greeks, implied volatility, binomial trees, Monte Carlo pricing | 84 |

**424 tests.**

**Numerical & computational methods** -- the computational engines
everything else quietly leans on.

| # | Project | Core mathematics | Tests |
|---|---|---|---|
| 17 | [automatic-differentiation](phase7_numerical_methods/automatic-differentiation) | Dual numbers, forward mode, reverse-mode autodiff, backpropagation | 70 |
| 20 | [ode-engine](phase7_numerical_methods/ode-engine) | Euler methods, RK4, adaptive step-doubling | 45 |
| 21 | [fourier-analysis](phase7_numerical_methods/fourier-analysis) | DFT, Cooley-Tukey FFT, convolution, periodograms | 58 |
| 25 | [numerical-pde](phase7_numerical_methods/numerical-pde) | Finite differences, von Neumann stability, Black-Scholes PDE | 66 |
| 28 | [quasi-monte-carlo](phase7_numerical_methods/quasi-monte-carlo) | Van der Corput and Halton sequences, discrepancy measures | 56 |

**295 tests.**

**Inference & learning from data** -- extracting structure and parameters
from data rather than simulating a process forward.

| # | Project | Core mathematics | Tests |
|---|---|---|---|
| 18 | [time-series-mathematics](phase8_inference_and_learning/time-series-mathematics) | ACF/PACF, AR/MA/ARMA, Yule-Walker, ADF stationarity testing | 105 |
| 19 | [bayesian-inference](phase8_inference_and_learning/bayesian-inference) | Conjugate priors, grid approximation, Metropolis-Hastings MCMC | 76 |
| 26 | [hidden-markov-models](phase8_inference_and_learning/hidden-markov-models) | Forward-backward algorithm, Viterbi, Baum-Welch | 45 |
| 27 | [information-theory](phase8_inference_and_learning/information-theory) | Entropy, KL divergence, mutual information, Huffman coding | 97 |
| 22 | [svd-from-scratch](phase8_inference_and_learning/svd-from-scratch) | Singular value decomposition, pseudoinverse, condition numbers | 60 |

**383 tests.**

**Quant finance engineering** -- where the rest of it gets applied to an
actual portfolio, and then gets stress-tested for where it lies to you.

| # | Project | Core mathematics | Tests |
|---|---|---|---|
| 23 | [convex-optimization](phase9_quant_finance/convex-optimization) | Newton's method, projected gradient descent, KKT/duality | 92 |
| 29 | [portfolio-mathematics](phase9_quant_finance/portfolio-mathematics) | Markowitz frontier, CAPM, Kelly criterion | 76 |
| 30 | [model-risk](phase9_quant_finance/model-risk) | VaR backtesting, AIC/BIC, backtest overfitting demonstration | 64 |

**232 tests.**

**1,334 tests, 16 projects.**

<details>
<summary>Build order (the sequence these were actually written in)</summary>

1. stochastic-processes
2. black-scholes
3. automatic-differentiation
4. time-series-mathematics
5. bayesian-inference
6. ode-engine
7. fourier-analysis
8. svd-from-scratch
9. convex-optimization
10. stochastic-differential-equations
11. numerical-pde
12. hidden-markov-models
13. information-theory
14. quasi-monte-carlo
15. portfolio-mathematics
16. model-risk

</details>

**2,247 tests, 30 projects, one dataset.**

## Setup

Each project is self-contained: standard library only for the
implementation, `pytest` for the test suite, `numpy`/`scipy` only inside
`cli.py verify`.

```bash
cd phase1_foundations/project1_logic_combinatorics
python3 -m pytest
python3 cli.py verify
```

Run tests from inside a project folder rather than from the repository
root. Several projects have modules with the same file name (for
example `fx_application.py`), so each suite is collected on its own,
exactly as CI does on every push.

## What "from scratch" means, and doesn't

Every core implementation is pure Python: no NumPy, SciPy, pandas, or
scikit-learn calls inside the mathematics itself. `numpy` and `scipy`
appear in exactly two places per project: the `cli.py verify` command,
and the test files that check the implementation against them. That
distinction is enforced, not just claimed — every core module can be
grepped for library imports and none will turn up.

This is not a performance library. The matrix engine's determinant is
O(n!) by cofactor expansion for small matrices; NumPy's is not. Where an
implementation has a known limitation, it's stated in that project's
README rather than glossed over.

## License

MIT. Use it, fork it, learn from it.

## Contact

Trading, tools, contacts: [TJT_Pro on TradingView](https://www.tradingview.com/u/TJT_Pro/)
