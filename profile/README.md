# TJT-Pro

Hey. I'm TJT-Pro. A medic, short-term trader, self-taught developer,
and somewhere along the way, a math enthusiast who couldn't leave it
alone.

Medicine teaches you not to trust a symptom until you understand the
mechanism underneath it. I got fascinated with patterns a few years ago,
watching markets move and noticing they weren't random so much as
structured, if you looked at the right timescale. That fascination
turned into my own recursive time framework for tracking where big
money is probably already moving before price confirms it. Somewhere in
building that out, the trader in me turned into someone who wanted to
understand markets as a mathematical object, not just a chart.

This repository is what that curiosity became, and it's my give-back
for it: derive the mathematics myself, implement it from the
definition, verify it against NumPy/SciPy/statsmodels so I can't fool
myself, then run it against one real dataset -- daily EUR/USD exchange
rates from the Federal Reserve, 1999-01-04 to 2026-08-21 -- and see what
actually survives contact with real data.

No NumPy or SciPy inside any implementation here. They appear only as
the check that the from-scratch version is right.

**2,247 tests. 30 projects. One dataset.**

Together, these 30 projects make up **Quant From First Principles** --
derive it, implement it, verify it, apply it to real data. The
foundational curriculum lives in the repository of that name; the more
advanced projects built on top of it are each published as their own
repository, listed below.

Yours: [TJT_Pro on TradingView](https://www.tradingview.com/u/TJT_Pro/)

## Start here

The foundational curriculum lives in
[quant-from-first-principles](https://github.com/TJT-Pro/quant-from-first-principles):
14 projects, phase 1 through 5 -- discrete math and probability,
statistics and inference, decision theory and risk, calculus, linear
algebra and optimization. 913 tests.

## The quant engine

Sixteen more advanced repositories built on that foundation, published
independently so each is a complete, discoverable piece of work on its
own. Publishing was in progress in the order listed under each
repository's own README; as of now (13/09/2026), all repositories are complete!

**Stochastic calculus & derivatives**

- [stochastic-processes](https://github.com/TJT-Pro/stochastic-processes) -- random walks, Markov chains, Poisson processes, Brownian motion, martingales, first-passage times
- [stochastic-differential-equations](https://github.com/TJT-Pro/stochastic-differential-equations) -- Euler-Maruyama, Milstein scheme, Ito's lemma
- [black-scholes](https://github.com/TJT-Pro/black-scholes) -- closed-form pricing, Greeks, implied volatility, binomial trees, Monte Carlo pricing

**Numerical & computational methods**

- [automatic-differentiation](https://github.com/TJT-Pro/automatic-differentiation) -- dual numbers, forward mode, reverse-mode autodiff, backpropagation
- [ode-engine](https://github.com/TJT-Pro/ode-engine) -- Euler methods, RK4, adaptive step-doubling
- [fourier-analysis](https://github.com/TJT-Pro/fourier-analysis) -- DFT, Cooley-Tukey FFT, convolution, periodograms
- [numerical-pde](https://github.com/TJT-Pro/numerical-pde) -- finite differences, von Neumann stability, the Black-Scholes PDE
- [quasi-monte-carlo](https://github.com/TJT-Pro/quasi-monte-carlo) -- van der Corput and Halton sequences, discrepancy measures

**Inference & learning from data**

- [time-series-mathematics](https://github.com/TJT-Pro/time-series-mathematics) -- ACF/PACF, AR/MA/ARMA, Yule-Walker, ADF stationarity testing
- [bayesian-inference](https://github.com/TJT-Pro/bayesian-inference) -- conjugate priors, grid approximation, Metropolis-Hastings MCMC
- [hidden-markov-models](https://github.com/TJT-Pro/hidden-markov-models) -- forward-backward algorithm, Viterbi, Baum-Welch
- [information-theory](https://github.com/TJT-Pro/information-theory) -- entropy, KL divergence, mutual information, Huffman coding
- [svd-from-scratch](https://github.com/TJT-Pro/svd-from-scratch) -- singular value decomposition, pseudoinverse, condition numbers

**Quant finance engineering**

- [convex-optimization](https://github.com/TJT-Pro/convex-optimization) -- Newton's method, projected gradient descent, KKT/duality
- [portfolio-mathematics](https://github.com/TJT-Pro/portfolio-mathematics) -- Markowitz frontier, CAPM, Kelly criterion
- [model-risk](https://github.com/TJT-Pro/model-risk) -- VaR backtesting, AIC/BIC, backtest overfitting demonstration (the capstone)

## Why from scratch

Most quantitative code hides its mathematics behind a library call. That's
the right call in production. It's the wrong call if the goal is to
understand what the library is actually computing. The goal was never to
replace NumPy or SciPy -- it was to earn the right to use them.

ENJOY! (●'◡'●)
