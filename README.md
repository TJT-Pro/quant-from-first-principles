# hidden-markov-models

A Gaussian Hidden Markov Model built from scratch: the scaled
forward-backward algorithm, Viterbi decoding, and Baum-Welch (EM)
parameter estimation, applied to uncover latent volatility regimes in
real EUR/USD data.

This repository is independent of the sibling repositories and reuses
none of their code.

## Project structure

```
hidden-markov-models/
    README.md
    data/DEXUSEU_returns.csv
    hmm.py               GaussianHMM: forward, backward, Viterbi, Baum-Welch
    model_selection.py   AIC/BIC selection of the number of hidden states
    fx_application.py    regime detection applied to real EUR/USD data
    test_*.py
    cli.py
```

## Status

45 tests, 10/10 verify checks, all passing.

## Setup

Python 3.11+. All inference code uses only the standard library.
`cli.py verify` uses numpy and scipy as independent oracles.

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest test_hmm.py test_model_selection.py test_fx_application.py -v
```

Note: several tests fit real HMMs via EM (with multiple random
restarts) and take a noticeable fraction of a minute in total; none
are slow by accident.

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

### The model

A hidden Markov model has an unobserved state sequence
$S_1,\dots,S_T \in \{0,\dots,K-1\}$ following a first-order Markov
chain (transition matrix $A$, $A_{ij}=P(S_t=j\mid S_{t-1}=i)$, initial
distribution $\pi$), and an observed sequence $X_1,\dots,X_T$ where
$X_t$ depends only on $S_t$: here, $X_t \mid S_t{=}j \sim N(\mu_j,
\sigma_j^2)$. The states are never observed directly; everything below
is inferred purely from $X_1,\dots,X_T$.

### The forward algorithm, and why it needs scaling

The forward variable $\alpha_t(j) = P(X_1,\dots,X_t, S_t=j)$ satisfies
the recursion $\alpha_t(j) = \left[\sum_i \alpha_{t-1}(i) A_{ij}\right]
B_j(X_t)$, a sum of products of probabilities that decays
geometrically with $t$: computed directly, it underflows a float
within a few hundred observations for any realistic problem.
Rabiner's (1989) scaling trick renormalises $\alpha_t$ to sum to 1 at
every $t$, dividing by $c_t = \sum_j \alpha_t(j)$ computed *before* the
unscaled recursion would have underflowed (since it operates on the
previous step's already-renormalised values); the sequence
log-likelihood is then recovered exactly as $\sum_t \ln c_t$, since
each $c_t$ is precisely the probability mass the unscaled recursion
would have carried forward. This is checked directly against a
brute-force enumeration of every possible state sequence for a short
6-observation sequence, not merely trusted: `verify` sums the exact
joint probability of all $2^6$ paths and confirms it matches the
scaled forward algorithm's log-likelihood to $10^{-8}$.

### The backward algorithm, and combining the two

$\beta_t(i) = P(X_{t+1},\dots,X_T \mid S_t=i)$, rescaled at every step
by the *same* $c_t$ the forward pass computed (not a separately
derived backward scaling factor), which is exactly what makes
$\alpha_t(i)\beta_t(i)$ proportional to the true state posterior
$P(S_t=i\mid X_{1:T})$ once renormalised across $i$ (`gamma`). The
pairwise posterior $P(S_t{=}i, S_{t+1}{=}j \mid X_{1:T})$ (`xi`) follows
the same combination one step further. A direct consistency identity
between the two, $\sum_j \xi_t(i,j) = \gamma_t(i)$, is verified for
every $t$ and every $i$ in the test suite: since $\xi_t$ tracks a joint
distribution over $(S_t, S_{t+1})$ and $\gamma_t$ tracks the marginal
over $S_t$ alone, this identity has to hold by the basic definition of
a marginal, and checking it directly catches implementation bugs a
looser end-to-end test could miss.

### Viterbi: the single best path, not the marginal posteriors

$\gamma_t$ answers "what is the most likely state at time $t$,
considered on its own"; Viterbi answers a different question, "what is
the single most likely *entire sequence* of states," which need not
even agree with stringing together each $\gamma_t$'s individual
maximiser. The recursion $\delta_t(j) = \max_i[\delta_{t-1}(i) + \ln
A_{ij}] + \ln B_j(X_t)$ works in log-space, tracking a backpointer at
each step; taking a running max, unlike a running sum, is exact under
an additive shift, so no scaling-and-rescaling machinery is needed
here the way it is for the forward algorithm. Checked directly against
brute-force enumeration on the same short synthetic sequence used for
the forward-algorithm check, and separately checked to recover true
generating states with over 98% accuracy on longer, well-separated
synthetic data.

### Baum-Welch: EM for HMMs

Each iteration is an E-step (compute $\gamma$ and $\xi$ from the
current parameters via forward-backward) and an M-step (re-estimate
every parameter as a $\gamma$/$\xi$-weighted maximum-likelihood
update: $\pi_j \leftarrow \gamma_1(j)$,
$A_{ij} \leftarrow \sum_t\xi_t(i,j) / \sum_t\gamma_t(i)$,
$\mu_j \leftarrow \sum_t\gamma_t(j)X_t / \sum_t\gamma_t(j)$, and the
analogous weighted-variance update), with $\gamma_t(j)$ standing in
for "how much of observation $t$ belongs to state $j$" in place of a
hard, observed assignment. **EM's defining theoretical guarantee, that
the observed-data log-likelihood never decreases from one iteration to
the next, is checked directly** on every fit in the test suite and in
`verify`, not merely cited: the full log-likelihood trajectory is
returned by `fit` specifically so this can be verified rather than
assumed.

**Local optima are a genuine, structural feature of this problem, not
an edge case.** Baum-Welch converges to *a* local maximum of the
likelihood, and which one depends on the (essentially arbitrary)
random initialisation. `model_selection.fit_best_of_n_restarts`
mitigates this the standard way: run EM to convergence from several
random starting points and keep whichever reaches the highest
log-likelihood. `random_init` seeds each state's mean at a distinct,
randomly chosen observation (rather than all states starting at the
same sample mean, which would make every state initially identical
and unable to differentiate during EM at all) and gives the initial
transition matrix a mild, randomly perturbed self-transition
preference.

### Choosing the number of hidden states

A $K$-state Gaussian HMM has $(K-1) + K(K-1) + K + K = K^2+2K-1$ free
parameters: $K-1$ for the initial distribution, $K(K-1)$ for the
transition matrix (each of $K$ rows independently sums to 1, so
doubling $K$ roughly quadruples this term alone), and $K$ each for the
means and variances. Because a $K$-state HMM can always mimic a
$(K-1)$-state one (by making one state a near-duplicate of another),
the achievable log-likelihood is non-decreasing in $K$ by construction,
so raw likelihood comparison would trivially favour the largest $K$
tried; AIC ($2k - 2\ln\hat L$) and BIC ($k\ln n - 2\ln\hat L$) penalise
the added parameters and are what make the comparison meaningful.
Verified directly: on synthetic 2-state data, both criteria correctly
select $K=2$ over $K=1$ and $K=3$.

## Application: how many volatility regimes does EUR/USD actually have?

**The naive 2-state fit finds exactly the split you'd expect.**
Fitting a 2-state Gaussian HMM to the full 6,930-observation return
series separates a low-volatility regime (daily sd 0.38%, annualised
6.1%, occupying 44.8% of the decoded path) from a high-volatility
regime (daily sd 0.70%, annualised 11.2%, occupying 55.2%), a
substantial 1.84x volatility ratio. Both regimes are highly persistent
(self-transition probabilities above 0.99, model-implied expected
durations of 118 and 129 days), which is what makes them genuine
*regimes* rather than day-to-day noise reclassification.

**But AIC and BIC both say the real structure needs three states, not
two, and the third state is genuinely interesting.** Running a formal
model comparison across $K \in \{1,2,3\}$: log-likelihoods of
25,916.2, 26,370.3, and 26,451.9, giving AIC of $-51{,}828$,
$-52{,}727$, and $-52{,}876$ and BIC of $-51{,}815$, $-52{,}679$, and
$-52{,}780$. Both criteria prefer $K=3$ at every step, meaning the
extra parameters earn their keep even after the $K\ln n$ penalty. The
resulting three regimes are not simply "low/medium/high" versions of
the same thing:

| regime | daily sd (annualised) | mean daily return | stationary probability | model-implied duration |
|---|---|---|---|---|
| calm | 0.37% (5.8%) | +0.000066 | 40.2% | 201 days |
| normal | 0.60% (9.6%) | -0.000002 | 51.4% | 94 days |
| stress | 0.99% (15.8%) | **-0.000297** | **8.5%** | **24 days** |

The third regime is not just "higher volatility": it is **rare**
(occupying only 4.7% of the decoded path), **short-lived** (24-day
model-implied duration, versus 94-201 days for the other two), and
carries a **real negative drift** roughly 4-5 times larger in magnitude
than either other regime's mean return, all three properties pointing
at genuine, distinct market-stress episodes (the kind driven by
specific events) rather than a smooth continuation of "somewhat more
volatile than normal." A naive 2-state model, chosen without running
AIC/BIC at all, would have missed this entirely by folding the stress
episodes into the same bucket as ordinary elevated volatility.

**A real, specific discrepancy worth reporting rather than glossing
over.** The 2-state model's *decoded* regime episodes last far longer
on average (259 and 318.5 days) than the *fitted model's own*
geometric-duration implication (118 and 129 days respectively) would
predict, and the 3-state model shows the same pattern (decoded
episodes of 318, 338, and 81 days against model-implied durations of
201, 94, and 24). This is not a bug: a first-order Markov chain's
regime-duration distribution is necessarily geometric (memoryless), so
the *expected* duration is a correctly computed property of the fitted
transition matrix, but the realised path's actual episode lengths
reflect the full geometric distribution's variance, not just its mean,
and with only 8-12 realised episodes in the whole 27-year sample, the
empirical mean episode length is a noisy estimate of that geometric
mean rather than a contradiction of it. It is also the specific,
well-documented critique of basic Markov-switching volatility models
in the literature: real regime persistence often has fatter tails than
a geometric distribution allows (a regime that has already lasted a
year is, empirically, often more likely to continue than a
memoryless model would predict), which is exactly the direction this
project's own numbers point.

## Limitations

- `fit_best_of_n_restarts` mitigates but does not eliminate the local-
  optimum problem: a small, fixed number of restarts (3-5 throughout
  this project) is not a guarantee of finding the global maximum-
  likelihood fit, only a practical improvement over a single run.
- The `expected_duration_days` figures are the exact expectation of a
  geometric distribution implied by the fitted self-transition
  probability; as the application section discusses, the realised
  path's episode lengths are a small, noisy sample from that
  distribution (single digits to low teens of episodes over the whole
  27-year history), not a large enough sample to distinguish "the
  geometric model is a good fit with high variance" from "the true
  duration distribution is not geometric at all" with any real
  statistical power.
- The label-switching handling here (sorting states by fitted
  variance) is a simple, order-based convention that works cleanly
  whenever the fitted states are well separated in volatility, which
  they are in every case examined in this project; it would need a
  more careful matching procedure (e.g. by full Kullback-Leibler
  distance between fitted and reference emission distributions) for a
  problem where two states could plausibly have similar variances but
  differ in mean instead.

## CLI examples

```bash
# Simulate from a known Gaussian HMM
python3 cli.py simulate --means "0,5" --variances "1,1" --transition "0.9,0.1;0.1,0.9" --length 20

# Simulate then fit via Baum-Welch, check state recovery
python3 cli.py fit --means "0,5" --variances "1,1" --transition "0.9,0.1;0.1,0.9" --length 2000

# AIC/BIC selection of the number of hidden states on real EUR/USD data
python3 cli.py selectstates --k-range "1,2,3" --restarts 3

# Fit and summarise EUR/USD volatility regimes (2 or 3 states)
python3 cli.py fxregimes --n-states 3 --restarts 3

# Cross-check every algorithm against scipy oracles and brute-force
# enumeration over all possible state sequences
python3 cli.py verify
```
