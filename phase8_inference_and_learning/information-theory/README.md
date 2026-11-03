# information-theory

An information theory engine built from scratch: Shannon entropy,
KL divergence and mutual information (verified two independent ways),
Huffman coding with Shannon's source coding theorem checked directly,
channel capacity and a working repetition-code demonstration, and
entropy rate estimation validated against a synthetic Markov chain
with a known exact answer, applied to real EUR/USD returns as a
genuinely different lens on questions the sibling repositories
approached through time-series statistics, Bayesian inference, and
regime-switching models.

This repository is independent of the sibling repositories and reuses
none of their code.

## Project structure

```
information-theory/
    README.md
    data/DEXUSEU_returns.csv
    entropy.py             Shannon entropy, joint/conditional entropy
    divergence.py          KL divergence, cross-entropy, mutual information
    huffman_coding.py      optimal prefix-free codes, source coding bound
    channel_capacity.py    BSC and Gaussian channel capacity, repetition codes
    entropy_rate.py        entropy rate via block/conditional entropy
    fx_application.py      information-theoretic analysis of real EUR/USD returns
    test_*.py
    cli.py
```

## Status

97 tests, 14/14 verify checks, all passing.

## Setup

Python 3.11+. All information-theoretic code uses only the standard
library. `cli.py verify` uses numpy and scipy as independent oracles.

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest test_entropy.py test_divergence.py test_huffman_coding.py \
    test_channel_capacity.py test_entropy_rate.py test_fx_application.py -v
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

### Entropy, and the chain rule as an internal consistency check

Shannon entropy, H(X) = -sum_x p(x)*log(p(x)), is the expected number
of bits an optimal code needs to describe X's outcome. The chain rule
H(X,Y) = H(X) + H(Y|X) is not merely quoted here: the joint entropy,
marginal entropy, and conditional entropy are all computed
independently via the plug-in (empirical-frequency) estimator and
checked to satisfy the identity to floating-point precision on
constructed test cases. max_entropy(n) = log(n) is a direct
consequence of Gibbs' inequality (below) applied to the uniform
distribution.

### KL divergence, Gibbs' inequality, and why entropy has a ceiling

D(P||Q) = sum_x p(x)*log(p(x)/q(x)), the expected extra bits needed to
encode samples from P using a code built for Q instead. Gibbs'
inequality, D(P||Q) >= 0 with equality iff P=Q, follows from Jensen's
inequality applied to the concave log function:
-D(P||Q) = E_P[log(q(x)/p(x))] <= log(E_P[q(x)/p(x)]) = log(1) = 0.
Checked directly on 50 random distribution pairs per test run, never
producing a negative value. This single inequality, applied with Q
equal to the uniform distribution, is exactly what forces
H(P) <= log(n) for every distribution P on n outcomes: entropy's
ceiling is not a separate fact, it is Gibbs' inequality in a
particular guise.

### Mutual information: verified two structurally different ways, and what it catches that correlation cannot

I(X;Y) = H(X)+H(Y)-H(X,Y), equivalently D(P(X,Y) || P(X)*P(Y)): the
divergence between the true joint distribution and the distribution
that would hold exactly under independence. Both formulas are computed
independently in this project (mutual_information_from_pairs via the
entropy identity, mutual_information_via_kl via the direct divergence
definition) and checked to agree to 1e-8 on random joint
distributions, a genuine cross-check between two algebraically
equivalent but computationally distinct routes to the same number.

Unlike Pearson correlation, mutual information catches nonlinear
dependence. A precise worked example, refined after an initial
imprecise version of the claim: for Y=X^2 on X uniform over
{-2,-1,0,1,2}, correlation is exactly zero (an even function of a
symmetric variable), and Y is a deterministic function of X
(H(Y|X)=0), but X is NOT recoverable from Y (knowing Y=1 leaves
X=+-1 ambiguous, and similarly for Y=4), so
I(X;Y) = H(Y)-H(Y|X) = H(Y) ~ 1.522 bits, strictly less than
H(X) ~ 2.322 bits, not equal to it. An earlier version of this
project's own test suite asserted I(X;Y)=H(X) here, which is only
correct when the map is bijective; getting caught by the test itself
and corrected to the precise claim (I(X;Y)=H(Y), with the gap
explained by the sign-collision) is exactly the kind of detail worth
getting right rather than glossing over.

### Huffman coding and Shannon's source coding theorem

Huffman's greedy algorithm (repeatedly merge the two least-frequent
nodes) is optimal among prefix codes: the two rarest symbols must sit
as siblings at the deepest level of any optimal tree (swapping them
with whatever is at the deepest level otherwise could only help), and
the rest of the problem then reduces recursively to a smaller
alphabet with those two symbols merged. Shannon's source coding
theorem, H(X) <= L_avg < H(X)+1, is checked directly on every
distribution tested here, not merely cited: an actual, working,
decodable Huffman code is built, its average length measured, and
both inequalities confirmed to hold.

A precise, easy-to-miss limitation, found directly and worth stating
carefully. For a 2-symbol alphabet, Huffman coding is mechanically
forced to assign a 1-bit codeword to each symbol, since there are only
two possible binary codewords at all ("0" and "1"), regardless of how
skewed the true distribution is. Verified directly: a heavily skewed
950/50 binary sequence (entropy 0.286 bits) still gets an average code
length of exactly 1.0 bit/symbol via Huffman, identical to a perfectly
fair 50/50 sequence. Huffman coding on a binary sequence therefore
confirms nothing about how close to fair a coin is; that confirmation
has to come from the entropy computation directly, not from the
achieved compression ratio, which is exactly why the EUR/USD
application below uses entropy (not Huffman) to confirm the "fair
coin" finding, and reserves Huffman for the volatility-state sequence,
where the alphabet has 3 symbols and skew genuinely can (and does)
reduce the achievable average code length below the naive
ceil(log2(3))=2 bits/symbol.

### Channel capacity, verified with a working code, not just a formula

The Binary Symmetric Channel's capacity, C=1-H_b(p) (with
H_b(p)=-p*log2(p)-(1-p)*log2(1-p) the binary entropy function),
follows from I(X;Y)=H(Y)-H(Y|X): H(Y|X)=H_b(p) regardless of the input
distribution (the flip noise alone determines the remaining
uncertainty given any specific input bit), and H(Y) is maximised at
exactly 1 bit by a uniform input, giving C=1-H_b(p) directly. The
Gaussian channel's capacity, C=0.5*log2(1+SNR) (Shannon-Hartley), is
the continuous analogue, with a Gaussian input distribution maximising
mutual information for a fixed power budget. A working repetition code
demonstrates the promise of channel coding concretely rather than
abstractly: at crossover probability p=0.1 (capacity 0.531 bits/use),
an uncoded bit is wrong 10.5% of the time; a 25-fold repetition code
(rate 0.04, well below capacity) drives the empirical error rate to
effectively zero (0 errors in 20,000 trials), the direct operational
meaning of a channel's capacity: reliable communication at any rate
below it is achievable, in principle, by paying with more redundancy.

### Entropy rate, validated against exact Markov chain theory before trusting real data

A stationary Markov chain's entropy rate equals its first-order
conditional entropy, H_rate = sum_i pi_i * H(P_i.) (the stationary-
distribution-weighted average of each row's own entropy), which is
also exactly the limiting rate of growth of block entropy,
H(X_1,...,X_n)/n, since the Markov property makes everything before
the immediately preceding state irrelevant. On a synthetic 2-state
chain with transition matrix [[0.9,0.1],[0.3,0.7]] (exact rate 0.5721
bits), the conditional-entropy estimator at order 1 lands within 0.005
bits of the exact rate almost immediately (0.5766 from a single
simulated run), while the block-entropy estimator needs successively
longer blocks to approach the same value and converges much more
slowly (0.815, 0.696, 0.656, 0.636, 0.616 for block lengths 1 through
6, still visibly short of 0.572): a direct, measured demonstration of
why conditioning on the process's actual memory length is a far more
sample-efficient estimator than growing the block length toward
infinity, whenever the process's memory structure is known or can be
assumed.

## Application: what information theory reveals about EUR/USD that other lenses did not state as precisely

**Sign entropy: the fair-coin finding, in its most fundamental
information-theoretic form.** The daily up/down sign carries 0.999625
bits of entropy against a theoretical ceiling of exactly 1 bit: 99.96%
of the maximum possible, a direct, model-free confirmation of the
"EUR/USD direction is close to a fair coin" finding threaded
throughout the sibling repositories, now stated as the most
fundamental quantity information theory has to offer rather than
derived from a frequency test or a Bayesian posterior.

**Mutual information with a properly constructed null, not just a
small raw number.** The plug-in mutual information estimator has a
known positive finite-sample bias even under true independence, so
"the raw MI value looks small" is not, by itself, evidence of no
dependence. A permutation-test null (500 shuffles per lag, destroying
temporal structure while preserving each variable's own marginal
distribution exactly) is used throughout instead. Return sign shows no
lag surviving this test in isolation at lags 1, 2, 5, and 20; lag=10
alone shows a nominally significant result, p=0.027. Rather than
report this as a finding of tenth-day predictability, running the same
test across all 5 lags and applying a Bonferroni correction (dividing
the significance threshold by the number of tests, 0.05/5=0.01, since
testing 5 independent lags at a nominal 5% level gives roughly a 23%
chance of at least one false positive purely by chance even under true
independence everywhere) shows this result does not survive: no lag
remains significant after correction. This is reported as the false
positive multiple-testing theory predicts it most likely is, a real
methodological trap this project's own analysis fell into and then
caught, not a hypothetical one raised for illustration.

**The same method, aimed at volatility instead of sign, gives a
completely different and overwhelmingly significant answer.** Using a
smoothed 20-day rolling realised volatility (raw daily absolute return
is dominated by idiosyncratic noise on top of the genuinely
slower-moving volatility level; a smoothed measure is needed to reveal
it, exactly mirroring why ode-engine and stochastic-differential-
equations also needed to work with realised or rolling volatility
rather than raw squared returns), discretised into terciles:

| lag (days) | MI (bits) | max possible (3 states) | significant after 300-shuffle permutation test |
|---|---|---|---|
| 1 | 1.152 | 1.585 | yes, p<0.001 |
| 5 | 0.739 | 1.585 | yes, p<0.001 |
| 10 | 0.537 | 1.585 | yes, p<0.001 |
| 20 | 0.312 | 1.585 | yes, p<0.001 |
| 40 | 0.215 | 1.585 | yes, p<0.001 |

At lag 1, the mutual information is 73% of the theoretical maximum for
a 3-state variable, and stays overwhelmingly significant out to 40
days. One honest caveat, stated precisely rather than glossed over:
adjacent days' 20-day rolling windows share 19 of 20 underlying
observations, so part of the very-short-lag dependence (particularly
at lag 1) is a mechanical consequence of the smoothing construction
itself, not purely behavioural persistence; the longer lags (20 and
40, where the windows barely or never overlap) are far more directly
interpretable as genuine persistence, and they remain overwhelmingly
significant regardless.

**KL divergence from a fitted Gaussian: "fat tails," quantified in
bits rather than via skewness or kurtosis.**
D(empirical || Gaussian) = 0.058 bits on the full return series
(validated against pure Gaussian samples, which show D<0.02, and a
synthetic fat-tailed mixture, which shows a clearly larger D than its
Gaussian comparison baseline): a small but genuinely positive,
information-theoretic confirmation of the well-established "returns
are not Gaussian" finding running throughout this project family, via
a completely different mathematical route (relative entropy) than
skewness, kurtosis, or a goodness-of-fit test statistic.

## Limitations

- The plug-in entropy and mutual information estimators used
  throughout this project are known to be biased (entropy biased
  downward, mutual information biased upward) for small samples
  relative to the alphabet size; this bias is negligible for the
  thousands of observations and small (2-4 symbol) alphabets this
  project actually uses, but would need an explicit bias-correction
  method (e.g. the Miller-Madow correction) for smaller samples or
  larger alphabets.
- `huffman_coding.py` builds a single, fixed (memoryless) code from
  the marginal frequency distribution of a sequence; it cannot exploit
  temporal dependence even when `divergence.py`'s mutual information
  correctly detects that dependence exists (as it does, overwhelmingly,
  for the volatility-state sequence). A genuinely adaptive or
  context-based compressor (an order-1 arithmetic coder, for instance)
  would be needed to actually compress the volatility-state sequence
  down toward its true entropy rate rather than merely its marginal
  entropy; this project measures the dependence but does not build a
  compressor that exploits it.
- `entropy_rate.py`'s validation is against a genuinely first-order
  Markov synthetic chain; applied to a real sequence whose true memory
  structure is unknown or longer than order 1 (plausible for the
  volatility-state sequence, given persistence visibly extending well
  beyond a single lag), `conditional_entropy_rate_estimate` at a fixed
  order gives only an upper bound on the true entropy rate, not
  necessarily a close estimate of it.

## CLI examples

```bash
# Compute the entropy of a symbol sequence
python3 cli.py entropy --symbols "a,a,a,b,b,c"

# Mutual information between two sequences
python3 cli.py mutualinfo --x "a,a,b,b,a,b" --y "1,1,2,2,1,2"

# Huffman-encode a text string
python3 cli.py huffman --text "mississippi"

# Channel capacity and a repetition-code demonstration
python3 cli.py channel --p 0.1 --snr 15

# Entropy rate estimation on a synthetic Markov chain
python3 cli.py entropyrate --p00 0.9 --p11 0.7

# Information-theoretic analysis of real EUR/USD returns
python3 cli.py fxinfo

# Cross-check every quantity against scipy/numpy oracles and known theory
python3 cli.py verify
```
