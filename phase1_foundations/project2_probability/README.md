# Project 2 - Probability Simulator & Bayes Engine

Simulates probability experiments from scratch and implements Bayes'
theorem.  The only thing borrowed from a library is the raw uniform draw;
everything on top of it, the sampling and the probability logic, is coded
by hand.

| Component | What it does | Uses a library for the math? |
|---|---|---|
| Coin and dice simulator | flips and rolls, empirical vs theoretical probability | No |
| Conditional probability | marginals and P(A|B) from a joint table | No |
| Bayes engine | posterior from prior and likelihood | No |
| Distribution samplers | Binomial, Geometric, Poisson from uniform draws | No |
| Law of large numbers | running means, convergence plots | No |

Plots are drawn with matplotlib, but only in the plotting functions and
only when you ask for a plot.  The core code runs on the standard library
alone.

---

## The math

### Empirical vs theoretical probability

Run an experiment `n` times.  The empirical probability of an event is
the share of runs where it happened, `hits / n`.  By the law of large
numbers this converges to the true probability as `n` grows.  For a fair
coin `P(H) = 1/2`; for a fair `s`-sided die each face has `1/s`.

### Conditional probability from a joint table

A joint table gives `P(X=x, Y=y)` for every pair.  From it:

    P(X=x)  = sum over y of P(X=x, Y=y)
    P(X=x | Y=y)  =  P(X=x, Y=y) / P(Y=y)

with `P(Y=y) = sum over x of P(X=x, Y=y)`.  The conditional is undefined
when `P(Y=y) = 0`.  The sampler draws `(x, y)` pairs from the table by
cumulative inversion of a uniform draw, so the empirical conditional can
be checked against the computed one.

### Bayes' theorem

For hypotheses `h` and evidence `E`:

    P(H=h | E)  =  P(E | H=h) * P(H=h) / P(E)
    P(E)        =  sum over h of P(E | H=h) * P(H=h)

The engine takes a prior `P(H)` and a likelihood `P(E | H)` and returns
the posterior.  The classic base-rate problem is built in: 1% prevalence,
99% sensitivity, 5% false-positive rate.  A positive test gives only
`0.99 * 0.01 / (0.99 * 0.01 + 0.05 * 0.99) = 1/6`, about 16.7%.  The
intuition that it should be near 99% is exactly the base-rate fallacy the
exercise exists to expose.

### The three distributions

| Distribution | Definition | Sampled how | Mean |
|---|---|---|---|
| Binomial(n, p) | successes in n independent trials | sum of n Bernoulli draws | n*p |
| Geometric(p) | trials until first success, support 1, 2, ... | keep drawing until a draw falls below p | 1/p |
| Poisson(lambda) | count of rare events, support 0, 1, ... | Knuth's exact method on uniforms | lambda |

Knuth's method for Poisson: with `L = exp(-lambda)`, multiply uniform
draws into a running product until it drops below `L`; the number of draws
used, minus one, is the sample.  No library distribution code is involved.

The theoretical PMFs are implemented from their formulas so the empirical
distribution of any sample can be compared against what the math predicts:

    Binomial:   C(n,k) p^k (1-p)^(n-k)
    Geometric:  (1-p)^(k-1) p
    Poisson:    exp(-lambda) lambda^k / k!

### Law of large numbers

The running mean of a sample converges to the expected value as `n` grows:
for dice rolls the mean of a fair die is `(sides + 1) / 2` (3.5 for a
six-sided die), for coin flips coded as 1/0 it is 0.5.  The plots show
the running average approaching the theoretical line.

---

## Files

- `probability_simulator.py` - all implementations
- `cli.py` - command-line interface
- `test_probability_simulator.py` - the tests
- `samples/joint_rain_umbrella.json` - example joint table
- `plots/` - generated PNGs (created by the plot commands)

## Setup

Python 3.11.  The core needs only the standard library.  The plotting
commands additionally want matplotlib:

```bash
pip install --break-system-packages matplotlib
```

## Usage

Run from inside the project folder:

```bash
python3 cli.py flip 20000 --seed 1
python3 cli.py roll 30000 --sides 6 --face 3 --seed 1

python3 cli.py joint samples/joint_rain_umbrella.json
python3 cli.py cond samples/joint_rain_umbrella.json rain no --seed 1

python3 cli.py bayes --prior 0.01 0.99 --likelihood 0.99 0.05 --labels disease healthy
python3 cli.py bayes-medical

python3 cli.py sample binomial --n 10 --p 0.3 --size 30000 --seed 1
python3 cli.py sample geometric --p 0.25 --size 30000 --seed 1
python3 cli.py sample poisson --lambda 2.5 --size 30000 --seed 1

python3 cli.py lln coin --seed 1
python3 cli.py lln dice --seed 1

python3 cli.py sample binomial --n 10 --p 0.3 --plot plots/binomial.png --seed 1
```

The `--seed` flag makes every random run reproducible.  Plot commands
print the path of the saved PNG.  The joint table format is a JSON object
whose keys are `(X, Y)` pairs written as `"[x, y]"` strings.

## Tests

```bash
python3 -m unittest -v
```

37 tests cover: coin and dice bounds and convergence, marginal and
conditional values on a known table (including the zero-probability
error), the simulation-vs-computed check, every Bayes case including the
1/6 medical result and the error paths, sampler means and support, PMF
sums to one, and a fixed-seed reproducibility check.

## Self-verification

```bash
python3 cli.py verify
```

16 checks.  Deterministic because every run uses a fixed seed.  Empirical
statistics are compared against the theory (P(H) near 0.5, P(3) near 1/6,
sample means near their expected values, PMFs summing to one, the medical
posterior exactly 1/6).

## Choices worth knowing about

- Uniform draws come from `random.random()`; everything else is derived
  here, so the whole thing could be ported to any language in an
  afternoon.
- Every randomized function takes an optional `rng` argument, which is
  what makes the tests and the verify command reproducible.
- All PMFs and posterior computations use exact arithmetic where possible;
  the only floats are the probabilities themselves.
- The joint table key order is always `(X, Y)`: first key is the row
  variable, second the column variable.
