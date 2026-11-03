# automatic-differentiation

An automatic differentiation engine built from scratch, two ways:
forward mode via dual numbers, and reverse mode via a computational
graph with backpropagation. Both are cross-checked against each other,
against finite differences, and against a from-scratch neural network
built entirely on the reverse-mode engine, then applied to a genuine
(and genuinely mixed) EUR/USD volatility-forecasting exercise.

This repository is independent of `math-implementations`,
`stochastic-processes`, and `black-scholes`, and reuses none of their
code, though the application below extends the volatility-clustering
theme those repositories established.

## Project structure

```
automatic-differentiation/
    README.md
    data/DEXUSEU_returns.csv
    dual_numbers.py           forward-mode AD via dual numbers, Jacobians
    autodiff_engine.py        reverse-mode AD: computational graph, backprop
    jacobian_comparison.py    forward vs reverse vs numerical, cost model
    neural_network.py         tiny MLP built on the reverse-mode engine
    test_dual_numbers.py
    test_autodiff_engine.py
    test_jacobian_comparison.py
    test_neural_network.py
    cli.py
```

## Status

70 tests, 11/11 verify checks, all passing.

## Setup

Python 3.11+. All differentiation and training code uses only the
standard library. `cli.py verify` uses numpy for one independent
finite-difference cross-check.

```bash
pip install --break-system-packages numpy
```

## Running tests

```bash
python3 -m pytest test_dual_numbers.py test_autodiff_engine.py \
    test_jacobian_comparison.py test_neural_network.py -v
```

## Running the oracle

```bash
python3 cli.py verify
```

## The data

`data/DEXUSEU_returns.csv`: daily EUR/USD exchange rates from the
Federal Reserve (FRED series DEXUSEU), 1999-01-04 to 2026-08-21
(6,931 price levels, 6,930 daily returns), the same file used in
`stochastic-processes` and `black-scholes`.

## Mathematical background

### Forward mode: dual numbers

A dual number $a + b\varepsilon$ adjoins a formal symbol $\varepsilon$
to the reals, defined by $\varepsilon^2 = 0$ (the same construction
that adjoins $i$ with $i^2=-1$ to form the complex numbers, with a
different defining relation). Evaluating an ordinary function $f$ at
$x + \varepsilon$ using dual arithmetic gives, by Taylor's theorem,

$$f(x + \varepsilon) = f(x) + f'(x)\varepsilon + \frac{f''(x)}{2}\varepsilon^2 + \cdots = f(x) + f'(x)\varepsilon,$$

since every term from the second onward vanishes exactly (not
approximately) because $\varepsilon^2=0$. The derivative falls out of
ordinary function evaluation carried out in this algebra; no symbolic
differentiation and no finite-difference approximation is involved
anywhere. `Dual.__mul__` and `Dual.__truediv__` derive the product and
quotient rules directly from this relation (each docstring works
through the algebra), and every elementary function (`sin`, `cos`,
`exp`, `log`, `sqrt`, `tanh`) applies the corresponding derivative
rule to the dual part by the chain rule.

**Extending to a gradient or a full Jacobian** costs one evaluation of
$f$ per **input** dimension: `gradient_forward_mode` seeds each input
direction's $\varepsilon$-component to 1 (and every other input's to
0) in turn, and `jacobian_forward_mode` does the same to fill in the
Jacobian one column at a time. This cost structure, linear in the
number of inputs, regardless of how many outputs $f$ has, is the
entire reason forward mode is well suited to functions with few
inputs and poorly suited to ones with many, which section
"Comparing the two modes" below makes concrete.

### Reverse mode: a computational graph and backpropagation

`autodiff_engine.Value` wraps a scalar and, every time it is used in
an operation, records the operation's inputs and a closure describing
how to route a gradient backward through that specific operation (the
local derivative, e.g. multiplication's `_backward` pushes `other.data
* out.grad` onto `self.grad`, directly implementing the product rule
one edge at a time). This builds a computational graph forward as the
expression is evaluated. Calling `.backward()` on the final output
then:

1. Topologically sorts the graph (every node appears after all of its
   children), so no node is processed before every path from it to
   the root has finished contributing.
2. Walks that order in reverse starting from `self.grad = 1.0`,
   calling each node's `_backward` closure, which is exactly the
   multivariable chain rule applied one local step at a time:
   $\frac{\partial L}{\partial v} = \sum_{\text{children } c} \frac{\partial L}{\partial c}\cdot\frac{\partial c}{\partial v}$,
   accumulated (`+=`, not overwritten) since a node can feed into
   multiple downstream computations and its total gradient is the sum
   over every path.

This single backward pass computes the gradient of the output with
respect to **every** node in the graph, including every input,
regardless of how many inputs there are. That is reverse mode's
defining cost structure: one pass per **output**, not per input,
which is why it dominates for a scalar loss function of many
parameters (exactly a neural network's situation) and why forward mode
dominates for the opposite shape.

**A real limitation, found by testing at realistic scale, not assumed
away.** The topological sort was initially written as the natural
recursive DFS. It worked on every unit test (small, hand-built
expressions) but crashed with `RecursionError: maximum recursion depth
exceeded` the first time it was used inside `neural_network.train` on
a full-batch loss built from 5,540 real EUR/USD training examples,
because Python's default recursion limit (1,000 stack frames) is far
smaller than the depth of a graph built by folding thousands of
training examples into one expression. Raising the recursion limit
would have papered over the symptom without fixing the underlying
constraint (the limit would just move to some other, larger dataset).
The actual fix, implemented in `Value.backward`, replaces the
recursive DFS with an explicit, stack-based iterative one: the same
algorithm, but using a heap-allocated Python list as the traversal
stack instead of the interpreter's call stack, which is bounded only
by available memory rather than a fixed frame count. This is a
standard, well-known technique for exactly this class of problem
(any depth-first traversal of a graph whose depth cannot be bounded in
advance), and the fix is verified by `test_neural_network.py`
succeeding on the real full-batch training loop where the recursive
version could not.

### Comparing the two modes directly

`jacobian_comparison.compare_costs(n_inputs, n_outputs)` states the
asymmetry as a number rather than a slogan: forward mode costs
`n_inputs` passes, reverse mode costs `n_outputs` passes. For a
function $\mathbb{R}^1 \to \mathbb{R}^{1000}$, forward mode needs a
single pass and reverse mode needs a thousand; for
$\mathbb{R}^{1{,}000{,}000} \to \mathbb{R}^1$ (a huge parameter vector
collapsing to a single scalar loss, the exact shape of a neural
network's training objective), reverse mode needs a single backward
pass and forward mode would need a million forward passes. This is
why every modern deep learning framework is built on reverse-mode
automatic differentiation, and the comparison is checked directly,
not just asserted, by computing the same Jacobian all three ways
(forward, reverse, and central-difference numerical differentiation)
on shared test functions and confirming all three agree.

### A tiny neural network on the reverse-mode engine

`neural_network.py` builds `Neuron`, `Layer`, and `MLP` classes
entirely out of `Value` objects: every weight and bias is a `Value`,
so the forward pass through the whole network is automatically
recorded onto the computational graph, and a single `.backward()` call
on the network's loss populates every parameter's gradient via the
chain rule with no additional bookkeeping. Training is full-batch
gradient descent: zero every parameter's accumulated gradient (since
`backward` accumulates rather than overwrites, stale gradients from
the previous epoch must be cleared first), compute the loss, call
`.backward()` once, and step every parameter by `-learning_rate *
grad`.

**Validated on a known ground truth before trusting it on real data.**
Fitting $y = 2x_1 - 3x_2 + 1$ (an exactly-learnable linear function)
from 150 synthetic examples drives the loss from about 16.6 to under
0.1 within 150-300 epochs, confirming the engine, the chain rule
wiring, and the training loop are all correct before any of them are
asked a question with an ambiguous answer.

## Application: forecasting EUR/USD volatility, and an honest negative result

`stochastic-processes` established, through several independent
statistical tests, that EUR/USD daily returns cluster in variance.
The natural next question for a from-scratch autodiff engine to ask is
whether a small neural network can turn that population-level
statistical fact into a usable day-ahead forecast. The honest answer,
tested rather than assumed, is: not with this scale of data, network,
and training regime, and the reasons are worth stating precisely
rather than glossing over.

**The compute constraint came first.** A pure Python, scalar-valued
computational graph (no vectorization: every individual number is its
own `Value` object with real Python-level overhead) does not scale to
full-batch training on thousands of examples for hundreds of epochs in
reasonable time. Benchmarking directly: 5 epochs over 200 examples
took about 0.8 seconds, extrapolating to the full 5,540-example
training set at 150 epochs would take on the order of 20-25 minutes.
The practical fix used here is to train on a bounded recent slice of
the data (600 examples by default), which keeps a single run to about
a minute; the real fix, used by every production deep learning
framework, is to vectorize the `Value` class so that one node wraps an
entire numpy array (or a GPU tensor) instead of a single Python float,
turning $O(\text{number of individual numbers})$ Python-level overhead
into $O(\text{number of operations})$ overhead with the actual
arithmetic offloaded to fast, compiled, batched routines. That
rewrite is a substantial undertaking in its own right and is out of
scope for a from-scratch educational engine, but it is the concrete,
correct answer to "how would you actually make this fast," not a hand
wave.

**With that constraint accepted, the forecasting result itself is a
genuine negative, not a bug.** Predicting the next day's squared log
return from the previous 5 days' squared log returns, training a
small (6-hidden-unit) network on 600 recent in-sample observations,
gives a test-set MSE about 24% *worse* than the naive baseline of
always predicting the training-set mean. Widening the target to a
5-day forward-averaged squared return (a smoother, less noisy realized
-variance proxy, which ought to be easier to predict if there is
usable signal) made the comparison worse, not better (about -85%
relative to baseline). The training loss did decrease during fitting
in both cases (confirming the network is genuinely learning something
about the 600 training examples), so this is a textbook case of
overfitting to sample-specific noise rather than a broken training
loop, and it points to a real, specific limitation of trying to
recover a statistically well-established but *quantitatively weak*
effect from a small, general-purpose model: `stochastic-processes`
detected volatility clustering using large-sample (6,930-observation),
purpose-built statistical tests (a chi-square dispersion test, a
transition-matrix independence test) specifically designed to average
over exactly the kind of single-day idiosyncratic noise that dominates
a raw squared return. A single day's squared return is a highly
volatile, unbiased estimator of the underlying latent variance
process; recovering that process's genuinely predictable component
from it requires either far more training data than the compute
budget here allowed, or purpose-built volatility models (GARCH-family
models, or the realized-variance and dispersion-test machinery already
built in `stochastic-processes` and `black-scholes`) that are
explicitly designed to separate the noise from the signal, rather than
asking a small, generically-configured neural network to rediscover
that separation on 600 examples of raw squared returns. Reporting a
model that quietly "worked" here, after two different target
formulations both failed to beat the baseline, would have been
cherry-picking; the honest result is that a real, population-level
statistical effect does not automatically hand a small, naively
applied model a usable out-of-sample edge, and that gap is itself the
finding worth recording.

## Limitations

- The neural network's training-set size for the real-data application
  is capped by wall-clock compute cost, not by data availability;
  results at this scale should not be read as a definitive statement
  about whether EUR/USD volatility is forecastable from lagged squared
  returns by *any* model, only by this one at this scale.
- `Value.backward`'s iterative topological sort fixes the crash found
  during development, but building a fresh computational graph for
  every training epoch (rather than reusing and resetting one graph)
  is itself a further, unaddressed inefficiency; a production system
  would typically reuse the graph structure across epochs and only
  reset the numeric data and gradients.
- `jacobian_reverse_mode` rebuilds the graph from scratch for every
  output row rather than sharing one forward pass and resetting
  gradients between backward passes; this is simpler to reason about
  correctly but costs an extra $m-1$ forward evaluations relative to
  the theoretical minimum for an $m$-output function.
- Dual numbers here support only single-variable seeding per pass
  (one $\varepsilon$ component); a "vector forward mode" that carries
  an entire directional-derivative vector through a single pass (and
  can therefore compute several columns of a Jacobian at once) is a
  standard extension not implemented here.

## CLI examples

```bash
# Differentiate an arbitrary expression via forward-mode AD
python3 cli.py derivative --expr "sin(x)*exp(x)" --x 0.8

# Concrete forward-vs-reverse-mode pass-count comparison
python3 cli.py costs --n-inputs 1000000 --n-outputs 1

# Train the from-scratch neural network on synthetic data, then on
# real EUR/USD volatility forecasting (see the README for why the
# latter is a reported negative result, not a bug)
python3 cli.py volatility --lags 5 --forward-window 1 --train-size 600 --epochs 150

# Cross-check every derivative and Jacobian three ways, and verify the
# training loop end to end against a finite-difference gradient
python3 cli.py verify
```
