# Project 3: Poisson Processes

Arrival counts, inter-arrival times, superposition, thinning, and
time-varying rates for the Poisson process, closed out with a formal
test of whether large EUR/USD moves actually arrive as a homogeneous
Poisson process (they do not).

## Mathematical background

### The counting process and its exact distribution

A homogeneous Poisson process with intensity $\lambda$ counts events
in $[0, t]$ according to

$$P(N(t) = k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}, \qquad k = 0, 1, 2, \dots$$

implemented in log-space (`k*log(mean) - mean - lgamma(k+1)`, then
exponentiated) so that the factorial term stays numerically stable for
large $k$ instead of overflowing `math.factorial`. The defining
**equidispersion** property, $E[N(t)] = \mathrm{Var}(N(t)) = \lambda t$,
falls straight out of this pmf and is checked directly, and again
empirically through simulation.

### Inter-arrival times and the memoryless property

The gaps between successive arrivals of a Poisson($\lambda$) process
are i.i.d. Exponential($\lambda$). `simulate_arrivals` generates them
by inverse-transform sampling: if $U \sim \mathrm{Uniform}(0,1)$ then

$$-\frac{\ln U}{\lambda} \sim \mathrm{Exponential}(\lambda),$$

because $P(-\ln U / \lambda > x) = P(U < e^{-\lambda x}) = e^{-\lambda x}$,
which is exactly the exponential survival function. Summing these
gaps until the running total exceeds the horizon produces a full
arrival-time path. `simulate_count_direct` generates a single count
$N(t)$ a completely different way (Knuth's algorithm: multiply
uniforms until the running product falls below $e^{-\lambda t}$,
which exploits the fact that the number of uniforms multiplied before
crossing that threshold is itself Poisson-distributed) so that the two
independent constructions can be checked against each other without
either one being trusted in advance.

### Superposition and thinning

**Superposition.** If $N_1, \dots, N_k$ are independent Poisson
processes with rates $\lambda_1, \dots, \lambda_k$, their union
(merge all arrivals and sort) is again Poisson, with rate
$\sum_i \lambda_i$. This follows because sums of independent Poisson
random variables are Poisson with summed means, applied to every
sub-interval simultaneously. `superpose` just merges and sorts; the
rate arithmetic is verified against direct simulation and against a
numpy Poisson oracle at the combined rate.

**Thinning.** If each arrival of a Poisson($\lambda$) process is
independently kept with probability $p$, the kept and discarded
streams are themselves independent Poisson processes with rates
$\lambda p$ and $\lambda(1-p)$. This is the converse operation to
superposition, and the two theorems are checked against each other in
`verify`: thinning a merged process and merging a thinned process both
have to reproduce the original rate structure.

### Inhomogeneous processes and the Lewis-Shedler thinning algorithm

When the intensity varies with time, $\lambda(t)$, the mean count over
$[0, t]$ is $\int_0^t \lambda(u)\, du$ (implemented here with a plain
trapezoidal rule, `expected_inhomogeneous_count`, since the goal is a
correct from-scratch quadrature rather than an adaptive one).
Simulating the process directly from `simulate_arrivals` at a varying
rate is not valid, since that routine assumes a constant rate between
gaps. Instead, `simulate_inhomogeneous` uses the standard
**Lewis-Shedler thinning algorithm**: generate a homogeneous
Poisson($\lambda_{\max}$) process, where $\lambda_{\max}$ bounds
$\lambda(t)$ over the horizon, then keep each candidate point $t$
independently with probability $\lambda(t) / \lambda_{\max}$. The
kept points are Poisson-thinned by a time-varying probability, which
is exactly the inhomogeneous process by construction. The simulated
mean count is checked against the trapezoidal integral, and the
integral itself is cross-checked against a Monte Carlo estimate.

### Testing Poisson-ness: the dispersion test

Given observed counts $n_1, \dots, n_m$ in $m$ equal-length windows,
the null hypothesis that they are i.i.d. Poisson($\mu$) implies

$$D = \sum_{i=1}^m \frac{(n_i - \bar n)^2}{\bar n} \sim \chi^2_{m-1}$$

approximately (**Fisher's dispersion test**), because under Poisson
equidispersion the variance-to-mean ratio should sit at 1, and $D$ is
$(m-1)$ times the sample estimate of that ratio. Values of $D$ far
above $m - 1$ mean the counts are **overdispersed**: more clustered in
time than a homogeneous Poisson process would produce. Getting a
p-value requires the chi-square survival function, which requires the
regularised incomplete gamma function; the same series/continued-
fraction construction from Project 2 is reimplemented here (rather
than imported) so this project stands on its own, and is again checked
against `scipy.stats.chi2.sf`.

## Application: do large EUR/USD moves arrive as a Poisson process?

"Large move" is defined as any day whose absolute return exceeds the
95th percentile of the absolute-return distribution, with the
threshold computed from the data itself rather than fixed in advance.
That gives 347 large-move days out of 6,930, an estimated rate of
about 0.05 per trading day, so a homogeneous Poisson null hypothesis
would predict inter-arrival gaps averaging $1/\hat\lambda \approx 20$
trading days with a coefficient of variation of exactly 1 (the
defining feature of the exponential distribution the null implies).

Two independent pieces of evidence reject that null decisively:

1. **Coefficient of variation of the gaps is about 2.13**, roughly
   double what memorylessness would produce. A CV well above 1 means
   the gap distribution is far more spread out than exponential: long
   quiet stretches punctuated by tight clusters of large moves, rather
   than a steady trickle.
2. **Fisher's dispersion test on 20 equal-length windows gives an
   index of dispersion around 12.6** (statistic 239.7 on 19 degrees of
   freedom), with a p-value on the order of $10^{-40}$. Counts of
   large moves per window vary far more than a Poisson process of the
   same mean rate would allow.

Both point the same direction as Project 2's finding on the same
data: large moves cluster together in time (this is the same
volatility-clustering phenomenon Project 2 detected through its
5-state magnitude chain, seen here from a completely different angle,
through arrival-time statistics rather than a transition matrix). A
homogeneous Poisson process is the natural formal null hypothesis for
"large moves happen independently and at a constant rate," and the
data rejects it about as strongly as a null hypothesis can be
rejected. This is a genuinely useful negative result for risk
management: sizing tail-risk exposure as if large moves were Poisson-
distributed in time would understate the probability of several large
moves occurring in quick succession.

## Limitations

- The 95th-percentile threshold is one reasonable choice; a different
  quantile changes $\hat\lambda$ and the absolute dispersion statistic,
  though the qualitative rejection of the Poisson null is robust to
  this choice (checked at the 90th percentile as well, in the test
  suite).
- Fisher's dispersion test is a large-sample chi-square approximation;
  with only 347 arrivals split across many windows some bins can be
  sparse, which the code does not separately guard against beyond
  requiring a non-zero mean count.
- `simulate_inhomogeneous` requires an explicit bound $\lambda_{\max}$
  on the rate function over the horizon and raises if that bound is
  violated; it does not search for the bound automatically.
- The coefficient-of-variation diagnostic and the dispersion test both
  detect departures from Poisson-ness but do not by themselves
  identify the mechanism (volatility clustering, macro announcement
  calendars, and autocorrelated variance all produce overdispersion in
  different ways); distinguishing between these is out of scope here.

## CLI examples

```bash
# Exact Poisson count distribution
python3 cli.py pmf --rate 2.0 --t 3.0 --kmax 10

# Simulate a homogeneous process and check gap statistics
python3 cli.py simulate --rate 3.0 --t 100 --seed 1

# Superposition of two independent processes
python3 cli.py superpose --rate1 2.0 --rate2 3.0 --t 100 --seed 1

# Thinning by independent marking
python3 cli.py thin --rate 5.0 --t 100 --keep-prob 0.3 --seed 1

# Time-varying rate via Lewis-Shedler thinning
python3 cli.py inhomogeneous --base-rate 1.0 --slope 0.2 --t 20 --seed 1

# Test EUR/USD large moves against the Poisson null
python3 cli.py fxpoisson --quantile 0.95 --bins 20

# Cross-check every formula against scipy/numpy oracles
python3 cli.py verify
```

## Tests

```bash
python3 -m pytest test_poisson_processes.py -v
```

47 tests covering: the exact pmf/cdf and its equidispersion property,
two independent simulation methods for arrival counts, inter-arrival
gap statistics, superposition and thinning rate arithmetic, the
Lewis-Shedler algorithm for time-varying rates, the regularised
incomplete gamma function behind the dispersion test, and the EUR/USD
application including a direct check that real large-move arrivals are
overdispersed relative to the Poisson null.
