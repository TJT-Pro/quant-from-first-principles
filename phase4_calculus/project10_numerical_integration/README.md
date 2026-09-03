# Project 10: Numerical Integrator & Taylor Series Expander

Numerical integration and Taylor series built entirely from definitions.
No `scipy.integrate`, no `sympy`, no special-function libraries.

---

## What is implemented

### Riemann sums
Three variants, each showing how the choice of sample point inside each
subinterval determines accuracy:

| Method | Sample point | Accuracy |
|--------|-------------|----------|
| Left   | left edge   | O(h): first-order |
| Right  | right edge  | O(h): first-order |
| Midpoint | centre    | O(h²): the cancellation of the linear error term gives second-order accuracy for the same cost |

### Trapezoidal rule
Connects adjacent function values with straight lines and sums trapezoid
areas. O(h²). Exact for linear functions. For periodic integrands on a
full period, endpoint errors cancel and it achieves spectral accuracy.

### Simpson's rule
Fits a quadratic through each triple of adjacent points and integrates
it exactly. O(h⁴): halving n reduces error by a factor of 16. n must
be even. The standard method for smooth functions.

Both trapezoidal and Simpson's include `_error_estimate` functions that
apply Richardson extrapolation to estimate the remaining error without
knowing the exact answer.

### Adaptive quadrature
Recursive adaptive Simpson's rule. Subdivides subintervals until the
local error estimate falls below the user's tolerance. Concentrates
evaluations where the function varies rapidly, achieving a target
accuracy with far fewer evaluations than a uniform grid.

### Gauss-Legendre quadrature
Pre-computed nodes and weights for n = 2, 3, 4, 5, 6, 7, 10.
Integrates polynomials of degree ≤ 2n−1 exactly. Converges
exponentially fast for smooth functions, far outperforming equally-
spaced methods.

A linear substitution maps [a, b] to [−1, 1]:

    x = (a+b)/2 + (b-a)/2 * t,   dx = (b-a)/2 * dt

### Improper integrals
Handles infinite limits by a change of variables. For [a, ∞):

    t = (x−a) / (1 + (x−a)),   x = a + t/(1−t),   dx = 1/(1−t)² dt

This maps t ∈ (0, 1) to x ∈ (a, ∞). The transformed integrand is
f(a + t/(1−t)) / (1−t)². (−∞, b] is handled symmetrically; (−∞, ∞)
splits at 0.

### Taylor series
Coefficients are computed by applying the central difference formula for
each derivative order, followed by one Richardson extrapolation step to
cancel the leading O(h²) error:

    f^(k)(x₀) ≈ [4·D(h/2) − D(h)] / 3

where D(h) is the central difference estimate at step h.

Evaluation uses Horner's method:

    a₀ + a₁(x−x₀) + a₂(x−x₀)² + ⋯
    = a₀ + (x−x₀)(a₁ + (x−x₀)(a₂ + ⋯))

requiring only n multiplications and n additions.

The radius of convergence is estimated via the ratio test on the last
pair of non-negligible coefficients.

### Probability applications
- **Normal CDF**: adaptive numerical integration of the standard normal
  PDF; matches `math.erf` to 8 decimal places
- **Exponential CDF**: adaptive integration of the exponential PDF
- **Beta CDF**: Simpson's rule with singular-endpoint handling

### Finance applications
- **Present value of continuous cash flows**: `∫₀ᵀ c(t) e^{−rt} dt`
- **Black-Scholes call/put**: full formula using the numerically
  computed normal CDF; put-call parity verified
- **Continuous bond duration**: both the price integral and the
  weighted-time integral computed numerically

---

## Formulas

### Trapezoidal error
    error ≈ (I(2n) − I(n)) / 3

### Simpson's error
    error ≈ (I(2n) − I(n)) / 15

### Black-Scholes call
    C = S·N(d₁) − K·e^{−rT}·N(d₂)
    d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d₂ = d₁ − σ√T

### Continuous bond
    P  = ∫₀ᵀ c·e^{−yt} dt + e^{−yT}
    D  = (1/P) · [∫₀ᵀ t·c·e^{−yt} dt + T·e^{−yT}]

---

## Limitations

**Taylor coefficients** are reliable for degree ≤ 6. Degrees 7 and 9
use antisymmetric stencils with points at 4h and 5h from x₀; the
near-cancellation of large numbers limits accuracy to 2–3 significant
digits for those specific coefficients regardless of step size. Degrees
6, 8, 10 (symmetric stencils) are unaffected. If you need reliable
high-degree coefficients, use a computer algebra system.

**Improper integrals** require the integrand to decay fast enough that
the transformed integral is well-conditioned. Slowly decaying
integrands may need a larger n.

**Beta CDF** uses a fixed n=500 by default. For very small or very
large shape parameters the integrand is sharply peaked and may need n
increased.

---

## Running the tests

```
python3 -m pytest test_integrator.py -v
```

111 tests covering left/right/midpoint Riemann sums, trapezoidal rule,
Simpson's rule, convergence rates, adaptive quadrature, Gauss-Legendre
quadrature, improper integrals, Taylor coefficients, Taylor evaluation,
Taylor error, radius of convergence, normal CDF, exponential CDF,
beta CDF, present value, Black-Scholes, and bond duration.

## Oracle verification

```
python3 cli.py verify
```

53 checks. Integration methods against exact polynomial and
trigonometric integrals, convergence ratios against expected O(h²) and
O(h⁴) rates, Gauss-Legendre exactness on polynomials, adaptive
Simpson's against a tight tolerance, improper integrals against
analytic values, Taylor coefficients of sin and exp against their exact
series, the normal CDF against `math.erf`, the exponential CDF against
`1 − e^{−x}`, Black-Scholes put-call parity, and the present value of
a constant cash flow against its closed form.

## CLI examples

```bash
# Simpson's rule on sin(x)
python3 cli.py integrate --f "sin(x)" --a 0 --b "pi" --method simpsons --n 100

# Compare all methods at n=1000
python3 cli.py compare --f "exp(x)" --a 0 --b 1 --exact 1.718281828459045

# Adaptive quadrature to 1e-10
python3 cli.py adaptive --f "sin(x)" --a 0 --b "pi" --tol 1e-10

# 5-point Gauss-Legendre
python3 cli.py gauss --f "exp(x)" --a 0 --b 1 --n 5

# Improper integral: Gaussian
python3 cli.py improper --f "exp(-x**2)" --a 0 --b inf

# Taylor series of sin(x) at 0 up to degree 9
python3 cli.py taylor --f "sin(x)" --x0 0 --degree 9 --at "-1,0,0.5,1"

# Normal CDF at 1.96
python3 cli.py normal-cdf --x 1.96

# Black-Scholes ATM call
python3 cli.py bs-call --S 100 --K 100 --r 0.05 --sigma 0.2 --T 1

# Continuous bond duration
python3 cli.py bond-dur --yield 0.05 --coupon 0.06 --T 10
```
