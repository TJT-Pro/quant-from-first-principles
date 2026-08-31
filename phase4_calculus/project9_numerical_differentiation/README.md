# Project 9 - Numerical Differentiation Engine

A numerical differentiation library built entirely from the definitions.
No symbolic algebra library is used anywhere. Every derivative is
computed by evaluating the function at nearby points and applying the
appropriate finite difference formula.

| Idea | What it computes |
|---|---|
| Forward difference | f'(x) ≈ [f(x+h) - f(x)] / h, O(h) accurate |
| Backward difference | f'(x) ≈ [f(x) - f(x-h)] / h, O(h) accurate |
| Central difference | f'(x) ≈ [f(x+h) - f(x-h)] / (2h), O(h^2) accurate |
| Five-point stencil | f'(x) from four evaluations, O(h^4) accurate |
| Second derivative | f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h^2 |
| Gradient | vector of partial derivatives of a scalar field |
| Directional derivative | rate of change along a unit vector |
| Mixed partial | d^2f / dx_i dx_j by applying central difference twice |
| Jacobian | matrix of partial derivatives of a vector-valued function |
| Product rule | (fg)' = f'g + fg', verified numerically |
| Chain rule | (f o g)' = f'(g) * g', verified numerically |
| Gradient descent | follow the negative gradient to find a local minimum |
| Sensitivity | how much each input affects the output |
| Bond duration | -dP/dy / P, the price sensitivity to yield |
| Bond convexity | d^2P/dy^2 / P, the curvature of the price-yield curve |

## Why four methods

All four formulas approximate the same thing but with different accuracy
and cost.

The **forward difference** uses f(x) and f(x+h). The truncation error
is O(h): halving h roughly halves the error until floating-point
cancellation takes over. Use this only when you cannot evaluate the
function to the left of x.

The **central difference** uses f(x+h) and f(x-h). The odd-order Taylor
terms cancel and the error drops to O(h^2): halving h quarters the
error. Almost always prefer this over forward or backward.

The **five-point stencil** uses f(x±h) and f(x±2h). The h, h^2, and h^3
error terms all cancel, leaving O(h^4) accuracy. At the same step size
h=1e-3, the five-point error is roughly 10,000 times smaller than the
forward difference error and about 100 times smaller than the central
difference error.

The **optimal step size** balances truncation error (smaller h is
better) against floating-point cancellation (larger h is better). The
optimal h is approximately:

    forward/backward : eps^(1/2) ~ 1.5e-8
    central          : eps^(1/3) ~ 6.1e-6
    five-point       : eps^(1/5) ~ 3.7e-4
    second derivative: eps^(1/4) ~ 1.2e-4

where eps = 2.22e-16 is machine precision. The module computes these
defaults automatically, scaled to the magnitude of x.

## The error-versus-h curve

If you run the error command you will see the characteristic shape: the
error decreases as h shrinks (truncation error falling) until a minimum
is reached, then rises again as h shrinks further (cancellation error
growing). The minimum occurs at the optimal h above. The five-point
stencil reaches its minimum at a larger h than central difference,
which is why it can use a coarser step and still be more accurate.

## Gradient and partial derivatives

The gradient of a scalar field f at a point x is the vector of all
partial derivatives:

    grad f(x) = [df/dx_0, df/dx_1, ..., df/dx_{n-1}]

Each partial derivative is computed by central difference, perturbing
only the i-th coordinate while holding the others fixed.

The gradient points in the direction of steepest ascent of f. Its
magnitude is the rate of change in that direction. This is what
gradient descent uses.

## Gradient descent

Starting from x0, the algorithm repeatedly takes a step in the
direction of the negative gradient:

    x_{k+1} = x_k - alpha * grad f(x_k)

where alpha is the learning rate. The step is small enough (when alpha
is chosen well) that the function value decreases at each step. The
algorithm stops when the gradient magnitude is below the tolerance.

Gradient descent is the foundation of every machine learning optimiser.
The Adam and momentum variants in Project 14 extend this same idea.

## Bond duration and convexity

Bond analytics are a direct application of differentiation to finance.

The price of a fixed-coupon bond is:

    P(y) = sum_{t=1}^{n} C / (1+y)^t + F / (1+y)^n

where C is the coupon per period, F is the face value, and y is the
yield per period.

**Modified duration** is the negative logarithmic derivative:

    D_mod = -dP/dy / P

It measures price sensitivity: a bond with modified duration 7 loses
approximately 7% in price when yield rises 1%.

**Convexity** is the second derivative divided by price:

    C = d^2P/dy^2 / P

The combined approximation is more accurate than duration alone:

    dP/P ≈ -D_mod * dy + 0.5 * C * dy^2

The module computes both duration and convexity by numerical
differentiation and compares the approximations against the exact price
change to show when the convexity correction matters.

## Quick start

Differentiate sin(x) at x=1 using each method:

```
python3 cli.py diff --f "sin(x)" --x 1.0 --method central
python3 cli.py diff --f "sin(x)" --x 1.0 --method five-point
```

Error versus step size table for sin(x) at x=1:

```
python3 cli.py error --f "sin(x)" --x 1.0 --exact 0.5403023058681398
```

Gradient of x^2 + y^2 at (3, 4):

```
python3 cli.py gradient --f "x[0]**2 + x[1]**2" --x 3.0,4.0
```

Find the minimum of (x-2)^2 + (y-3)^2 by gradient descent:

```
python3 cli.py descent --f "(x[0]-2)**2 + (x[1]-3)**2" --x 0.0,0.0 --lr 0.1
```

Bond analytics for a 5% coupon, 10-year bond at 5% yield:

```
python3 cli.py bond --yield 0.05 --coupon 50 --face 1000 --n 10
```

Same bond with a 1% yield shock:

```
python3 cli.py bond --yield 0.05 --coupon 50 --face 1000 --n 10 --dy 0.01
```

Cross-check everything against exact values:

```
python3 cli.py verify
```

## Limitations

Finite difference methods require the function to be smooth at the
point of differentiation. Applying them to non-smooth functions
produces numerically large but mathematically meaningless results.
For example, `second_derivative(abs, 0.0)` returns a large number
because `abs` has a kink at zero and no second derivative there. The
module does not detect non-smoothness -- that is your responsibility
as the caller. If you are differentiating a function and get a
suspiciously large or unstable result, check whether the function is
smooth at that point.

Gradient descent converges only for convex functions or from a
starting point close enough to a local minimum. On non-convex
functions it finds a local minimum, not necessarily the global one.
The learning rate matters: too large and it overshoots, too small
and it converges slowly. The module does not tune the learning rate
automatically.

## Running the tests

```
python3 -m pytest test_differentiator.py -v
```

84 tests covering all four finite difference methods, second
derivatives, gradients, partial derivatives, the Jacobian, product and
chain rules, gradient descent, sensitivity, and bond analytics.
