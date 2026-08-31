# Phase 4 - Calculus

Calculus I, II, and III implemented numerically. This phase builds the
computational machinery that underlies every quantitative model:
derivatives, integrals, series expansions, and multivariable calculus.

The approach is numerical rather than symbolic. Every derivative is
computed by evaluating a function at nearby points. Every integral is
computed by summing up areas. This is how these operations are actually
done in software -- and building them from scratch is what makes the
formulas stick.

| Project | Folder | Tests | Status |
|---|---|---|---|
| 9 - Numerical Differentiation Engine | `project9_numerical_differentiation/` | 84 | done |
| 10 - Numerical Integrator & Taylor Series | `project10_numerical_integration/` | 111 | done |
| 11 - Gradient Field Visualiser | `project11_gradient_field/` | 70 | done |

## What is implemented

**Project 9** builds four finite difference methods (forward, backward,
central, five-point stencil) with automatic optimal step size selection,
second derivatives, gradient of a scalar field, directional derivatives,
mixed second partial derivatives, the Jacobian matrix of a vector-valued
function, numerical verification of the product and chain rules, gradient
descent in n dimensions, sensitivity analysis, and bond duration and
convexity computed as derivatives of the price-yield function.

**Project 10** builds Riemann sums (left, right, midpoint), the
trapezoidal rule, Simpson's rule with error estimates, adaptive
Simpson's quadrature, Gauss-Legendre quadrature (n = 2–10), improper
integrals via substitution, convergence rate tables, Taylor series
coefficients via central differences and Richardson extrapolation,
Horner evaluation, radius of convergence estimation, and applied
examples: normal/exponential/beta CDFs, Black-Scholes option pricing,
bond duration, and present value of continuous cash flows.

**Project 11** builds partial derivatives in 2D and 3D, the gradient
vector, directional derivatives, the Hessian matrix, the Laplacian,
critical point classification via the second derivative test, gradient
descent with full trajectory tracking, learning rate comparison,
gradient field evaluation on a grid, level curve sampling, divergence,
and the scalar curl of a 2D vector field.

## Why numerical

Symbolic differentiation gives exact answers but requires a computer
algebra system. Numerical differentiation works on any function you can
evaluate -- a black-box model, a simulation, a lookup table. The trade-off
is accuracy limited by floating-point arithmetic, which is why choosing
the right step size matters and why four methods with different accuracy
orders are all worth knowing.

The error analysis in Project 9 shows this directly: for a given step
size h, the forward difference error is O(h), the central difference
error is O(h^2), and the five-point stencil error is O(h^4). But make h
too small and floating-point cancellation dominates, so each method has
an optimal step size. The module computes these automatically.

## Running

```bash
cd phase4_calculus/project9_numerical_differentiation && python3 -m pytest test_differentiator.py -v
cd phase4_calculus/project9_numerical_differentiation && python3 cli.py verify
```
