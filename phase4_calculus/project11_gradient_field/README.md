# Project 11 — Gradient Field Visualiser & Partial Derivative Engine

Multivariable calculus implemented from scratch: partial derivatives,
gradients, Hessians, Laplacians, gradient descent with trajectory
tracking, vector field divergence and curl.  No numpy, no sympy,
no autograd.

---

## What is implemented

### Partial derivatives
Central difference with Richardson extrapolation, applied to the i-th
variable of a function that takes a list of floats.  Works in any
number of dimensions.

    df/dx_i ≈ [4·D(h/2) − D(h)] / 3   where D(h) = [f(x+h·eᵢ) − f(x−h·eᵢ)] / (2h)

O(h⁴) accurate after extrapolation.

### Gradient
The full gradient vector [df/dx₀, df/dx₁, …, df/dx_{n−1}] computed
by calling the partial derivative routine for each coordinate.  The
gradient points in the direction of steepest increase and its magnitude
is the maximum rate of change.

### Directional derivative
D_v f(x) = ∇f(x) · v̂, the rate of change in a given direction.  The
direction vector is normalised automatically.  By the Cauchy-Schwarz
inequality this is maximised when v is aligned with ∇f (giving ||∇f||)
and zero when v is perpendicular (level curve direction).

### Hessian matrix
n×n matrix of second partial derivatives H[i][j] = d²f/(dx_i dx_j).
Pure second derivatives use the three-point stencil; mixed partials
use the four-point cross stencil.  Both followed by Richardson
extrapolation.  The matrix is computed symmetrically (H[i][j] = H[j][i]).

### Laplacian
∇²f = trace(H) = sum of pure second partial derivatives.  Zero for
harmonic functions (x²−y², log r, etc.).  The sign indicates whether
the point is locally a minimum (positive) or maximum (negative) of the
function relative to its neighbourhood.

### Critical point classification
For 2D functions the second derivative test uses the Hessian determinant:

    D = f_xx·f_yy − f_xy²

    D > 0, f_xx > 0  →  local minimum
    D > 0, f_xx < 0  →  local maximum
    D < 0            →  saddle point
    D ≈ 0            →  inconclusive

### Gradient descent with trajectory
Standard first-order gradient descent:

    x_{k+1} = x_k − η · ∇f(x_k)

Returns the full trajectory as a list of (x, y, f) triples so descent
paths can be plotted on a contour map.  Also returns the gradient norm
at each step, which measures how close the iterate is to a critical
point.

### Learning rate comparison
Runs gradient descent at several learning rates from the same starting
point and reports iterations, convergence, final f value, and final
gradient norm for each.  Used to study how η affects speed and stability.

### Gradient field on a grid
Evaluates ∇f at every point of a 2D grid, returning gx, gy, and
||∇f|| arrays ready for quiver-plot rendering or analysis.

### Vector field operations
- **Divergence**: ∇·F = ∂Fx/∂x + ∂Fy/∂y — measures local source/sink
  strength.  Zero for incompressible (divergence-free) fields.
- **Scalar curl**: (∇×F)_z = ∂Fy/∂x − ∂Fx/∂y — measures local
  rotation.  Zero for irrotational (conservative) fields, which are
  exactly the gradient fields of scalar potentials.

---

## Key formulas

### Mixed second partial (Schwarz/Young's theorem)
    d²f/(dx dy) ≈ [f(x+h,y+h) − f(x+h,y−h) − f(x−h,y+h) + f(x−h,y−h)] / (4h²)

### Directional derivative via gradient
    D_v f = ∇f · (v / ||v||)

### Hessian determinant (2D)
    D = f_xx·f_yy − (f_xy)²

---

## Limitations

All derivatives are numerical.  On non-smooth functions or at
discontinuities, the central difference formulas return large but
meaningless values.  The step size defaults are tuned for smooth
functions; if you are differentiating something with large curvature
or near a singularity, pass an explicit h.

Gradient descent finds local minima, not global ones.  The learning
rate must be chosen carefully: too large and iterates oscillate or
diverge, too small and convergence is slow.  No line search or
adaptive step size is implemented -- this is the vanilla algorithm.

---

## Running the tests

```
python3 -m pytest test_gradient_field.py -v
```

70 tests covering partial derivatives in 2D and 3D, gradient vectors,
gradient magnitude, directional derivatives, second partial derivatives,
Hessian matrices, Laplacians, critical point classification (minimum,
maximum, saddle), gradient descent convergence and trajectory,
learning rate comparison, gradient field grids, level curve sampling,
divergence, and curl.

## Oracle verification

```
python3 cli.py verify
```

27 checks against exact analytical values: partial derivatives of
polynomials and trigonometric functions, 3D partial derivatives, gradient
of x²+y², directional derivatives along and perpendicular to the
gradient, second partials, Laplacian of harmonic function, critical point
classification for minimum/maximum/saddle, gradient descent convergence
and accuracy, learning rate ordering, gradient field grid values,
divergence of (x,y) and (1,1), and curl of the rotation field and a
gradient field.

## CLI examples

```bash
# Partial derivative df/dx at a point
python3 cli.py partial --f "x**2+y**2" --point 3,4 --i 0

# Full gradient
python3 cli.py gradient --f "sin(x)*cos(y)" --point 0.5,1.0

# Directional derivative
python3 cli.py directional --f "x**2+y**2" --point 3,4 --dir 1,0

# Hessian and Laplacian
python3 cli.py hessian --f "x**2+y**2" --point 1,1

# Classify critical point
python3 cli.py classify --f "x**2-y**2" --point 0,0

# Gradient descent
python3 cli.py descent --f "x**2+y**2" --x0 3 --y0 4 --lr 0.1 --verbose

# Compare learning rates
python3 cli.py compare --f "x**2+y**2" --x0 5 --y0 5 --rates 0.01,0.1,0.5

# Gradient field summary on a grid
python3 cli.py field --f "x**2+y**2" --xmin -2 --xmax 2 --ymin -2 --ymax 2

# Divergence
python3 cli.py divergence --fx "x" --fy "y" --point 1,2

# Scalar curl
python3 cli.py curl --fx "-y" --fy "x" --point 1,2
```
