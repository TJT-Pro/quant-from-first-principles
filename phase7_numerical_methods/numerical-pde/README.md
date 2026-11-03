# numerical-pde

A numerical PDE engine built from scratch: finite difference schemes
(explicit, implicit, Crank-Nicolson) for the heat and wave equations,
von Neumann stability analysis deriving the exact CFL conditions from
first principles, and the Black-Scholes PDE solved on a log-price grid
and validated against a freshly-derived closed-form formula, applied
to price a real EUR/USD option.

This repository is independent of the sibling repositories and reuses
none of their code, including its own from-scratch closed-form
Black-Scholes formula (no code shared with the sibling black-scholes
repository).

## Project structure

```
numerical-pde/
    README.md
    data/DEXUSEU_returns.csv
    linear_algebra.py       Thomas algorithm (tridiagonal solver)
    heat_equation.py        explicit, implicit, Crank-Nicolson schemes
    stability_analysis.py   von Neumann stability analysis, CFL derivation
    wave_equation.py        explicit central-difference scheme
    black_scholes_pde.py    Black-Scholes PDE via finite differences
    fx_application.py       pricing a real EUR/USD option
    test_*.py
    cli.py
```

## Status

66 tests, 14/14 verify checks, all passing.

## Setup

Python 3.11+. All solver code uses only the standard library.
`cli.py verify` uses numpy and scipy as independent oracles.

```bash
pip install --break-system-packages numpy scipy
```

## Running tests

```bash
python3 -m pytest test_linear_algebra.py test_heat_equation.py test_stability_analysis.py \
    test_wave_equation.py test_black_scholes_pde.py test_fx_application.py -v
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

### The Thomas algorithm

Implicit and Crank-Nicolson schemes both reduce, at every time step,
to solving one tridiagonal linear system (a standard second-derivative
stencil only ever couples each grid point to its immediate
neighbours). The Thomas algorithm exploits this directly: a forward
sweep eliminates the sub-diagonal in O(n) time by expressing each row
as a multiple of the row above it, followed by O(n) back-substitution,
against a generic dense solver's O(n^3). Verified to match
numpy.linalg.solve to machine precision on random diagonally-dominant
systems.

### The heat equation: three schemes, one CFL condition

For du/dt = alpha*d^2u/dx^2, the explicit (FTCS) scheme,
u_i^{n+1} = u_i^n + r*(u_{i+1}^n - 2*u_i^n + u_{i-1}^n) with
r = alpha*dt/dx^2, computes each new time level directly with no
linear solve, at the cost of only being stable for r<=0.5. The
implicit (backward Euler) scheme evaluates the spatial term at the new
time level, giving a tridiagonal system solved via the Thomas
algorithm, and is unconditionally stable for any r. Crank-Nicolson
averages the two, unconditionally stable like the implicit scheme but
with O(dt^2) local truncation error instead of O(dt).

Von Neumann stability analysis derives the exact r<=0.5 bound, not
just states it. Substituting a Fourier mode u_i^n = xi^n*e^{ikx_i}
into the explicit update gives the amplification factor
xi = 1 - 4*r*sin^2(theta/2) (using 2*cos(theta)-2 = -4*sin^2(theta/2)),
which must satisfy |xi|<=1 for every mode theta; the worst case
sin^2(theta/2)=1 forces r<=0.5 exactly. The corresponding implicit and
Crank-Nicolson amplification factors,
xi = 1/(1+4*r*sin^2(theta/2)) and
xi = (1-2*r*sin^2(theta/2))/(1+2*r*sin^2(theta/2)), are both bounded by
1 in magnitude for any r>=0, confirming unconditional stability
directly from the formula rather than by citation.

Both halves of this prediction are checked empirically, not just
derived. At r=0.625 (deliberately above 0.5) on a sinusoidal initial
condition with a known exact solution, the explicit scheme's error
explodes to 9.4e17 while implicit (4.7e-4) and Crank-Nicolson
(9.9e-5, smaller again, matching its better order) remain accurate.
The maximum amplification factor across all modes, computed directly
from the formulas, tracks this exactly: 1.0 at r<=0.5 for every
scheme, then 1.4, 3.0, and 19.0 for the explicit scheme at
r=0.6,1.0,5.0 while implicit and Crank-Nicolson stay at exactly 1.0
throughout. A single-mode simulation's measured amplitude ratio after
one step matches the theoretical growth factor to machine precision
(e.g. 0.8000000000000032 measured against 0.7999999999999997
theoretical) at both a stable and an unstable r.

### The wave equation and the Courant condition

For d^2u/dt^2 = c^2*d^2u/dx^2, the explicit central-difference
("leapfrog") scheme needs two previous time levels, so the first step
uses a special Taylor-expansion formula incorporating the initial
velocity and the PDE itself (substituting u_tt=c^2*u_xx for the term a
plain first-order step would otherwise drop). Its CFL condition is the
Courant number C = c*dt/dx <= 1: verified directly on a standing-wave
test case with known exact solution u(x,t)=sin(kx)*cos(kct), matching
to within 6e-5 at C=0.25, and exploding to a magnitude of 6.5e7 at
C=1.25. D'Alembert's solution, u(x,t) = 0.5*(f(x-ct)+f(x+ct)), is also
checked directly: a Gaussian pulse launched with zero initial velocity
splits into two half-amplitude peaks that have moved exactly plus and
minus ct by time t, as the closed-form solution requires.

### The Black-Scholes PDE, from scratch

Transforming to log-price x=ln(S) and reversed time tau=T-t turns
Black-Scholes into a constant-coefficient convection-diffusion-reaction
equation, dV/dtau = 0.5*sigma^2*V_xx + (r-0.5*sigma^2)*V_x - r*V, an
initial-value problem starting from the terminal payoff at tau=0.
Central differences give diffusion coefficient a=0.5*sigma^2/dx^2 and
convection coefficient b=(r-0.5*sigma^2)/(2*dx), combined into the
same three-scheme structure (explicit, implicit, Crank-Nicolson) as
the heat equation. The closed-form Black-Scholes formula used as
ground truth is reimplemented fully from scratch here (using math.erf
for the normal CDF), sharing no code with the sibling black-scholes
repository's own closed-form module, and checked directly against
scipy.stats.norm-based computation and against put-call parity holding
to 1e-10.

A finding that needed a second, more careful measurement to see
correctly. At a first, coarse choice of grid (n_space=200), all three
schemes' errors against the closed form looked similar (0.0053,
0.0063, 0.0058 for explicit, implicit, Crank-Nicolson respectively):
Crank-Nicolson showed no visible advantage, seemingly contradicting
its known better time-order. Investigating why: at that grid, the
fixed spatial discretization error dominated for every scheme equally,
masking any difference in time-order entirely. Refining the spatial
grid to n_space=400 and then varying only the number of time steps
isolates the real effect cleanly: the implicit scheme's error shrinks
roughly linearly as time steps increase (0.0538, 0.0224, 0.0119,
0.0067, 0.0035 for 20 to 500 steps), while Crank-Nicolson's error drops
almost immediately to a floor set by the (now-dominant) spatial error
and stays there (0.0285, 0.0014, 0.0014, 0.0014, 0.0014): exactly the
O(dt) versus O(dt^2) distinction the theory predicts, visible only
once the comparison is set up to isolate the right error source.

## Application: pricing a real EUR/USD option via the PDE engine

Volatility is calibrated directly from the sample standard deviation
of daily log returns, annualised: sigma_hat=0.0913 (9.1%) on the full
6,930-observation EUR/USD series. Pricing a 3-month at-the-money call
(spot 1.1684, matching strike, r=3%) gives a closed-form price of
0.024945, with all three PDE schemes agreeing to within 3e-6: explicit
0.024947, implicit 0.024942, Crank-Nicolson 0.024945 (the closest of
the three, consistent with its higher accuracy). A direct convergence
study refining only the spatial grid (holding the time grid fixed at
1,000 steps) shows the expected rapid improvement: errors of 9.3e-4,
1.3e-4, 1.6e-5, and 4.6e-7 at n_space=25,50,100,200, each roughly a
quarter of the previous as dx halves, consistent with the scheme's
second-order spatial accuracy, down to an error of under two
hundred-thousandths of a cent on an option worth about two and a half
cents.

## Limitations

- The Black-Scholes PDE is solved on a truncated log-price domain
  with approximate asymptotic boundary conditions (zero for a deep
  out-of-the-money call, intrinsic-value growth for a deep
  in-the-money one); a domain not wide enough relative to
  sigma*sqrt(T) would let these approximate boundaries meaningfully
  contaminate the price near the region of actual interest, which is
  why fx_application.py sizes the domain to 5 standard deviations of
  the terminal log-price distribution on each side of the strike.
- wave_equation.solve_wave_explicit's stability condition (C<=1) is
  demonstrated and verified empirically here, but this project does
  not include a full von Neumann derivation of the wave equation's own
  amplification factor (a quadratic, not linear, characteristic
  equation in xi, genuinely different from the heat equation's) the
  way stability_analysis.py does for the heat equation; only the heat
  equation's stability boundary is derived symbolically from first
  principles here.
- All finite-difference schemes in this project use uniform grids in
  both space and time; a genuinely nonuniform grid (finer near the
  strike, where an option's payoff has a kink and the second
  derivative used for Greeks is most sensitive) would need a modified
  discretization stencil not implemented here.

## CLI examples

```bash
# Solve the heat equation via all three schemes, compare to the exact solution
python3 cli.py heat --alpha 0.5 --length 1.0 --t-end 0.1 --n-space 50 --n-time 200

# Solve the wave equation, compare to the exact standing-wave solution
python3 cli.py wave --c 1.0 --length 1.0 --t-end 0.5 --n-space 100 --n-time 200

# Von Neumann stability analysis across a range of r
python3 cli.py stability

# Price a European option via closed form and all three PDE schemes
python3 cli.py price --s0 100 --k 100 --r 0.05 --sigma 0.2 --t 1.0

# Price a real EUR/USD option, with a spatial-grid convergence study
python3 cli.py fxprice --r 0.03 --t 0.25

# Cross-check every solver against scipy/numpy oracles and known theory
python3 cli.py verify
```
