import math
from typing import Dict

import linear_algebra as la


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_closed_form(s: float, k: float, r: float, sigma: float, t: float,
                               option_type: str = "call") -> float:
    """The closed-form Black-Scholes price, used here purely as an
    independent ground truth for the finite-difference PDE solvers
    below (reimplemented fresh, sharing no code with the sibling
    black-scholes repository's own closed-form module): with
    d1 = [ln(S/K) + (r+0.5*sigma^2)*T] / (sigma*sqrt(T)),
    d2 = d1 - sigma*sqrt(T), the price is
    S*N(d1) - K*e^{-rT}*N(d2) for a call and
    K*e^{-rT}*N(-d2) - S*N(-d1) for a put.
    """
    if t <= 0:
        intrinsic = max(s - k, 0.0) if option_type == "call" else max(k - s, 0.0)
        return intrinsic
    d1 = (math.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if option_type == "call":
        return s * norm_cdf(d1) - k * math.exp(-r * t) * norm_cdf(d2)
    return k * math.exp(-r * t) * norm_cdf(-d2) - s * norm_cdf(-d1)


def _setup_log_price_grid(k: float, r: float, sigma: float, t: float, x_min: float, x_max: float,
                           n_space: int, n_time: int, option_type: str):
    """Common setup for all three PDE schemes below: transforming to
    log-price x=ln(S) and reversed time tau=T-t turns the Black-Scholes
    PDE into the constant-coefficient convection-diffusion-reaction
    equation

        dV/dtau = 0.5*sigma^2*d^2V/dx^2 + (r-0.5*sigma^2)*dV/dx - r*V,

    an initial-value problem in tau starting from the option's terminal
    payoff at tau=0 (equivalently, t=T), which is the natural direction
    to march a finite-difference scheme forward in, exactly the same
    reversed-time convention that turns the heat equation's terminal
    condition (if it had one) into an initial condition. Central
    differences for both dV/dx and d^2V/dx^2 give coefficients
    a=0.5*sigma^2/dx^2 (diffusion) and b=(r-0.5*sigma^2)/(2*dx)
    (convection, from the risk-neutral drift), combined into the
    single spatial operator coefficients (a+b), -(2a+r), (a-b) for the
    i+1, i, i-1 grid points respectively.
    """
    dx = (x_max - x_min) / n_space
    dtau = t / n_time
    x = la.linspace(x_min, x_max, n_space + 1)

    a = 0.5 * sigma ** 2 / dx ** 2
    b = (r - 0.5 * sigma ** 2) / (2 * dx)

    if option_type == "call":
        v = [max(math.exp(xi) - k, 0.0) for xi in x]
    else:
        v = [max(k - math.exp(xi), 0.0) for xi in x]

    def left_boundary(tau):
        if option_type == "call":
            return 0.0
        return k * math.exp(-r * tau) - math.exp(x_min)

    def right_boundary(tau):
        if option_type == "call":
            return math.exp(x_max) - k * math.exp(-r * tau)
        return 0.0

    return x, v, dx, dtau, a, b, left_boundary, right_boundary


def solve_bs_pde_explicit(k: float, r: float, sigma: float, t: float, x_min: float, x_max: float,
                           n_space: int, n_time: int, option_type: str = "call") -> Dict:
    """Explicit finite-difference solution of the Black-Scholes PDE.
    Stability of this scheme (a direct extension of the heat
    equation's own explicit scheme, with the extra convection and
    reaction terms) requires the same underlying diffusion term to
    satisfy an r-like bound in addition to a CFL-style bound on the
    convection term; in practice this means n_time must be chosen
    large enough relative to n_space's implied dx, exactly analogous
    to the heat equation's r<=0.5 condition (see stability_analysis.py).
    """
    x, v, dx, dtau, a, b, left_boundary, right_boundary = _setup_log_price_grid(
        k, r, sigma, t, x_min, x_max, n_space, n_time, option_type)

    for n in range(n_time):
        tau_next = (n + 1) * dtau
        v_new = [0.0] * (n_space + 1)
        v_new[0] = left_boundary(tau_next)
        v_new[n_space] = right_boundary(tau_next)
        for i in range(1, n_space):
            rhs = (a + b) * v[i + 1] + (-2 * a - r) * v[i] + (a - b) * v[i - 1]
            v_new[i] = v[i] + dtau * rhs
        v = v_new

    return {"x": x, "v": v, "s": [math.exp(xi) for xi in x], "dx": dx, "dtau": dtau}


def solve_bs_pde_implicit(k: float, r: float, sigma: float, t: float, x_min: float, x_max: float,
                           n_space: int, n_time: int, option_type: str = "call") -> Dict:
    """Implicit (backward Euler in tau) finite-difference solution:
    unconditionally stable, at the cost of solving one tridiagonal
    system per time step via linear_algebra.thomas_algorithm.
    """
    x, v, dx, dtau, a, b, left_boundary, right_boundary = _setup_log_price_grid(
        k, r, sigma, t, x_min, x_max, n_space, n_time, option_type)

    n_interior = n_space - 1
    lower = [0.0] + [-dtau * (a - b)] * (n_interior - 1)
    diag = [1 + dtau * (2 * a + r)] * n_interior
    upper = [-dtau * (a + b)] * (n_interior - 1) + [0.0]

    for n in range(n_time):
        tau_next = (n + 1) * dtau
        left_val = left_boundary(tau_next)
        right_val = right_boundary(tau_next)

        rhs = v[1:n_space]
        rhs[0] += dtau * (a - b) * left_val
        rhs[-1] += dtau * (a + b) * right_val

        interior = la.thomas_algorithm(lower, diag, upper, rhs)
        v = [left_val] + interior + [right_val]

    return {"x": x, "v": v, "s": [math.exp(xi) for xi in x], "dx": dx, "dtau": dtau}


def solve_bs_pde_crank_nicolson(k: float, r: float, sigma: float, t: float, x_min: float, x_max: float,
                                 n_space: int, n_time: int, option_type: str = "call") -> Dict:
    """Crank-Nicolson finite-difference solution: unconditionally
    stable and second-order accurate in both x and tau, by averaging
    the explicit and implicit schemes' spatial operators exactly as in
    heat_equation.solve_heat_crank_nicolson.
    """
    x, v, dx, dtau, a, b, left_boundary, right_boundary = _setup_log_price_grid(
        k, r, sigma, t, x_min, x_max, n_space, n_time, option_type)

    n_interior = n_space - 1
    lower = [0.0] + [-0.5 * dtau * (a - b)] * (n_interior - 1)
    diag = [1 + 0.5 * dtau * (2 * a + r)] * n_interior
    upper = [-0.5 * dtau * (a + b)] * (n_interior - 1) + [0.0]

    for n in range(n_time):
        tau_next = (n + 1) * dtau
        left_val = left_boundary(tau_next)
        right_val = right_boundary(tau_next)

        rhs = []
        for i in range(1, n_space):
            explicit_rhs = (a + b) * v[i + 1] + (-2 * a - r) * v[i] + (a - b) * v[i - 1]
            rhs.append(v[i] + 0.5 * dtau * explicit_rhs)
        rhs[0] += 0.5 * dtau * (a - b) * left_val
        rhs[-1] += 0.5 * dtau * (a + b) * right_val

        interior = la.thomas_algorithm(lower, diag, upper, rhs)
        v = [left_val] + interior + [right_val]

    return {"x": x, "v": v, "s": [math.exp(xi) for xi in x], "dx": dx, "dtau": dtau}


def price_at_spot(result: Dict, s0: float) -> float:
    """Linearly interpolate the PDE grid's solution at an arbitrary
    spot price s0 (which will not, in general, land exactly on a grid
    point), since the grid is built in log-price and s0's log will
    typically fall strictly between two adjacent grid values of x.
    """
    x0 = math.log(s0)
    x = result["x"]
    if x0 <= x[0]:
        return result["v"][0]
    if x0 >= x[-1]:
        return result["v"][-1]
    for i in range(len(x) - 1):
        if x[i] <= x0 <= x[i + 1]:
            frac = (x0 - x[i]) / (x[i + 1] - x[i])
            return result["v"][i] + frac * (result["v"][i + 1] - result["v"][i])
    raise ValueError("price_at_spot: could not locate s0 within the grid")
