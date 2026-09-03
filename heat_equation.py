import math
from typing import Callable, Dict

import linear_algebra as la


def solve_heat_explicit(alpha: float, length: float, t_end: float, n_space: int, n_time: int,
                         initial_condition: Callable[[float], float],
                         left_bc: Callable[[float], float], right_bc: Callable[[float], float]) -> Dict:
    """Solve du/dt = alpha*d^2u/dx^2 on [0,length] via the explicit
    (forward-time, centred-space, "FTCS") finite difference scheme:

        u_i^{n+1} = u_i^n + r*(u_{i+1}^n - 2*u_i^n + u_{i-1}^n),  r = alpha*dt/dx^2

    obtained by replacing du/dt with a forward difference in time and
    d^2u/dx^2 with the standard centred second difference in space.
    Every new time level is computed directly from the previous one
    with no linear system to solve, the cheapest possible scheme per
    step, at the cost of only being stable for r<=0.5 (see
    stability_analysis.py for the von Neumann derivation of this exact
    bound, and for what happens numerically once r exceeds it).
    """
    dx = length / n_space
    dt = t_end / n_time
    r = alpha * dt / dx ** 2

    x = la.linspace(0.0, length, n_space + 1)
    u = [initial_condition(xi) for xi in x]
    u[0] = left_bc(0.0)
    u[-1] = right_bc(0.0)

    for n in range(n_time):
        t_next = (n + 1) * dt
        u_new = [0.0] * (n_space + 1)
        u_new[0] = left_bc(t_next)
        u_new[n_space] = right_bc(t_next)
        for i in range(1, n_space):
            u_new[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1])
        u = u_new

    return {"x": x, "u": u, "r": r, "dx": dx, "dt": dt}


def solve_heat_implicit(alpha: float, length: float, t_end: float, n_space: int, n_time: int,
                         initial_condition: Callable[[float], float],
                         left_bc: Callable[[float], float], right_bc: Callable[[float], float]) -> Dict:
    """Solve the same heat equation via the implicit (backward Euler)
    scheme: replacing du/dt with a forward difference but evaluating
    the spatial second difference at the new time level n+1 instead of
    the old one,

        u_i^{n+1} - r*(u_{i+1}^{n+1} - 2*u_i^{n+1} + u_{i-1}^{n+1}) = u_i^n,

    which rearranges into a tridiagonal linear system for the entire
    new time level at once,
    -r*u_{i-1}^{n+1} + (1+2r)*u_i^{n+1} - r*u_{i+1}^{n+1} = u_i^n,
    solved here via linear_algebra.thomas_algorithm. This is
    unconditionally stable for any r>0 (see stability_analysis.py), at
    the cost of an O(n) tridiagonal solve every time step instead of
    the explicit scheme's O(n) direct update, the same asymptotic cost
    per step, but with the constant-factor overhead of a linear solve,
    in exchange for never needing to worry about a stability
    restriction on the time step at all.
    """
    dx = length / n_space
    dt = t_end / n_time
    r = alpha * dt / dx ** 2

    x = la.linspace(0.0, length, n_space + 1)
    u = [initial_condition(xi) for xi in x]
    u[0] = left_bc(0.0)
    u[-1] = right_bc(0.0)

    n_interior = n_space - 1
    lower = [0.0] + [-r] * (n_interior - 1)
    diag = [1 + 2 * r] * n_interior
    upper = [-r] * (n_interior - 1) + [0.0]

    for n in range(n_time):
        t_next = (n + 1) * dt
        left_val = left_bc(t_next)
        right_val = right_bc(t_next)

        rhs = u[1:n_space]
        rhs[0] += r * left_val
        rhs[-1] += r * right_val

        interior = la.thomas_algorithm(lower, diag, upper, rhs)
        u = [left_val] + interior + [right_val]

    return {"x": x, "u": u, "r": r, "dx": dx, "dt": dt}


def solve_heat_crank_nicolson(alpha: float, length: float, t_end: float, n_space: int, n_time: int,
                               initial_condition: Callable[[float], float],
                               left_bc: Callable[[float], float], right_bc: Callable[[float], float]) -> Dict:
    """Solve the heat equation via Crank-Nicolson: average the
    explicit and implicit schemes' spatial second differences (one
    evaluated at the old time level, one at the new), which is
    equivalent to centring the time discretization at the midpoint
    t_{n+1/2} rather than at either endpoint:

        -r/2*u_{i-1}^{n+1} + (1+r)*u_i^{n+1} - r/2*u_{i+1}^{n+1}
            = r/2*u_{i-1}^n + (1-r)*u_i^n + r/2*u_{i+1}^n

    Also tridiagonal (solved the same way as the implicit scheme) and
    unconditionally stable, but with local truncation error O(dt^2)
    instead of O(dt) in time, since centring at the midpoint cancels
    the leading time-discretization error term the same way a centred
    spatial difference cancels the leading spatial error term, giving
    this scheme second-order accuracy in both space and time at
    roughly the same per-step cost as the implicit scheme.
    """
    dx = length / n_space
    dt = t_end / n_time
    r = alpha * dt / dx ** 2

    x = la.linspace(0.0, length, n_space + 1)
    u = [initial_condition(xi) for xi in x]
    u[0] = left_bc(0.0)
    u[-1] = right_bc(0.0)

    n_interior = n_space - 1
    lower = [0.0] + [-r / 2] * (n_interior - 1)
    diag = [1 + r] * n_interior
    upper = [-r / 2] * (n_interior - 1) + [0.0]

    for n in range(n_time):
        t_next = (n + 1) * dt
        left_val = left_bc(t_next)
        right_val = right_bc(t_next)

        rhs = []
        for i in range(1, n_space):
            explicit_part = (r / 2) * u[i - 1] + (1 - r) * u[i] + (r / 2) * u[i + 1]
            rhs.append(explicit_part)
        rhs[0] += (r / 2) * left_val
        rhs[-1] += (r / 2) * right_val

        interior = la.thomas_algorithm(lower, diag, upper, rhs)
        u = [left_val] + interior + [right_val]

    return {"x": x, "u": u, "r": r, "dx": dx, "dt": dt}


def exact_sinusoidal_solution(mode: int, alpha: float, length: float) -> Callable[[float, float], float]:
    """The exact solution of the heat equation on [0,length] with
    homogeneous Dirichlet boundaries (u(0,t)=u(length,t)=0) and initial
    condition u(x,0)=sin(mode*pi*x/length): separation of variables
    gives u(x,t)=sin(mode*pi*x/length)*exp(-alpha*(mode*pi/length)^2*t)
    directly, since substituting this ansatz into the PDE reduces it
    to an ordinary exponential-decay ODE in time for the sinusoid's
    amplitude, with decay rate alpha*(mode*pi/length)^2 read off
    straight from matching the spatial second derivative
    -(mode*pi/length)^2*sin(...) against the time derivative.
    """
    k = mode * math.pi / length

    def u(x: float, t: float) -> float:
        return math.sin(k * x) * math.exp(-alpha * k ** 2 * t)

    return u
