import math
from typing import Callable, Dict

import linear_algebra as la


def solve_wave_explicit(c: float, length: float, t_end: float, n_space: int, n_time: int,
                         initial_displacement: Callable[[float], float],
                         initial_velocity: Callable[[float], float],
                         left_bc: Callable[[float], float], right_bc: Callable[[float], float]) -> Dict:
    """Solve d^2u/dt^2 = c^2*d^2u/dx^2 on [0,length] via the standard
    explicit central-difference ("leapfrog") scheme: replacing both
    time and space second derivatives with centred second differences
    gives

        u_i^{n+1} = 2*u_i^n - u_i^{n-1} + C^2*(u_{i+1}^n - 2*u_i^n + u_{i-1}^n),

    where C = c*dt/dx is the Courant number, a direct analogue of the
    heat equation's r but for a second-order-in-time equation. This
    recursion needs two previous time levels, so the very first step
    (from u^0 to u^1) is special: no u^{-1} exists yet. It is obtained
    instead from a Taylor expansion of u(x,dt) using the given initial
    velocity g(x) directly,

        u_i^1 = u_i^0 + dt*g(x_i) + 0.5*C^2*(u_{i+1}^0 - 2*u_i^0 + u_{i-1}^0),

    which is itself second-order accurate in dt (matching the scheme's
    own overall accuracy) since it keeps the Taylor expansion's
    quadratic term via the PDE itself (substituting
    d^2u/dt^2=c^2*d^2u/dx^2 at t=0 for the u_tt term a plain
    first-order Taylor step would otherwise have to drop).

    Von Neumann stability analysis of this scheme gives the CFL
    condition C<=1: the wave equation's amplification factor has a
    different, quadratic-in-xi characteristic equation from the heat
    equation's (see stability_analysis.py for the heat equation's own
    derivation), but the same underlying method applies and gives this
    well-known Courant condition for hyperbolic PDEs.
    """
    dx = length / n_space
    dt = t_end / n_time
    courant = c * dt / dx

    x = la.linspace(0.0, length, n_space + 1)
    u_prev = [initial_displacement(xi) for xi in x]
    u_prev[0] = left_bc(0.0)
    u_prev[-1] = right_bc(0.0)

    u_curr = [0.0] * (n_space + 1)
    u_curr[0] = left_bc(dt)
    u_curr[-1] = right_bc(dt)
    for i in range(1, n_space):
        u_curr[i] = (u_prev[i] + dt * initial_velocity(x[i])
                     + 0.5 * courant ** 2 * (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1]))

    for n in range(1, n_time):
        t_next = (n + 1) * dt
        u_next = [0.0] * (n_space + 1)
        u_next[0] = left_bc(t_next)
        u_next[n_space] = right_bc(t_next)
        for i in range(1, n_space):
            u_next[i] = (2 * u_curr[i] - u_prev[i]
                         + courant ** 2 * (u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]))
        u_prev, u_curr = u_curr, u_next

    return {"x": x, "u": u_curr, "courant": courant, "dx": dx, "dt": dt}


def exact_standing_wave_solution(mode: int, c: float, length: float) -> Callable[[float, float], float]:
    """The exact solution for the wave equation on [0,length] with
    homogeneous Dirichlet boundaries, initial displacement
    u(x,0)=sin(mode*pi*x/length), and zero initial velocity: separation
    of variables gives u(x,t)=sin(mode*pi*x/length)*cos(mode*pi*c*t/length),
    a standing wave whose spatial shape never changes, only oscillating
    in amplitude at angular frequency mode*pi*c/length. Substituting
    this ansatz into the PDE directly confirms it: both
    d^2u/dt^2=-(mode*pi*c/length)^2*u and
    c^2*d^2u/dx^2=-c^2*(mode*pi/length)^2*u match exactly.
    """
    k = mode * math.pi / length
    omega = k * c

    def u(x: float, t: float) -> float:
        return math.sin(k * x) * math.cos(omega * t)

    return u


def exact_traveling_wave_dalembert(f: Callable[[float], float], c: float) -> Callable[[float, float], float]:
    """D'Alembert's solution for the wave equation on an (effectively)
    infinite domain with initial displacement f(x) and zero initial
    velocity: u(x,t) = 0.5*(f(x-c*t) + f(x+c*t)), a direct consequence
    of the wave equation's general solution being any sum of a
    right-moving and a left-moving profile, F(x-ct)+G(x+ct); matching
    both the initial displacement f(x)=F(x)+G(x) and the zero initial
    velocity 0=-c*F'(x)+c*G'(x) (which forces F=G up to a constant)
    pins F=G=f/2 exactly.
    """
    def u(x: float, t: float) -> float:
        return 0.5 * (f(x - c * t) + f(x + c * t))

    return u
