import math

import heat_equation as he


def explicit_growth_factor(r: float, theta: float) -> float:
    """The von Neumann amplification factor for the explicit scheme:
    substituting a single Fourier mode u_i^n = xi^n * exp(i*k*x_i)
    (theta = k*dx being the mode's phase advance per grid spacing)
    into the explicit update and dividing through by the common factor
    xi^n * exp(i*k*x_i) gives

        xi = 1 + r*(e^{i*theta} - 2 + e^{-i*theta}) = 1 - 4*r*sin^2(theta/2),

    using e^{i*theta}+e^{-i*theta}-2 = 2cos(theta)-2 = -4sin^2(theta/2).
    A mode is stable (does not grow without bound as n increases) if
    and only if |xi|<=1; since sin^2(theta/2) ranges over [0,1] across
    all possible modes theta, the worst case is sin^2(theta/2)=1,
    requiring 1-4r >= -1, i.e. r<=0.5: the exact CFL-type stability
    bound for this scheme, derived here rather than only quoted.
    """
    return 1 - 4 * r * math.sin(theta / 2) ** 2


def implicit_growth_factor(r: float, theta: float) -> float:
    """The amplification factor for the implicit (backward Euler)
    scheme: substituting the same Fourier mode ansatz into the
    implicit update, but now with the spatial term evaluated at the
    new time level, gives

        xi * (1 + 4*r*sin^2(theta/2)) = 1   =>   xi = 1 / (1 + 4*r*sin^2(theta/2)).

    For any r>0, the denominator is strictly greater than or equal to
    1, so 0 < xi <= 1 for every mode theta: the scheme is
    unconditionally stable, with no restriction on r (equivalently, on
    the time step dt) at all.
    """
    return 1.0 / (1 + 4 * r * math.sin(theta / 2) ** 2)


def crank_nicolson_growth_factor(r: float, theta: float) -> float:
    """The amplification factor for Crank-Nicolson: averaging the
    explicit scheme's spatial operator (evaluated at the old level)
    and the implicit scheme's (evaluated at the new level) gives

        xi = (1 - 2*r*sin^2(theta/2)) / (1 + 2*r*sin^2(theta/2)).

    Since sin^2(theta/2) >= 0 and r >= 0, both numerator and
    denominator magnitudes satisfy |numerator| <= denominator for
    every theta and every r >= 0 (the numerator can only be made
    smaller in magnitude than the denominator, never larger, as
    2*r*sin^2(theta/2) is subtracted on top rather than added), so
    |xi| <= 1 unconditionally: Crank-Nicolson is unconditionally
    stable for any r, exactly like the implicit scheme, but (per
    heat_equation.py) with a smaller local truncation error.
    """
    s = 2 * r * math.sin(theta / 2) ** 2
    return (1 - s) / (1 + s)


def max_growth_factor_over_modes(growth_factor_fn, r: float, n_modes: int = 200) -> float:
    """The largest |amplification factor| across a fine sampling of
    every possible Fourier mode theta in [0, pi] (by symmetry, modes
    in [pi, 2*pi] mirror those in [0, pi] for a real-valued scheme, so
    sampling only the first half already covers the worst case): this
    is the single number that determines whether a scheme is stable
    (max <= 1) or unstable (max > 1) at a given r.
    """
    thetas = [math.pi * k / n_modes for k in range(n_modes + 1)]
    return max(abs(growth_factor_fn(r, theta)) for theta in thetas)


def empirically_measure_mode_growth(alpha: float, length: float, n_space: int, r_target: float,
                                     mode: int) -> dict:
    """A direct empirical check of the von Neumann prediction: run the
    actual explicit finite-difference scheme starting from a single
    pure Fourier mode as the initial condition, and measure the ratio
    of the mode's amplitude after one time step to its amplitude
    before, which should match the theoretical growth factor exactly
    (up to the boundary effects introduced by this being a finite
    domain with Dirichlet ends rather than the infinite, boundary-free
    domain the von Neumann analysis technically assumes, kept small
    here by using a mode number small relative to n_space, and checked
    against a comfortably loose tolerance for exactly this reason).
    """
    dx = length / n_space
    theta = mode * math.pi / n_space
    ic = lambda x: math.sin(mode * math.pi * x / length)
    dt = r_target * dx ** 2 / alpha

    result = he.solve_heat_explicit(alpha, length, dt, n_space, 1, ic, lambda t: 0.0, lambda t: 0.0)
    amplitude_before = max(abs(v) for v in [ic(x) for x in result["x"]])
    amplitude_after = max(abs(v) for v in result["u"])
    empirical_ratio = amplitude_after / amplitude_before if amplitude_before > 0 else float("nan")

    theoretical = explicit_growth_factor(r_target, theta)
    return {"empirical_ratio": empirical_ratio, "theoretical_growth_factor": abs(theoretical), "r": r_target}
