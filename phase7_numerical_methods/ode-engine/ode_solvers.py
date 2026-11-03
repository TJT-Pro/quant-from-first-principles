import math
from typing import Callable, List, Dict, Union

State = Union[float, List[float]]


def _add(y: List[float], scale: float, k: List[float]) -> List[float]:
    return [yi + scale * ki for yi, ki in zip(y, k)]


def _to_vector(y: State) -> List[float]:
    return [y] if isinstance(y, (int, float)) else list(y)


def _from_vector(y: List[float], was_scalar: bool):
    return y[0] if was_scalar else y


def euler_step(f: Callable, t: float, y: List[float], h: float) -> List[float]:
    """Forward Euler: y_{n+1} = y_n + h*f(t_n, y_n).

    Geometrically, this walks along the tangent line to the true
    solution at (t_n, y_n) for a distance h. Taylor's theorem gives
    y(t_n+h) = y(t_n) + h*y'(t_n) + O(h^2), so Euler's step matches the
    true solution through the first-order term and drops an O(h^2)
    term at each step: the *local* truncation error (one step, exact
    starting point) is O(h^2), and accumulating that error over the
    O(1/h) steps needed to cover a fixed interval gives a *global*
    error of O(h): halving h should roughly halve the total error,
    which `error_analysis.py` verifies directly rather than only
    stating.
    """
    fy = f(t, y)
    return _add(y, h, fy)


def improved_euler_step(f: Callable, t: float, y: List[float], h: float) -> List[float]:
    """Improved Euler / Heun's method: predict with a plain Euler step,
    then average the slope at the start and at the predicted endpoint:

        y_predict = y_n + h*f(t_n, y_n)
        y_{n+1} = y_n + (h/2)*[f(t_n, y_n) + f(t_n+h, y_predict)]

    This is the simplest instance of a predictor-corrector scheme, and
    is equivalent to approximating the integral of f over [t_n, t_n+h]
    by the trapezoidal rule using an Euler-predicted endpoint slope.
    Averaging two slope estimates cancels the leading error term that
    plain Euler leaves behind, raising the local truncation error to
    O(h^3) and the global error to O(h^2).
    """
    k1 = f(t, y)
    y_predict = _add(y, h, k1)
    k2 = f(t + h, y_predict)
    avg = [(a + b) / 2 for a, b in zip(k1, k2)]
    return _add(y, h, avg)


def midpoint_step(f: Callable, t: float, y: List[float], h: float) -> List[float]:
    """The midpoint method: estimate the slope at the interval's
    midpoint (using a half-Euler-step to get there) and use that
    single slope estimate for the whole step:

        k1 = f(t_n, y_n)
        y_mid = y_n + (h/2)*k1
        y_{n+1} = y_n + h*f(t_n + h/2, y_mid)

    Also O(h^2) globally, by a similar cancellation argument to
    Improved Euler but using the midpoint rule for the integral
    approximation instead of the trapezoidal rule; the two methods
    have the same order but generally different constants in front of
    the leading error term, which is why they do not give identical
    numerical answers even though both are "the" standard second-order
    Runge-Kutta method in different textbooks' conventions.
    """
    k1 = f(t, y)
    y_mid = _add(y, h / 2, k1)
    k2 = f(t + h / 2, y_mid)
    return _add(y, h, k2)


def rk4_step(f: Callable, t: float, y: List[float], h: float) -> List[float]:
    """The classical fourth-order Runge-Kutta method: four slope
    evaluations per step, combined with Simpson's-rule-like weights:

        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y_{n+1} = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    k1 and k4 estimate the slope at the interval's two endpoints, k2
    and k3 both estimate it at the midpoint (using different
    predicted positions), and the 1:2:2:1 weighting matches Simpson's
    rule for numerically integrating f over the step. This cancels
    error terms through fourth order, giving a global error of O(h^4):
    halving h should reduce the total error by a factor of 16, by far
    the fastest convergence of the four fixed-step methods here, at
    the cost of four function evaluations per step instead of one or
    two.
    """
    k1 = f(t, y)
    k2 = f(t + h / 2, _add(y, h / 2, k1))
    k3 = f(t + h / 2, _add(y, h / 2, k2))
    k4 = f(t + h, _add(y, h, k3))
    combined = [(a + 2 * b + 2 * c + d) / 6 for a, b, c, d in zip(k1, k2, k3, k4)]
    return _add(y, h, combined)


_STEPPERS = {
    "euler": euler_step,
    "improved_euler": improved_euler_step,
    "midpoint": midpoint_step,
    "rk4": rk4_step,
}


def solve_ivp(f: Callable[[float, State], State], t0: float, y0: State, t_end: float,
              h: float, method: str = "rk4") -> Dict[str, List]:
    """Fixed-step integration of dy/dt = f(t,y), y(t0) = y0, from t0 to
    t_end using the requested method, accepting either a scalar or a
    vector-valued y0 (a system of ODEs) transparently: internally,
    everything is handled as a list, and the result is converted back
    to scalars at the end if the input was scalar, so a single
    implementation of each stepper serves both cases without
    duplicated code.

    When (t_end - t0) is not an exact multiple of h, the requested
    step size is used for every full step and a single shorter final
    step exactly closes the remaining gap, landing precisely on
    t_end. An earlier version of this function instead took
    round((t_end-t0)/h) steps of the *full* size h and stopped,
    silently overshooting or undershooting t_end by up to h/2 whenever
    the interval was not an exact multiple of the step size (found
    directly by testing: a harmonic-oscillator system integrated to
    t_end=pi with h=0.001 landed at t=3.142 instead of pi, an error of
    about 4e-4 that looked like a solver accuracy problem but was
    actually the wrong endpoint being evaluated in the first place).
    Every solver in this project should be judged against the value it
    actually reports having reached, not the value it was asked to
    reach; this fix makes those the same value.
    """
    if method not in _STEPPERS:
        raise ValueError(f"unknown method: {method}")
    stepper = _STEPPERS[method]
    was_scalar = isinstance(y0, (int, float))

    def f_vec(t, y):
        result = f(t, y[0] if was_scalar else y)
        return _to_vector(result)

    total = t_end - t0
    n_full_steps = int(total / h + 1e-9)
    remainder = total - n_full_steps * h

    ts = [t0]
    ys = [_to_vector(y0)]
    t, y = t0, _to_vector(y0)
    for _ in range(n_full_steps):
        y = stepper(f_vec, t, y, h)
        t += h
        ts.append(t)
        ys.append(y)
    if remainder > 1e-12:
        y = stepper(f_vec, t, y, remainder)
        t = t_end
        ts.append(t)
        ys.append(y)

    return {"t": ts, "y": [_from_vector(yi, was_scalar) for yi in ys]}


def solve_ivp_adaptive(f: Callable[[float, State], State], t0: float, y0: State, t_end: float,
                        h_init: float, tol: float = 1e-6, h_min: float = 1e-8,
                        h_max: float = 1.0, method: str = "rk4") -> Dict[str, List]:
    """Adaptive step-size control via step-doubling (Richardson
    extrapolation): to advance from t to t+2h, take one step of size
    2h and, separately, two steps of size h, and compare the two
    results. For a method of order p, the step-of-size-H global error
    scales like C*H^p for some constant C depending on the true
    solution's higher derivatives (not on H itself); the two estimates
    y_big (one step of 2h) and y_small (two steps of h) therefore
    differ by approximately C*(2h)^p - 2*C*h^p = C*h^p*(2^p - 2),
    which gives a usable estimate of the error in y_small without
    knowing the true solution: error ~ |y_small - y_big| / (2^p - 1).
    If that estimated error exceeds the requested tolerance, the step
    is rejected and retried with a smaller h; if it comfortably beats
    the tolerance, the next step is grown, both via the standard
    proportional-control formula h_new = h * safety * (tol/error)^(1/(p+1)).
    """
    orders = {"euler": 1, "improved_euler": 2, "midpoint": 2, "rk4": 4}
    if method not in orders:
        raise ValueError(f"unknown method: {method}")
    p = orders[method]
    stepper = _STEPPERS[method]
    was_scalar = isinstance(y0, (int, float))

    def f_vec(t, y):
        result = f(t, y[0] if was_scalar else y)
        return _to_vector(result)

    t = t0
    y = _to_vector(y0)
    h = h_init
    ts, ys, hs = [t0], [y], []

    safety = 0.9
    while t < t_end - 1e-12:
        h = min(h, t_end - t)
        while True:
            y_big = stepper(f_vec, t, y, h)
            y_half = stepper(f_vec, t, y, h / 2)
            y_small = stepper(f_vec, t + h / 2, y_half, h / 2)

            error = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_small, y_big))) / (2 ** p - 1)
            error = max(error, 1e-300)

            if error <= tol or h <= h_min:
                t += h
                y = y_small
                ts.append(t)
                ys.append(y)
                hs.append(h)
                h_next = h * safety * (tol / error) ** (1.0 / (p + 1))
                h = min(max(h_next, h_min), h_max)
                break
            else:
                h = max(h * safety * (tol / error) ** (1.0 / (p + 1)), h_min)

    return {"t": ts, "y": [_from_vector(yi, was_scalar) for yi in ys], "step_sizes": hs}
