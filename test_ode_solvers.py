import math
import unittest

import ode_solvers as ode


class TestFixedStepMethodsOnExponentialDecay(unittest.TestCase):
    def setUp(self):
        self.f = lambda t, y: -y
        self.true_solution = lambda t: math.exp(-t)

    def test_euler_converges_with_small_h(self):
        result = ode.solve_ivp(self.f, 0.0, 1.0, 1.0, h=0.001, method="euler")
        self.assertAlmostEqual(result["y"][-1], self.true_solution(1.0), places=2)

    def test_rk4_much_more_accurate_than_euler_at_same_h(self):
        h = 0.05
        euler_result = ode.solve_ivp(self.f, 0.0, 1.0, 1.0, h, method="euler")
        rk4_result = ode.solve_ivp(self.f, 0.0, 1.0, 1.0, h, method="rk4")
        true_val = self.true_solution(1.0)
        euler_err = abs(euler_result["y"][-1] - true_val)
        rk4_err = abs(rk4_result["y"][-1] - true_val)
        self.assertLess(rk4_err, euler_err / 100)

    def test_all_methods_agree_reasonably_at_small_h(self):
        h = 0.001
        results = {m: ode.solve_ivp(self.f, 0.0, 1.0, 1.0, h, method=m)["y"][-1]
                   for m in ("euler", "improved_euler", "midpoint", "rk4")}
        true_val = self.true_solution(1.0)
        # Euler is only first-order: at h=0.001 its global error is
        # itself on the order of 1e-4, so it needs a looser tolerance
        # here than the higher-order methods, not a test bug to paper
        # over with a single shared tolerance for all four.
        self.assertAlmostEqual(results["euler"], true_val, places=3)
        for method in ("improved_euler", "midpoint", "rk4"):
            self.assertAlmostEqual(results[method], true_val, places=6, msg=method)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            ode.solve_ivp(self.f, 0.0, 1.0, 1.0, 0.1, method="nonsense")

    def test_time_grid_spans_full_interval(self):
        result = ode.solve_ivp(self.f, 0.0, 1.0, 2.0, h=0.1, method="rk4")
        self.assertAlmostEqual(result["t"][0], 0.0)
        self.assertAlmostEqual(result["t"][-1], 2.0, places=8)


class TestExactPolynomialCase(unittest.TestCase):
    def test_rk4_exact_for_cubic_solution(self):
        # y' = 3t^2, y(0) = 0 has exact solution y = t^3, a cubic that
        # RK4 (exact through degree 4 in the underlying Taylor
        # expansion for polynomial right-hand sides like this) should
        # reproduce to machine precision regardless of step size.
        f = lambda t, y: 3 * t ** 2
        result = ode.solve_ivp(f, 0.0, 0.0, 2.0, h=0.3, method="rk4")
        self.assertAlmostEqual(result["y"][-1], 2.0 ** 3, places=8)


class TestSystemsOfODEs(unittest.TestCase):
    def test_harmonic_oscillator_matches_known_solution(self):
        # y'' + y = 0, y(0)=1, y'(0)=0 has exact solution y(t) = cos(t).
        # As a first-order system: y1'=y2, y2'=-y1.
        f = lambda t, y: [y[1], -y[0]]
        result = ode.solve_ivp(f, 0.0, [1.0, 0.0], math.pi, h=0.001, method="rk4")
        self.assertAlmostEqual(result["y"][-1][0], math.cos(math.pi), places=6)
        self.assertAlmostEqual(result["y"][-1][1], -math.sin(math.pi), places=6)

    def test_two_dimensional_linear_system(self):
        # y1'=y2, y2'=y1, y1(0)=1, y2(0)=1 has solution y1=y2=e^t.
        f = lambda t, y: [y[1], y[0]]
        result = ode.solve_ivp(f, 0.0, [1.0, 1.0], 1.0, h=0.001, method="rk4")
        expected = math.exp(1.0)
        self.assertAlmostEqual(result["y"][-1][0], expected, places=5)
        self.assertAlmostEqual(result["y"][-1][1], expected, places=5)

    def test_scalar_and_vector_paths_agree_for_one_dimensional_system(self):
        f_scalar = lambda t, y: -2 * y
        f_vector = lambda t, y: [-2 * y[0]]
        scalar_result = ode.solve_ivp(f_scalar, 0.0, 1.0, 1.0, h=0.01, method="rk4")
        vector_result = ode.solve_ivp(f_vector, 0.0, [1.0], 1.0, h=0.01, method="rk4")
        self.assertAlmostEqual(scalar_result["y"][-1], vector_result["y"][-1][0], places=10)


class TestAdaptiveStepSize(unittest.TestCase):
    def test_converges_to_true_solution(self):
        f = lambda t, y: -y
        result = ode.solve_ivp_adaptive(f, 0.0, 1.0, 2.0, h_init=0.1, tol=1e-8, method="rk4")
        self.assertAlmostEqual(result["y"][-1], math.exp(-2.0), places=6)

    def test_uses_fewer_steps_than_a_conservative_fixed_grid(self):
        f = lambda t, y: -y
        adaptive = ode.solve_ivp_adaptive(f, 0.0, 1.0, 5.0, h_init=0.1, tol=1e-6, method="rk4")
        fixed = ode.solve_ivp(f, 0.0, 1.0, 5.0, h=0.001, method="rk4")
        self.assertLess(len(adaptive["t"]), len(fixed["t"]))

    def test_tighter_tolerance_gives_smaller_error(self):
        f = lambda t, y: -y
        loose = ode.solve_ivp_adaptive(f, 0.0, 1.0, 2.0, h_init=0.1, tol=1e-3, method="rk4")
        tight = ode.solve_ivp_adaptive(f, 0.0, 1.0, 2.0, h_init=0.1, tol=1e-10, method="rk4")
        true_val = math.exp(-2.0)
        loose_err = abs(loose["y"][-1] - true_val)
        tight_err = abs(tight["y"][-1] - true_val)
        self.assertLess(tight_err, loose_err)

    def test_step_sizes_are_within_bounds(self):
        f = lambda t, y: -y
        result = ode.solve_ivp_adaptive(f, 0.0, 1.0, 2.0, h_init=0.1, tol=1e-6,
                                         h_min=1e-6, h_max=0.5, method="rk4")
        for h in result["step_sizes"]:
            self.assertGreaterEqual(h, 1e-6 * 0.99)
            self.assertLessEqual(h, 0.5 * 1.01)

    def test_reaches_exactly_t_end(self):
        f = lambda t, y: -y
        result = ode.solve_ivp_adaptive(f, 0.0, 1.0, 3.7, h_init=0.2, tol=1e-6, method="rk4")
        self.assertAlmostEqual(result["t"][-1], 3.7, places=8)

    def test_unknown_method_raises(self):
        f = lambda t, y: -y
        with self.assertRaises(ValueError):
            ode.solve_ivp_adaptive(f, 0.0, 1.0, 1.0, 0.1, method="nonsense")


if __name__ == "__main__":
    unittest.main()
