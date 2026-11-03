import math
import unittest

import heat_equation as he
import linear_algebra as la


def sinusoidal_setup(mode=1, alpha=0.5, length=1.0):
    ic = lambda x: math.sin(mode * math.pi * x / length)
    left_bc = lambda t: 0.0
    right_bc = lambda t: 0.0
    exact = he.exact_sinusoidal_solution(mode, alpha, length)
    return ic, left_bc, right_bc, exact


class TestExplicitScheme(unittest.TestCase):
    def test_matches_exact_solution_when_stable(self):
        alpha, length, t_end = 0.5, 1.0, 0.1
        ic, left_bc, right_bc, exact = sinusoidal_setup(alpha=alpha, length=length)
        n_space, n_time = 50, 300
        result = he.solve_heat_explicit(alpha, length, t_end, n_space, n_time, ic, left_bc, right_bc)
        self.assertLess(result["r"], 0.5)
        exact_u = [exact(x, t_end) for x in result["x"]]
        self.assertLess(la.max_abs_diff(result["u"], exact_u), 0.01)

    def test_boundary_conditions_are_enforced(self):
        alpha, length, t_end = 0.5, 1.0, 0.1
        ic, left_bc, right_bc, _ = sinusoidal_setup(alpha=alpha, length=length)
        result = he.solve_heat_explicit(alpha, length, t_end, 50, 300, ic, left_bc, right_bc)
        self.assertAlmostEqual(result["u"][0], 0.0, places=8)
        self.assertAlmostEqual(result["u"][-1], 0.0, places=8)

    def test_nonzero_dirichlet_boundaries(self):
        alpha, length, t_end = 0.5, 1.0, 0.2
        ic = lambda x: 5.0
        left_bc = lambda t: 5.0
        right_bc = lambda t: 5.0
        result = he.solve_heat_explicit(alpha, length, t_end, 20, 500, ic, left_bc, right_bc)
        for u in result["u"]:
            self.assertAlmostEqual(u, 5.0, places=6)


class TestImplicitScheme(unittest.TestCase):
    def test_matches_exact_solution(self):
        alpha, length, t_end = 0.5, 1.0, 0.1
        ic, left_bc, right_bc, exact = sinusoidal_setup(alpha=alpha, length=length)
        result = he.solve_heat_implicit(alpha, length, t_end, 50, 100, ic, left_bc, right_bc)
        exact_u = [exact(x, t_end) for x in result["x"]]
        self.assertLess(la.max_abs_diff(result["u"], exact_u), 0.01)

    def test_remains_stable_at_large_r(self):
        alpha, length, t_end = 0.5, 1.0, 0.1
        ic, left_bc, right_bc, exact = sinusoidal_setup(alpha=alpha, length=length)
        result = he.solve_heat_implicit(alpha, length, t_end, 50, 20, ic, left_bc, right_bc)
        self.assertGreater(result["r"], 0.5)
        exact_u = [exact(x, t_end) for x in result["x"]]
        self.assertLess(la.max_abs_diff(result["u"], exact_u), 0.05)
        for u in result["u"]:
            self.assertLess(abs(u), 10.0)


class TestCrankNicolsonScheme(unittest.TestCase):
    def test_matches_exact_solution(self):
        alpha, length, t_end = 0.5, 1.0, 0.1
        ic, left_bc, right_bc, exact = sinusoidal_setup(alpha=alpha, length=length)
        result = he.solve_heat_crank_nicolson(alpha, length, t_end, 50, 100, ic, left_bc, right_bc)
        exact_u = [exact(x, t_end) for x in result["x"]]
        self.assertLess(la.max_abs_diff(result["u"], exact_u), 0.01)

    def test_more_accurate_than_implicit_at_same_grid(self):
        alpha, length, t_end = 0.5, 1.0, 0.1
        ic, left_bc, right_bc, exact = sinusoidal_setup(alpha=alpha, length=length)
        n_space, n_time = 50, 200
        cn_result = he.solve_heat_crank_nicolson(alpha, length, t_end, n_space, n_time, ic, left_bc, right_bc)
        imp_result = he.solve_heat_implicit(alpha, length, t_end, n_space, n_time, ic, left_bc, right_bc)
        exact_u = [exact(x, t_end) for x in cn_result["x"]]
        cn_err = la.max_abs_diff(cn_result["u"], exact_u)
        imp_err = la.max_abs_diff(imp_result["u"], exact_u)
        self.assertLess(cn_err, imp_err)

    def test_remains_stable_at_large_r(self):
        alpha, length, t_end = 0.5, 1.0, 0.1
        ic, left_bc, right_bc, exact = sinusoidal_setup(alpha=alpha, length=length)
        result = he.solve_heat_crank_nicolson(alpha, length, t_end, 50, 20, ic, left_bc, right_bc)
        self.assertGreater(result["r"], 0.5)
        for u in result["u"]:
            self.assertLess(abs(u), 10.0)


class TestExactSinusoidalSolution(unittest.TestCase):
    def test_satisfies_initial_condition(self):
        alpha, length, mode = 0.3, 2.0, 2
        exact = he.exact_sinusoidal_solution(mode, alpha, length)
        for x in [0.0, 0.5, 1.0, 1.5, 2.0]:
            expected = math.sin(mode * math.pi * x / length)
            self.assertAlmostEqual(exact(x, 0.0), expected, places=10)

    def test_satisfies_boundary_conditions(self):
        alpha, length, mode = 0.3, 2.0, 3
        exact = he.exact_sinusoidal_solution(mode, alpha, length)
        for t in [0.0, 0.1, 1.0]:
            self.assertAlmostEqual(exact(0.0, t), 0.0, places=8)
            self.assertAlmostEqual(exact(length, t), 0.0, places=8)

    def test_decays_over_time(self):
        alpha, length, mode = 0.5, 1.0, 1
        exact = he.exact_sinusoidal_solution(mode, alpha, length)
        early = abs(exact(0.5, 0.01))
        late = abs(exact(0.5, 1.0))
        self.assertGreater(early, late)


if __name__ == "__main__":
    unittest.main()
