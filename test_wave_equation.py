import math
import unittest

import wave_equation as we
import linear_algebra as la


class TestSolveWaveExplicitStandingWave(unittest.TestCase):
    def test_matches_exact_solution_when_stable(self):
        c, length, t_end, mode = 1.0, 1.0, 0.5, 1
        ic_disp = lambda x: math.sin(mode * math.pi * x / length)
        ic_vel = lambda x: 0.0
        left_bc = lambda t: 0.0
        right_bc = lambda t: 0.0
        exact = we.exact_standing_wave_solution(mode, c, length)

        result = we.solve_wave_explicit(c, length, t_end, 100, 200, ic_disp, ic_vel, left_bc, right_bc)
        self.assertLessEqual(result["courant"], 1.0)
        exact_u = [exact(x, t_end) for x in result["x"]]
        self.assertLess(la.max_abs_diff(result["u"], exact_u), 0.001)

    def test_blows_up_when_courant_exceeds_one(self):
        c, length, t_end, mode = 1.0, 1.0, 0.5, 1
        ic_disp = lambda x: math.sin(mode * math.pi * x / length)
        ic_vel = lambda x: 0.0
        left_bc = lambda t: 0.0
        right_bc = lambda t: 0.0

        result = we.solve_wave_explicit(c, length, t_end, 100, 40, ic_disp, ic_vel, left_bc, right_bc)
        self.assertGreater(result["courant"], 1.0)
        self.assertGreater(max(abs(v) for v in result["u"]), 1000.0)

    def test_boundary_conditions_enforced(self):
        c, length, t_end, mode = 1.0, 1.0, 0.5, 2
        ic_disp = lambda x: math.sin(mode * math.pi * x / length)
        ic_vel = lambda x: 0.0
        result = we.solve_wave_explicit(c, length, t_end, 50, 100, ic_disp, ic_vel,
                                         lambda t: 0.0, lambda t: 0.0)
        self.assertAlmostEqual(result["u"][0], 0.0, places=8)
        self.assertAlmostEqual(result["u"][-1], 0.0, places=8)

    def test_stationary_zero_solution_stays_zero(self):
        c, length, t_end = 1.0, 1.0, 1.0
        zero = lambda x: 0.0
        result = we.solve_wave_explicit(c, length, t_end, 50, 100, zero, zero, lambda t: 0.0, lambda t: 0.0)
        for u in result["u"]:
            self.assertAlmostEqual(u, 0.0, places=10)


class TestExactStandingWaveSolution(unittest.TestCase):
    def test_satisfies_initial_displacement(self):
        c, length, mode = 2.0, 3.0, 2
        exact = we.exact_standing_wave_solution(mode, c, length)
        for x in [0.0, 0.5, 1.0, 2.0, 3.0]:
            expected = math.sin(mode * math.pi * x / length)
            self.assertAlmostEqual(exact(x, 0.0), expected, places=10)

    def test_satisfies_boundary_conditions(self):
        c, length, mode = 2.0, 3.0, 3
        exact = we.exact_standing_wave_solution(mode, c, length)
        for t in [0.0, 0.5, 1.0]:
            self.assertAlmostEqual(exact(0.0, t), 0.0, places=8)
            self.assertAlmostEqual(exact(length, t), 0.0, places=8)

    def test_amplitude_never_exceeds_one(self):
        c, length, mode = 1.5, 2.0, 4
        exact = we.exact_standing_wave_solution(mode, c, length)
        for x in [0.1, 0.5, 1.0, 1.5, 1.9]:
            for t in [0.1, 0.7, 1.3, 2.5]:
                self.assertLessEqual(abs(exact(x, t)), 1.0 + 1e-10)


class TestDalembertSolution(unittest.TestCase):
    def test_matches_initial_condition_at_t_zero(self):
        f = lambda x: math.exp(-((x - 5) ** 2))
        c = 2.0
        u = we.exact_traveling_wave_dalembert(f, c)
        for x in [3.0, 4.0, 5.0, 6.0, 7.0]:
            self.assertAlmostEqual(u(x, 0.0), f(x), places=10)

    def test_pulse_splits_into_two_half_amplitude_peaks(self):
        f = lambda x: math.exp(-((x - 5) ** 2))
        c = 2.0
        u = we.exact_traveling_wave_dalembert(f, c)
        self.assertAlmostEqual(u(3.0, 1.0), 0.5, places=3)
        self.assertAlmostEqual(u(7.0, 1.0), 0.5, places=3)

    def test_conserves_total_energy_shape_symmetrically(self):
        f = lambda x: math.exp(-((x - 5) ** 2))
        c = 1.0
        u = we.exact_traveling_wave_dalembert(f, c)
        for d in [0.5, 1.0, 2.0]:
            self.assertAlmostEqual(u(5 - d, 0.7), u(5 + d, 0.7), places=10)


if __name__ == "__main__":
    unittest.main()
