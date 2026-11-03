import math
import unittest

from optimization import nelder_mead


class TestNelderMead(unittest.TestCase):
    def test_minimizes_simple_quadratic(self):
        f = lambda x: (x[0] - 3.0) ** 2 + (x[1] + 2.0) ** 2
        result = nelder_mead(f, [0.0, 0.0])
        self.assertAlmostEqual(result["x"][0], 3.0, places=3)
        self.assertAlmostEqual(result["x"][1], -2.0, places=3)
        self.assertLess(result["fun"], 1e-6)

    def test_minimizes_rosenbrock_like_function(self):
        # A classic hard-to-optimize test function; loosen tolerance
        # since Nelder-Mead converges slowly along its curved valley.
        f = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        result = nelder_mead(f, [-1.0, 1.0], max_iter=5000)
        self.assertAlmostEqual(result["x"][0], 1.0, delta=0.05)
        self.assertAlmostEqual(result["x"][1], 1.0, delta=0.05)

    def test_one_dimensional_minimization(self):
        f = lambda x: (x[0] - 5.0) ** 2 + 7.0
        result = nelder_mead(f, [0.0])
        self.assertAlmostEqual(result["x"][0], 5.0, places=3)
        self.assertAlmostEqual(result["fun"], 7.0, places=4)

    def test_does_not_stop_early_on_symmetric_function_value_coincidence(self):
        # Regression test for a real bug found during development: two
        # points straddling the minimum of an even function can have
        # nearly identical function values while still being far from
        # the minimum in x. A stopping rule based on function-value
        # spread alone declared "converged" at x=4.9 for this exact
        # function after only 10 iterations; the fix requires the
        # simplex to also have converged in parameter space.
        f = lambda x: (x[0] - 5.0) ** 2 + 7.0
        result = nelder_mead(f, [0.0])
        self.assertGreater(result["iterations"], 10)
        self.assertLess(abs(result["x"][0] - 5.0), 1e-3)

    def test_returns_iteration_count(self):
        f = lambda x: x[0] ** 2
        result = nelder_mead(f, [10.0])
        self.assertGreater(result["iterations"], 0)

    def test_already_at_minimum(self):
        f = lambda x: x[0] ** 2 + x[1] ** 2
        result = nelder_mead(f, [0.0, 0.0], step=1e-8)
        self.assertLess(result["fun"], 1e-6)


if __name__ == "__main__":
    unittest.main()
