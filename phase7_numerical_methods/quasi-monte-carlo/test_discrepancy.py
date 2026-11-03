import unittest

import discrepancy as disc
import low_discrepancy as ld


class TestL2StarDiscrepancySquared(unittest.TestCase):
    def test_single_point_matches_hand_derivation(self):
        # Regression test for a real bug found during development
        # (see the docstring in discrepancy.py): a single point at
        # (0.5, 0.5) in 2D has an exact, hand-derivable D^2 of
        # (1/3)^2 - (1/2)*0.5625 + 0.25 = 0.07986..., confirmed
        # independently against fine grid-integration of the defining
        # formula. An earlier, buggy version of this function returned
        # a negative value (-0.2014) for this exact case.
        result = disc.l2_star_discrepancy_squared([[0.5, 0.5]])
        self.assertAlmostEqual(result, 0.0798611111, places=8)
        self.assertGreater(result, 0.0)

    def test_never_negative(self):
        for seed in range(10):
            points = disc.random_uniform_points(30, 2, seed=seed)
            self.assertGreaterEqual(disc.l2_star_discrepancy_squared(points), -1e-12)

    def test_matches_1d_exact_case(self):
        # A single point at x=0.5 in 1D has exact D^2 = 1/12, derived
        # directly by integrating (F_N(t)-t)^2 over [0,1] by hand.
        result = disc.l2_star_discrepancy_squared([[0.5]])
        self.assertAlmostEqual(result, 1.0 / 12, places=10)

    def test_matches_independent_monte_carlo_estimate(self):
        points = disc.random_uniform_points(150, 2, seed=1)
        closed_form = disc.l2_star_discrepancy_squared(points)
        mc_estimate = disc.monte_carlo_l2_star_discrepancy_estimate(points, n_test_boxes=100000, seed=2)
        self.assertAlmostEqual(closed_form, mc_estimate, delta=mc_estimate * 0.15)

    def test_matches_monte_carlo_estimate_in_three_dimensions(self):
        points = disc.random_uniform_points(80, 3, seed=3)
        closed_form = disc.l2_star_discrepancy_squared(points)
        mc_estimate = disc.monte_carlo_l2_star_discrepancy_estimate(points, n_test_boxes=100000, seed=4)
        self.assertAlmostEqual(closed_form, mc_estimate, delta=mc_estimate * 0.2)


class TestHaltonBeatsRandomDiscrepancy(unittest.TestCase):
    def test_halton_has_lower_discrepancy_than_random_at_matched_n(self):
        for n in (50, 200, 500):
            random_pts = disc.random_uniform_points(n, 2, seed=1)
            halton_pts = ld.halton_sequence(n, 2)
            d_random = disc.l2_star_discrepancy_squared(random_pts)
            d_halton = disc.l2_star_discrepancy_squared(halton_pts)
            self.assertLess(d_halton, d_random)

    def test_relative_advantage_of_halton_grows_with_n(self):
        ratios = []
        for n in (50, 200, 500):
            random_pts = disc.random_uniform_points(n, 2, seed=1)
            halton_pts = ld.halton_sequence(n, 2)
            d_random = disc.l2_star_discrepancy_squared(random_pts) ** 0.5
            d_halton = disc.l2_star_discrepancy_squared(halton_pts) ** 0.5
            ratios.append(d_random / d_halton)
        for earlier, later in zip(ratios, ratios[1:]):
            self.assertGreater(later, earlier)


class TestRandomUniformPoints(unittest.TestCase):
    def test_shape(self):
        points = disc.random_uniform_points(20, 3, seed=1)
        self.assertEqual(len(points), 20)
        for p in points:
            self.assertEqual(len(p), 3)

    def test_all_in_unit_interval(self):
        points = disc.random_uniform_points(100, 2, seed=1)
        for p in points:
            for v in p:
                self.assertGreaterEqual(v, 0.0)
                self.assertLess(v, 1.0)

    def test_deterministic_given_seed(self):
        p1 = disc.random_uniform_points(10, 2, seed=42)
        p2 = disc.random_uniform_points(10, 2, seed=42)
        self.assertEqual(p1, p2)


if __name__ == "__main__":
    unittest.main()
