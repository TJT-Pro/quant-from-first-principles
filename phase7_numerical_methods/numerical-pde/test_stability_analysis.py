import math
import unittest

import stability_analysis as sa


class TestExplicitGrowthFactor(unittest.TestCase):
    def test_worst_case_mode_at_r_half_gives_boundary_stability(self):
        xi = sa.explicit_growth_factor(0.5, math.pi)
        self.assertAlmostEqual(xi, -1.0, places=10)

    def test_stable_below_half(self):
        max_xi = sa.max_growth_factor_over_modes(sa.explicit_growth_factor, 0.4)
        self.assertLessEqual(max_xi, 1.0 + 1e-10)

    def test_unstable_above_half(self):
        max_xi = sa.max_growth_factor_over_modes(sa.explicit_growth_factor, 0.6)
        self.assertGreater(max_xi, 1.0)

    def test_growth_factor_at_zero_mode_is_one(self):
        for r in [0.1, 0.5, 1.0, 5.0]:
            self.assertAlmostEqual(sa.explicit_growth_factor(r, 0.0), 1.0)


class TestImplicitGrowthFactor(unittest.TestCase):
    def test_unconditionally_stable_across_wide_range_of_r(self):
        for r in [0.1, 0.5, 1.0, 10.0, 1000.0]:
            max_xi = sa.max_growth_factor_over_modes(sa.implicit_growth_factor, r)
            self.assertLessEqual(max_xi, 1.0 + 1e-10)

    def test_growth_factor_always_positive(self):
        for r in [0.1, 1.0, 100.0]:
            for theta in [0.1, 1.0, 3.0]:
                self.assertGreater(sa.implicit_growth_factor(r, theta), 0.0)


class TestCrankNicolsonGrowthFactor(unittest.TestCase):
    def test_unconditionally_stable_across_wide_range_of_r(self):
        for r in [0.1, 0.5, 1.0, 10.0, 1000.0]:
            max_xi = sa.max_growth_factor_over_modes(sa.crank_nicolson_growth_factor, r)
            self.assertLessEqual(max_xi, 1.0 + 1e-10)

    def test_growth_factor_can_be_negative_but_bounded(self):
        xi = sa.crank_nicolson_growth_factor(1000.0, math.pi)
        self.assertLess(xi, 0.0)
        self.assertGreaterEqual(xi, -1.0)


class TestMaxGrowthFactorOverModes(unittest.TestCase):
    def test_matches_manual_worst_case_for_explicit(self):
        r = 0.7
        manual_worst = abs(sa.explicit_growth_factor(r, math.pi))
        computed = sa.max_growth_factor_over_modes(sa.explicit_growth_factor, r)
        self.assertAlmostEqual(computed, manual_worst, places=4)


class TestEmpiricalModeGrowth(unittest.TestCase):
    def test_matches_theory_for_unstable_r(self):
        result = sa.empirically_measure_mode_growth(alpha=0.5, length=1.0, n_space=50, r_target=0.6, mode=25)
        self.assertAlmostEqual(result["empirical_ratio"], result["theoretical_growth_factor"], places=6)

    def test_matches_theory_for_stable_r(self):
        result = sa.empirically_measure_mode_growth(alpha=0.5, length=1.0, n_space=50, r_target=0.3, mode=25)
        self.assertAlmostEqual(result["empirical_ratio"], result["theoretical_growth_factor"], places=6)

    def test_growth_ratio_exceeds_one_when_unstable(self):
        # mode=49 (n_space=50) gives theta close to pi, the true
        # worst-case mode where the explicit scheme's instability is
        # most severe; a lower mode (fewer oscillations across the
        # grid) has a smaller theta and can still be damped even at
        # an r that is globally unstable for the worst-case mode.
        result = sa.empirically_measure_mode_growth(alpha=0.5, length=1.0, n_space=50, r_target=0.9, mode=49)
        self.assertGreater(result["empirical_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
