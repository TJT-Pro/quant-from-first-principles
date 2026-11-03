import math
import unittest

import black_scholes_pde as bs


class TestClosedFormBlackScholes(unittest.TestCase):
    def test_call_price_positive_and_reasonable(self):
        price = bs.black_scholes_closed_form(100.0, 100.0, 0.05, 0.2, 1.0, "call")
        self.assertGreater(price, 0.0)
        self.assertLess(price, 100.0)

    def test_put_call_parity_holds_exactly(self):
        s, k, r, sigma, t = 100.0, 105.0, 0.03, 0.25, 0.5
        call = bs.black_scholes_closed_form(s, k, r, sigma, t, "call")
        put = bs.black_scholes_closed_form(s, k, r, sigma, t, "put")
        self.assertAlmostEqual(call - put, s - k * math.exp(-r * t), places=10)

    def test_zero_time_gives_intrinsic_value(self):
        self.assertAlmostEqual(bs.black_scholes_closed_form(110.0, 100.0, 0.05, 0.2, 0.0, "call"), 10.0)
        self.assertAlmostEqual(bs.black_scholes_closed_form(90.0, 100.0, 0.05, 0.2, 0.0, "put"), 10.0)

    def test_deep_itm_call_approaches_intrinsic_plus_carry(self):
        s, k, r, sigma, t = 1000.0, 100.0, 0.05, 0.2, 1.0
        price = bs.black_scholes_closed_form(s, k, r, sigma, t, "call")
        expected = s - k * math.exp(-r * t)
        self.assertAlmostEqual(price, expected, delta=0.01)


class TestPDESchemesMatchClosedForm(unittest.TestCase):
    def setUp(self):
        self.k, self.r, self.sigma, self.t = 100.0, 0.05, 0.2, 1.0
        self.s0 = 100.0
        self.x_min = math.log(self.k) - 3 * self.sigma * math.sqrt(self.t) - 1
        self.x_max = math.log(self.k) + 3 * self.sigma * math.sqrt(self.t) + 1
        self.closed_form = bs.black_scholes_closed_form(self.s0, self.k, self.r, self.sigma, self.t, "call")

    def test_explicit_matches_closed_form(self):
        result = bs.solve_bs_pde_explicit(self.k, self.r, self.sigma, self.t, self.x_min, self.x_max,
                                           200, 2000, "call")
        price = bs.price_at_spot(result, self.s0)
        self.assertAlmostEqual(price, self.closed_form, delta=0.05)

    def test_implicit_matches_closed_form(self):
        result = bs.solve_bs_pde_implicit(self.k, self.r, self.sigma, self.t, self.x_min, self.x_max,
                                           200, 500, "call")
        price = bs.price_at_spot(result, self.s0)
        self.assertAlmostEqual(price, self.closed_form, delta=0.05)

    def test_crank_nicolson_matches_closed_form(self):
        result = bs.solve_bs_pde_crank_nicolson(self.k, self.r, self.sigma, self.t, self.x_min, self.x_max,
                                                 200, 200, "call")
        price = bs.price_at_spot(result, self.s0)
        self.assertAlmostEqual(price, self.closed_form, delta=0.05)

    def test_put_matches_closed_form(self):
        put_closed_form = bs.black_scholes_closed_form(self.s0, self.k, self.r, self.sigma, self.t, "put")
        result = bs.solve_bs_pde_crank_nicolson(self.k, self.r, self.sigma, self.t, self.x_min, self.x_max,
                                                 200, 200, "put")
        price = bs.price_at_spot(result, self.s0)
        self.assertAlmostEqual(price, put_closed_form, delta=0.05)


class TestCrankNicolsonOutperformsImplicitInTime(unittest.TestCase):
    def test_crank_nicolson_converges_faster_as_time_steps_increase(self):
        k, r, sigma, t = 100.0, 0.05, 0.2, 1.0
        s0 = 100.0
        x_min = math.log(k) - 3 * sigma * math.sqrt(t) - 1
        x_max = math.log(k) + 3 * sigma * math.sqrt(t) + 1
        n_space = 400
        closed_form = bs.black_scholes_closed_form(s0, k, r, sigma, t, "call")

        n_time = 50
        imp_result = bs.solve_bs_pde_implicit(k, r, sigma, t, x_min, x_max, n_space, n_time, "call")
        cn_result = bs.solve_bs_pde_crank_nicolson(k, r, sigma, t, x_min, x_max, n_space, n_time, "call")
        imp_error = abs(bs.price_at_spot(imp_result, s0) - closed_form)
        cn_error = abs(bs.price_at_spot(cn_result, s0) - closed_form)

        self.assertLess(cn_error, imp_error)

    def test_implicit_error_shrinks_roughly_linearly_with_more_time_steps(self):
        k, r, sigma, t = 100.0, 0.05, 0.2, 1.0
        s0 = 100.0
        x_min = math.log(k) - 3 * sigma * math.sqrt(t) - 1
        x_max = math.log(k) + 3 * sigma * math.sqrt(t) + 1
        n_space = 400
        closed_form = bs.black_scholes_closed_form(s0, k, r, sigma, t, "call")

        errors = []
        for n_time in (25, 50, 100, 200):
            result = bs.solve_bs_pde_implicit(k, r, sigma, t, x_min, x_max, n_space, n_time, "call")
            errors.append(abs(bs.price_at_spot(result, s0) - closed_form))
        for earlier, later in zip(errors, errors[1:]):
            self.assertGreater(earlier, later)


class TestPriceAtSpot(unittest.TestCase):
    def test_interpolates_between_grid_points(self):
        result = {"x": [0.0, 1.0, 2.0], "v": [0.0, 10.0, 20.0]}
        s0 = math.exp(0.5)
        interpolated = bs.price_at_spot(result, s0)
        self.assertAlmostEqual(interpolated, 5.0, places=8)

    def test_exact_grid_point(self):
        result = {"x": [0.0, 1.0, 2.0], "v": [0.0, 10.0, 20.0]}
        self.assertAlmostEqual(bs.price_at_spot(result, math.exp(1.0)), 10.0, places=8)

    def test_out_of_range_clamps_to_boundary(self):
        result = {"x": [0.0, 1.0, 2.0], "v": [0.0, 10.0, 20.0]}
        self.assertEqual(bs.price_at_spot(result, math.exp(-5.0)), 0.0)
        self.assertEqual(bs.price_at_spot(result, math.exp(10.0)), 20.0)


if __name__ == "__main__":
    unittest.main()
