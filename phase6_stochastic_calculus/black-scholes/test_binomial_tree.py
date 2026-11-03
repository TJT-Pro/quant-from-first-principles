import math
import unittest

import black_scholes as bs
import binomial_tree as bt


class TestCRRParameters(unittest.TestCase):
    def test_u_and_d_are_reciprocal(self):
        u, d, p = bt.crr_parameters(sigma=0.2, r=0.03, q=0.0, dt=0.01)
        self.assertAlmostEqual(u * d, 1.0, places=10)

    def test_probability_reproduces_risk_neutral_growth(self):
        sigma, r, q, dt = 0.25, 0.04, 0.01, 0.02
        u, d, p = bt.crr_parameters(sigma, r, q, dt)
        self.assertAlmostEqual(p * u + (1 - p) * d, math.exp((r - q) * dt), places=10)

    def test_probability_in_valid_range_for_reasonable_dt(self):
        u, d, p = bt.crr_parameters(sigma=0.3, r=0.05, q=0.0, dt=0.001)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_excessive_dt_raises(self):
        with self.assertRaises(ValueError):
            bt.crr_parameters(sigma=0.05, r=0.5, q=0.0, dt=10.0)


class TestBinomialConvergesToBlackScholes(unittest.TestCase):
    def test_european_call_converges(self):
        s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.2
        bs_price = bs.call_price(s, k, t, r, sigma)
        tree_price = bt.binomial_price(s, k, t, r, sigma, 2000, "call", "european")
        self.assertAlmostEqual(tree_price, bs_price, delta=0.05)

    def test_european_put_converges(self):
        s, k, t, r, sigma, q = 100.0, 105.0, 0.75, 0.02, 0.25, 0.01
        bs_price = bs.put_price(s, k, t, r, sigma, q)
        tree_price = bt.binomial_price(s, k, t, r, sigma, 2000, "put", "european", q)
        self.assertAlmostEqual(tree_price, bs_price, delta=0.05)

    def test_convergence_error_shrinks_with_more_steps(self):
        s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.2
        bs_price = bs.call_price(s, k, t, r, sigma)
        errors = [abs(bt.binomial_price(s, k, t, r, sigma, n, "call", "european") - bs_price)
                  for n in (10, 100, 1000)]
        self.assertGreater(errors[0], errors[2])


class TestAmericanOptions(unittest.TestCase):
    def test_american_call_no_dividend_equals_european(self):
        # With no dividends, early exercise of an American call is
        # never optimal (you give up the remaining time value for no
        # offsetting benefit), so American and European calls must
        # have the same price.
        s, k, t, r, sigma = 100.0, 95.0, 1.0, 0.03, 0.2
        european = bt.binomial_price(s, k, t, r, sigma, 500, "call", "european")
        american = bt.binomial_price(s, k, t, r, sigma, 500, "call", "american")
        self.assertAlmostEqual(european, american, places=6)

    def test_american_put_worth_at_least_european(self):
        # Early exercise can be optimal for a put (locking in interest
        # on the strike proceeds sooner), so the American put must be
        # worth at least as much as the European put.
        s, k, t, r, sigma = 100.0, 110.0, 1.0, 0.05, 0.2
        european = bt.binomial_price(s, k, t, r, sigma, 500, "put", "european")
        american = bt.binomial_price(s, k, t, r, sigma, 500, "put", "american")
        self.assertGreaterEqual(american, european - 1e-9)

    def test_american_call_with_dividend_can_exceed_european(self):
        # With a large enough dividend/foreign yield, early exercise of
        # an American call can become optimal (capturing the dividend
        # rather than holding an option on an ex-dividend underlying).
        s, k, t, r, sigma, q = 100.0, 90.0, 1.0, 0.02, 0.15, 0.10
        european = bt.binomial_price(s, k, t, r, sigma, 500, "call", "european", q)
        american = bt.binomial_price(s, k, t, r, sigma, 500, "call", "american", q)
        self.assertGreaterEqual(american, european - 1e-9)

    def test_american_option_at_least_intrinsic_value(self):
        s, k, t, r, sigma = 80.0, 100.0, 0.5, 0.02, 0.2
        price = bt.binomial_price(s, k, t, r, sigma, 300, "put", "american")
        self.assertGreaterEqual(price, max(k - s, 0.0) - 1e-9)


class TestInputValidation(unittest.TestCase):
    def test_invalid_option_type_raises(self):
        with self.assertRaises(ValueError):
            bt.binomial_price(100.0, 100.0, 1.0, 0.02, 0.2, 100, option_type="straddle")

    def test_invalid_exercise_raises(self):
        with self.assertRaises(ValueError):
            bt.binomial_price(100.0, 100.0, 1.0, 0.02, 0.2, 100, exercise="bermudan")

    def test_zero_steps_raises(self):
        with self.assertRaises(ValueError):
            bt.binomial_price(100.0, 100.0, 1.0, 0.02, 0.2, 0)


class TestConvergenceSeries(unittest.TestCase):
    def test_series_length_matches_input(self):
        steps = [10, 50, 100, 500]
        series = bt.binomial_convergence_series(100.0, 100.0, 1.0, 0.02, 0.2, "call", steps)
        self.assertEqual(len(series), len(steps))

    def test_series_approaches_bs_price(self):
        s, k, t, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.2
        bs_price = bs.call_price(s, k, t, r, sigma)
        steps = [10, 100, 1000]
        series = bt.binomial_convergence_series(s, k, t, r, sigma, "call", steps)
        self.assertLess(abs(series[-1] - bs_price), abs(series[0] - bs_price))


if __name__ == "__main__":
    unittest.main()
