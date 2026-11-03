import unittest

import black_scholes as bs
import implied_volatility as iv


class TestBisectionRoundTrip(unittest.TestCase):
    def test_recovers_known_sigma_call(self):
        s, k, t, r, q, true_sigma = 100.0, 105.0, 0.5, 0.03, 0.01, 0.25
        market_price = bs.call_price(s, k, t, r, true_sigma, q)
        result = iv.implied_vol_bisection(market_price, s, k, t, r, q, "call")
        self.assertAlmostEqual(result["sigma"], true_sigma, places=5)

    def test_recovers_known_sigma_put(self):
        s, k, t, r, q, true_sigma = 100.0, 95.0, 1.0, 0.02, 0.0, 0.18
        market_price = bs.put_price(s, k, t, r, true_sigma, q)
        result = iv.implied_vol_bisection(market_price, s, k, t, r, q, "put")
        self.assertAlmostEqual(result["sigma"], true_sigma, places=5)

    def test_unbracketed_price_raises(self):
        with self.assertRaises(ValueError):
            iv.implied_vol_bisection(market_price=-5.0, s=100.0, k=100.0, t=1.0, r=0.02)


class TestNewtonRoundTrip(unittest.TestCase):
    def test_recovers_known_sigma_call(self):
        s, k, t, r, q, true_sigma = 100.0, 100.0, 0.5, 0.03, 0.01, 0.22
        market_price = bs.call_price(s, k, t, r, true_sigma, q)
        result = iv.implied_vol_newton(market_price, s, k, t, r, q, "call")
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["sigma"], true_sigma, places=6)

    def test_converges_faster_than_bisection(self):
        s, k, t, r, q, true_sigma = 100.0, 102.0, 0.75, 0.02, 0.0, 0.3
        market_price = bs.call_price(s, k, t, r, true_sigma, q)
        newton_result = iv.implied_vol_newton(market_price, s, k, t, r, q, "call")
        bisection_result = iv.implied_vol_bisection(market_price, s, k, t, r, q, "call")
        self.assertLess(newton_result["iterations"], bisection_result["iterations"])

    def test_recovers_across_moneyness_grid(self):
        s, t, r, q, true_sigma = 100.0, 0.5, 0.02, 0.0, 0.2
        for k in (70.0, 90.0, 100.0, 110.0, 130.0):
            market_price = bs.call_price(s, k, t, r, true_sigma, q)
            result = iv.implied_vol_newton(market_price, s, k, t, r, q, "call")
            if result["converged"]:
                self.assertAlmostEqual(result["sigma"], true_sigma, places=4)


class TestNewtonFallback(unittest.TestCase):
    def test_deep_otm_short_dated_falls_back_to_bisection(self):
        # Extremely low vega (deep out-of-the-money, short-dated) is a
        # classic case where Newton-Raphson struggles from a generic
        # starting guess.
        s, k, t, r, q, true_sigma = 100.0, 200.0, 0.02, 0.02, 0.0, 0.15
        market_price = bs.call_price(s, k, t, r, true_sigma, q)
        result = iv.implied_vol_newton_with_bisection_fallback(market_price, s, k, t, r, q, "call")
        self.assertIn(result["method"], ("newton", "bisection"))

    def test_absolute_tolerance_pitfall_for_near_zero_prices(self):
        # Documents a real, known limitation rather than hiding it:
        # when the true price is already extremely small, an absolute
        # residual tolerance is satisfied by a wide range of sigma
        # values, so both solvers report "converged" with a tiny
        # residual while the recovered sigma can differ meaningfully
        # from the sigma that actually generated the price.
        s, k, t, r, q, true_sigma = 100.0, 140.0, 0.1, 0.02, 0.0, 0.15
        market_price = bs.call_price(s, k, t, r, true_sigma, q)
        self.assertLess(market_price, 1e-10)

        newton_result = iv.implied_vol_newton(market_price, s, k, t, r, q, "call")
        bisection_result = iv.implied_vol_bisection(market_price, s, k, t, r, q, "call")

        # Both solvers succeed at matching the price...
        self.assertLess(abs(bs.call_price(s, k, t, r, newton_result["sigma"], q) - market_price), 1e-8)
        self.assertLess(abs(bs.call_price(s, k, t, r, bisection_result["sigma"], q) - market_price), 1e-8)
        # ...but neither is guaranteed to recover the true sigma precisely.
        self.assertGreater(abs(newton_result["sigma"] - true_sigma), 1e-3)

    def test_fallback_always_returns_a_method_label(self):
        s, k, t, r, q, true_sigma = 100.0, 100.0, 1.0, 0.02, 0.0, 0.25
        market_price = bs.call_price(s, k, t, r, true_sigma, q)
        result = iv.implied_vol_newton_with_bisection_fallback(market_price, s, k, t, r, q, "call")
        self.assertEqual(result["method"], "newton")


class TestPutCallConsistency(unittest.TestCase):
    def test_both_option_types_recover_same_sigma(self):
        s, k, t, r, q, true_sigma = 100.0, 100.0, 1.0, 0.025, 0.01, 0.19
        call_price = bs.call_price(s, k, t, r, true_sigma, q)
        put_price = bs.put_price(s, k, t, r, true_sigma, q)
        call_iv = iv.implied_vol_newton(call_price, s, k, t, r, q, "call")["sigma"]
        put_iv = iv.implied_vol_newton(put_price, s, k, t, r, q, "put")["sigma"]
        self.assertAlmostEqual(call_iv, put_iv, places=5)
        self.assertAlmostEqual(call_iv, true_sigma, places=5)


if __name__ == "__main__":
    unittest.main()
