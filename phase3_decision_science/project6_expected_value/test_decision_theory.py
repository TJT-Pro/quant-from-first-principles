"""Tests for the decision theory library.

Expected values and decision rule scores are checked against hand
computed numbers.  Each utility is checked for exact inversion, correct
risk aversion measures and domain behavior.  Risk premia are checked for
monotonicity in risk aversion, prospect theory for the loss aversion
effects, and the St. Petersburg game for its unbounded expectation and
finite certainty equivalent.  A real data test runs the invest decision
on the EUR/USD returns.
"""

import math
import random
import unittest

import decision_theory as dt


class TestExpectedValue(unittest.TestCase):
    def test_known_value(self):
        self.assertAlmostEqual(dt.expected_value([1, 2, 5], [0.2, 0.3, 0.5]),
                               0.2 + 0.6 + 2.5)

    def test_expected_function(self):
        self.assertAlmostEqual(dt.expected_function([1, 2, 3], [0.5, 0.3, 0.2],
                                                    lambda x: x * x),
                               0.5 + 1.2 + 1.8)

    def test_variance_stddev(self):
        outcomes = [1.0, 2.0, 5.0]
        probs = [0.2, 0.3, 0.5]
        mu = dt.expected_value(outcomes, probs)
        var = sum(p * (x - mu) ** 2 for p, x in zip(probs, outcomes))
        self.assertAlmostEqual(dt.variance(outcomes, probs), var)
        self.assertAlmostEqual(dt.stddev(outcomes, probs), math.sqrt(var))

    def test_constant_lottery(self):
        self.assertEqual(dt.expected_value([7, 7, 7], [0.2, 0.3, 0.5]), 7.0)
        self.assertAlmostEqual(dt.variance([7, 7, 7], [0.2, 0.3, 0.5]), 0.0)

    def test_bad_probabilities(self):
        with self.assertRaises(ValueError):
            dt.expected_value([1, 2], [0.5, 0.4])
        with self.assertRaises(ValueError):
            dt.expected_value([1, 2], [0.5, 0.5, 0.1])
        with self.assertRaises(ValueError):
            dt.expected_value([1, 2], [-0.1, 1.1])
        with self.assertRaises(ValueError):
            dt.expected_value([], [])


class TestDecisionRules(unittest.TestCase):
    PAYOFFS = [[2.0, -1.0, 4.0], [3.0, 2.0, 1.0], [1.0, 0.0, 5.0]]
    PROBS = [0.3, 0.4, 0.3]

    def test_expected_value_rule(self):
        scores = dt.decision_scores(self.PAYOFFS, "expected-value", self.PROBS)
        manual = [0.3 * r[0] + 0.4 * r[1] + 0.3 * r[2] for r in self.PAYOFFS]
        self.assertEqual(scores, manual)
        self.assertEqual(dt.best_action(self.PAYOFFS, "expected-value", self.PROBS),
                         (1, manual[1]))

    def test_maximax(self):
        self.assertEqual(dt.decision_scores(self.PAYOFFS, "maximax"), [4.0, 3.0, 5.0])
        self.assertEqual(dt.best_action(self.PAYOFFS, "maximax")[0], 2)

    def test_maximin(self):
        self.assertEqual(dt.decision_scores(self.PAYOFFS, "maximin"), [-1.0, 1.0, 0.0])
        self.assertEqual(dt.best_action(self.PAYOFFS, "maximin")[0], 1)

    def test_laplace(self):
        scores = dt.decision_scores(self.PAYOFFS, "laplace")
        manual = [sum(row) / 3.0 for row in self.PAYOFFS]
        self.assertEqual(scores, manual)
        self.assertEqual(dt.best_action(self.PAYOFFS, "laplace")[0], 1)

    def test_hurwicz(self):
        scores = dt.decision_scores(self.PAYOFFS, "hurwicz", alpha=0.5)
        self.assertEqual(scores, [1.5, 2.0, 2.5])
        self.assertEqual(dt.best_action(self.PAYOFFS, "hurwicz", alpha=0.5)[0], 2)

    def test_minimax_regret(self):
        regrets = dt.regret_matrix(self.PAYOFFS)
        best_per_state = [max(col) for col in zip(*self.PAYOFFS)]
        manual = [[best_per_state[j] - row[j] for j in range(3)] for row in self.PAYOFFS]
        self.assertEqual(regrets, manual)
        scores = dt.decision_scores(self.PAYOFFS, "minimax-regret")
        self.assertEqual(scores, [max(r) for r in manual])
        best, score = dt.best_action(self.PAYOFFS, "minimax-regret")
        self.assertEqual(best, min(range(3), key=lambda i: scores[i]))

    def test_expected_value_rule_needs_probs(self):
        with self.assertRaises(ValueError):
            dt.decision_scores(self.PAYOFFS, "expected-value")

    def test_hurwicz_alpha_range(self):
        with self.assertRaises(ValueError):
            dt.decision_scores(self.PAYOFFS, "hurwicz", alpha=1.5)

    def test_bad_rule(self):
        with self.assertRaises(ValueError):
            dt.decision_scores(self.PAYOFFS, "not-a-rule")

    def test_ragged_matrix(self):
        with self.assertRaises(ValueError):
            dt.decision_scores([[1.0, 2.0], [1.0]])


class TestDominance(unittest.TestCase):
    def test_strict_dominance(self):
        self.assertTrue(dt.dominates(0, 1, [[1.0, 2.0, 3.0], [1.0, 2.0, 2.0]]))

    def test_identical_not_dominating(self):
        self.assertFalse(dt.dominates(0, 1, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))

    def test_worse_in_one_state(self):
        self.assertFalse(dt.dominates(0, 1, [[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]]))

    def test_bad_index(self):
        with self.assertRaises(ValueError):
            dt.dominates(0, 5, [[1.0, 2.0]])


class TestUtilities(unittest.TestCase):
    def test_linear(self):
        u = dt.make_utility("linear")
        self.assertEqual(u.u(5.0), 5.0)
        self.assertEqual(u.inverse(5.0), 5.0)
        self.assertEqual(u.absolute_risk(10.0), 0.0)
        self.assertEqual(u.relative_risk(10.0), 0.0)

    def test_log(self):
        u = dt.make_utility("log")
        self.assertAlmostEqual(u.u(10.0), math.log(10.0))
        self.assertAlmostEqual(u.inverse(math.log(10.0)), 10.0)
        self.assertAlmostEqual(u.relative_risk(100.0), 1.0)
        self.assertAlmostEqual(u.absolute_risk(100.0), 0.01)
        with self.assertRaises(ValueError):
            u.u(0.0)

    def test_power(self):
        u = dt.make_utility("power", 0.5)
        self.assertAlmostEqual(u.u(9.0), 3.0)
        self.assertAlmostEqual(u.inverse(3.0), 9.0)
        self.assertAlmostEqual(u.relative_risk(20.0), 0.5)
        with self.assertRaises(ValueError):
            dt.make_utility("power", 0.0)
        with self.assertRaises(ValueError):
            dt.make_utility("power", 1.5)

    def test_exponential(self):
        u = dt.make_utility("exponential", 3.0)
        self.assertAlmostEqual(u.u(0.5), 1.0 - math.exp(-1.5))
        self.assertAlmostEqual(u.inverse(1.0 - math.exp(-1.5)), 0.5)
        self.assertAlmostEqual(u.absolute_risk(100.0), 3.0)
        with self.assertRaises(ValueError):
            u.inverse(1.5)
        with self.assertRaises(ValueError):
            dt.make_utility("exponential", -1.0)

    def test_quadratic(self):
        u = dt.make_utility("quadratic", 0.01)
        self.assertAlmostEqual(u.u(10.0), 10.0 - 1.0)
        self.assertAlmostEqual(u.inverse(10.0 - 1.0), 10.0)
        self.assertGreater(u.absolute_risk(10.0), 0.0)
        with self.assertRaises(ValueError):
            u.u(60.0)

    def test_unknown_utility(self):
        with self.assertRaises(ValueError):
            dt.make_utility("sigmoid")


class TestExpectedUtility(unittest.TestCase):
    def test_hand_computed(self):
        outcomes = [10.0, 30.0]
        probs = [0.5, 0.5]
        u = dt.make_utility("log")
        eu = dt.expected_utility(outcomes, probs, u)
        self.assertAlmostEqual(eu, 0.5 * math.log(10.0) + 0.5 * math.log(30.0))

    def test_ce_inverts_utility(self):
        outcomes = [10.0, 30.0, 50.0]
        probs = [0.2, 0.3, 0.5]
        for kind, param in (("log", None), ("power", 0.5),
                            ("exponential", 2.0), ("quadratic", 0.005)):
            u = dt.make_utility(kind, param)
            eu = dt.expected_utility(outcomes, probs, u)
            ce = dt.certainty_equivalent(outcomes, probs, u)
            self.assertAlmostEqual(u.u(ce), eu, places=9)

    def test_ce_below_ev_for_concave(self):
        outcomes = [10.0, 30.0]
        probs = [0.5, 0.5]
        ev = dt.expected_value(outcomes, probs)
        for u in (dt.make_utility("log"), dt.make_utility("power", 0.5),
                  dt.make_utility("exponential", 0.5)):
            ce = dt.certainty_equivalent(outcomes, probs, u)
            self.assertLess(ce, ev)
            self.assertGreater(dt.risk_premium(outcomes, probs, u), 0.0)

    def test_linear_ce_equals_ev(self):
        outcomes = [10.0, 30.0]
        probs = [0.5, 0.5]
        u = dt.make_utility("linear")
        self.assertAlmostEqual(dt.certainty_equivalent(outcomes, probs, u), 20.0)
        self.assertAlmostEqual(dt.risk_premium(outcomes, probs, u), 0.0)

    def test_premium_grows_with_risk_aversion(self):
        outcomes = [5.0, 50.0]
        probs = [0.5, 0.5]
        mild = dt.risk_premium(outcomes, probs, dt.make_utility("power", 0.8))
        strong = dt.risk_premium(outcomes, probs, dt.make_utility("power", 0.3))
        self.assertGreater(strong, mild)


class TestProspectTheory(unittest.TestCase):
    def test_probability_weight_edges(self):
        self.assertEqual(dt.probability_weight(0.0), 0.0)
        self.assertEqual(dt.probability_weight(1.0), 1.0)

    def test_weight_distortion(self):
        self.assertGreater(dt.probability_weight(0.1), 0.1)
        self.assertLess(dt.probability_weight(0.9), 0.9)

    def test_loss_aversion_makes_symmetric_negative(self):
        value = dt.prospect_value([100.0, -100.0], [0.5, 0.5])
        self.assertLess(value, 0.0)

    def test_gain_only_positive(self):
        value = dt.prospect_value([100.0, 200.0], [0.5, 0.5])
        self.assertGreater(value, 0.0)

    def test_more_loss_aversion_lowers_value(self):
        mild = dt.prospect_value([100.0, -100.0], [0.5, 0.5], loss_aversion=1.0)
        strong = dt.prospect_value([100.0, -100.0], [0.5, 0.5], loss_aversion=3.0)
        self.assertLess(strong, mild)

    def test_reference_point_shifts_value(self):
        outcomes = [0.0, 100.0]
        probs = [0.5, 0.5]
        at_zero = dt.prospect_value(outcomes, probs, reference=0.0)
        at_fifty = dt.prospect_value(outcomes, probs, reference=50.0)
        self.assertNotEqual(at_zero, at_fifty)


class TestStPetersburg(unittest.TestCase):
    def test_expected_value_formula(self):
        for n in (1, 2, 5, 20):
            outcomes, probs = dt.st_petersburg(n)
            self.assertAlmostEqual(dt.expected_value(outcomes, probs),
                                   n / (1.0 - 2.0 ** (-n)), places=10)

    def test_log_certainty_equivalent_bounded(self):
        outcomes, probs = dt.st_petersburg(40)
        ce = dt.certainty_equivalent(outcomes, probs, dt.make_utility("log"))
        self.assertGreater(ce, 3.5)
        self.assertLess(ce, 4.5)

    def test_bad_n(self):
        with self.assertRaises(ValueError):
            dt.st_petersburg(0)


class TestKelly(unittest.TestCase):
    def test_binary_hand_computed(self):
        self.assertAlmostEqual(dt.kelly_fraction_binary(0.6, 1.0), 0.2)
        self.assertAlmostEqual(dt.kelly_fraction_binary(0.4, 2.0), 0.1)
        self.assertAlmostEqual(dt.kelly_fraction_binary(0.55, 1.0), 0.1)

    def test_binary_unfavorable_no_bet(self):
        self.assertEqual(dt.kelly_fraction_binary(0.4, 1.0), 0.0)

    def test_binary_bad_inputs(self):
        with self.assertRaises(ValueError):
            dt.kelly_fraction_binary(0.0, 1.0)
        with self.assertRaises(ValueError):
            dt.kelly_fraction_binary(0.5, -1.0)

    def test_log_growth_rate_hand_computed(self):
        returns = [-0.5, 1.0]
        rate = dt.log_growth_rate(returns, 0.5)
        expected = 0.5 * math.log(1.0 + 0.5 * -0.5) + 0.5 * math.log(1.0 + 0.5 * 1.0)
        self.assertAlmostEqual(rate, expected)

    def test_bankruptcy_gives_negative_infinity(self):
        returns = [-1.0, 1.0]
        self.assertEqual(dt.log_growth_rate(returns, 1.0), float("-inf"))

    def test_optimal_kelly_exact_two_outcome(self):
        returns = [-1.0, 2.0]
        f = dt.optimal_kelly_fraction(returns)
        self.assertAlmostEqual(f, 0.25, places=6)
        self.assertGreaterEqual(dt.log_growth_rate(returns, f),
                                dt.log_growth_rate(returns, 0.1))
        self.assertGreaterEqual(dt.log_growth_rate(returns, f),
                                dt.log_growth_rate(returns, 0.4))

    def test_optimal_kelly_negative_mean_is_zero(self):
        returns = [-0.2, -0.1]
        self.assertEqual(dt.optimal_kelly_fraction(returns), 0.0)

    def test_expected_log_wealth_scales(self):
        returns = [-0.5, 1.0]
        g = dt.log_growth_rate(returns, 0.3)
        self.assertAlmostEqual(dt.expected_log_wealth(returns, 0.3, 100), 100 * g)

    def test_wealth_trajectory(self):
        returns = [-0.5, 1.0, 0.0]
        f = 0.3
        traj = dt.wealth_trajectory(returns, f, 1.0)
        manual = [1.0]
        for r in returns:
            manual.append(manual[-1] * (1.0 + f * r))
        self.assertEqual(len(traj), 4)
        for a, b in zip(traj, manual):
            self.assertAlmostEqual(a, b)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            dt.optimal_kelly_fraction([])
        with self.assertRaises(ValueError):
            dt.log_growth_rate([1.0], -0.1)
        with self.assertRaises(ValueError):
            dt.expected_log_wealth([1.0], 0.5, 0)
        with self.assertRaises(ValueError):
            dt.wealth_trajectory([1.0], -0.1)


class TestRealDataIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        import csv
        base = os.path.dirname(os.path.abspath(__file__))
        cls.returns = []
        with open(os.path.join(base, "data/DEXUSEU_returns.csv"), newline="") as handle:
            for row in csv.DictReader(handle):
                if row["return"] != "":
                    cls.returns.append(float(row["return"]))
        assert len(cls.returns) == 6930

    def test_invest_decision_on_market_days(self):
        probs = [1.0 / len(self.returns)] * len(self.returns)
        u = dt.make_utility("exponential", 10.0)
        ev = dt.expected_value(self.returns, probs)
        ce = dt.certainty_equivalent(self.returns, probs, u)
        premium = dt.risk_premium(self.returns, probs, u)
        self.assertAlmostEqual(ev, sum(self.returns) / len(self.returns))
        self.assertLess(ce, ev)
        self.assertGreater(premium, 0.0)

    def test_log_utility_needs_positive_domain(self):
        probs = [1.0 / len(self.returns)] * len(self.returns)
        with self.assertRaises(ValueError):
            dt.expected_utility(self.returns, probs, dt.make_utility("log"))

    def test_prospect_value_sign_of_a_market_day(self):
        random.seed(71)
        sample = random.sample(self.returns, 200)
        probs = [1.0 / len(sample)] * len(sample)
        value = dt.prospect_value(sample, probs)
        self.assertTrue(value < 0.0 or value > 0.0)


if __name__ == "__main__":
    unittest.main()
