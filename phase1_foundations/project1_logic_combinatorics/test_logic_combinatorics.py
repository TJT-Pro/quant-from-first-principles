"""Tests for Project 1 - Logic & Combinatorics Toolkit.

Run with:  python3 -m unittest -v
"""

import os
import unittest

import logic_combinatorics as lc


# propositional logic
# -------------------

class TestLogicParser(unittest.TestCase):

    def test_parse_variable(self):
        self.assertEqual(lc.parse("P"), ("var", "P"))

    def test_parse_not(self):
        self.assertEqual(lc.parse("NOT P"), ("not", ("var", "P")))

    def test_parse_and_or(self):
        self.assertEqual(
            lc.parse("P AND Q OR R"),
            ("or", ("and", ("var", "P"), ("var", "Q")), ("var", "R")))

    def test_precedence_not_binds_tightest(self):
        self.assertEqual(
            lc.parse("NOT P AND Q"),
            ("and", ("not", ("var", "P")), ("var", "Q")))

    def test_implication_looser_than_or(self):
        self.assertEqual(
            lc.parse("P OR Q -> R"),
            ("implies", ("or", ("var", "P"), ("var", "Q")), ("var", "R")))

    def test_iff_loosest(self):
        self.assertEqual(
            lc.parse("P -> Q <-> R"),
            ("iff", ("implies", ("var", "P"), ("var", "Q")), ("var", "R")))

    def test_parentheses_override(self):
        self.assertEqual(
            lc.parse("(P OR Q) -> R"),
            ("implies", ("or", ("var", "P"), ("var", "Q")), ("var", "R")))

    def test_symbolic_operators(self):
        self.assertEqual(lc.parse("~P & Q | P ^ Q"), lc.parse("(NOT P AND Q) OR (P XOR Q)"))
        self.assertEqual(lc.parse("P -> Q"), lc.parse("P => Q"))
        self.assertEqual(lc.parse("P <-> Q"), lc.parse("P <=> Q"))

    def test_multichar_variable(self):
        self.assertEqual(lc.parse("rain AND hail"), ("and", ("var", "rain"), ("var", "hail")))

    def test_empty_formula_raises(self):
        with self.assertRaises(lc.LogicParseError):
            lc.parse("")

    def test_unbalanced_paren_raises(self):
        with self.assertRaises(lc.LogicParseError):
            lc.parse("(P AND Q")

    def test_bad_character_raises(self):
        with self.assertRaises(lc.LogicParseError):
            lc.parse("P % Q")


class TestLogicEvaluation(unittest.TestCase):

    def test_and(self):
        expr = lc.parse("P AND Q")
        self.assertTrue(lc.evaluate(expr, {"P": True, "Q": True}))
        self.assertFalse(lc.evaluate(expr, {"P": True, "Q": False}))

    def test_implies(self):
        expr = lc.parse("P -> Q")
        self.assertFalse(lc.evaluate(expr, {"P": True, "Q": False}))
        self.assertTrue(lc.evaluate(expr, {"P": True, "Q": True}))
        self.assertTrue(lc.evaluate(expr, {"P": False, "Q": False}))

    def test_iff(self):
        expr = lc.parse("P <-> Q")
        self.assertTrue(lc.evaluate(expr, {"P": True, "Q": True}))
        self.assertTrue(lc.evaluate(expr, {"P": False, "Q": False}))
        self.assertFalse(lc.evaluate(expr, {"P": True, "Q": False}))

    def test_missing_variable_raises(self):
        with self.assertRaises(ValueError):
            lc.evaluate(lc.parse("P"), {})


class TestTruthTable(unittest.TestCase):

    def test_single_variable(self):
        names, rows = lc.truth_table("P")
        self.assertEqual(names, ["P"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ([False], False))
        self.assertEqual(rows[1], ([True], True))

    def test_and_table(self):
        names, rows = lc.truth_table("P AND Q")
        self.assertEqual(names, ["P", "Q"])
        self.assertEqual(rows, [
            ([False, False], False),
            ([False, True], False),
            ([True, False], False),
            ([True, True], True),
        ])

    def test_textbook_example(self):
        names, rows = lc.truth_table("(P AND Q) OR NOT R")
        self.assertEqual(names, ["P", "Q", "R"])
        expected = {
            (False, False, False): True,
            (False, False, True): False,
            (False, True, False): True,
            (False, True, True): False,
            (True, False, False): True,
            (True, False, True): False,
            (True, True, False): True,
            (True, True, True): True,
        }
        for vals, result in rows:
            self.assertEqual(result, expected[tuple(vals)])

    def test_variables_ordered_by_first_use(self):
        names, _ = lc.truth_table("Q AND P")
        self.assertEqual(names, ["Q", "P"])

    def test_format_truth_table_is_text(self):
        out = lc.format_truth_table("P OR NOT Q")
        self.assertIn("P", out)
        self.assertIn("Q", out)
        self.assertIn("True", out)
        self.assertIn("False", out)


# set operations engine (no set() built-ins)
# ------------------------------------------

class TestSetEngine(unittest.TestCase):

    def test_make_set_deduplicates_and_preserves_order(self):
        self.assertEqual(lc.make_set([3, 1, 2, 1, 3, 2]), [3, 1, 2])
        self.assertEqual(lc.make_set([]), [])

    def test_no_set_type_used_in_engine(self):
        import ast
        import inspect
        for name in ("make_set", "union", "intersection",
                     "difference", "symmetric_difference",
                     "power_set", "cartesian_product"):
            fn = getattr(lc, name)
            tree = ast.parse(inspect.getsource(fn))
            calls = [n for n in ast.walk(tree)
                     if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name)
                     and n.func.id == "set"]
            self.assertEqual(calls, [],
                             "{} calls the set() constructor".format(name))

    def test_union(self):
        self.assertEqual(lc.union([1, 2], [2, 3]), [1, 2, 3])

    def test_union_deduplicates_inputs(self):
        self.assertEqual(lc.union([1, 2, 3, 2], [3, 4]), [1, 2, 3, 4])

    def test_intersection(self):
        self.assertEqual(lc.intersection([1, 2, 3], [2, 3, 4]), [2, 3])
        self.assertEqual(lc.intersection([1], [2]), [])
        self.assertEqual(lc.intersection([1, 2, 1], [1, 3]), [1])

    def test_difference(self):
        self.assertEqual(lc.difference([1, 2, 3], [2]), [1, 3])

    def test_symmetric_difference(self):
        self.assertEqual(lc.symmetric_difference([1, 2], [2, 3]), [1, 3])

    def test_power_set(self):
        ps = lc.power_set([1, 2, 3])
        self.assertEqual(len(ps), 8)
        self.assertIn([], ps)
        self.assertIn([1, 2, 3], ps)
        self.assertIn([1, 3], ps)
        self.assertNotIn([1, 2, 3, 4], ps)

    def test_power_set_deduplicates_input(self):
        ps = lc.power_set([1, 1, 2])
        self.assertEqual(len(ps), 4)

    def test_cartesian_product(self):
        cp = lc.cartesian_product([1, 2], ["a", "b"])
        self.assertEqual(len(cp), 4)
        self.assertIn((1, "a"), cp)
        self.assertIn((2, "b"), cp)


# combinatorics
# -------------

class TestCombinatorics(unittest.TestCase):

    def test_factorial_known_values(self):
        self.assertEqual(lc.factorial(0), 1)
        self.assertEqual(lc.factorial(1), 1)
        self.assertEqual(lc.factorial(5), 120)
        self.assertEqual(lc.factorial(10), 3628800)

    def test_factorial_negative_raises(self):
        with self.assertRaises(ValueError):
            lc.factorial(-1)

    def test_factorial_float_raises(self):
        with self.assertRaises(TypeError):
            lc.factorial(3.0)

    def test_nPr_known_values(self):
        self.assertEqual(lc.nPr(5, 0), 1)
        self.assertEqual(lc.nPr(5, 1), 5)
        self.assertEqual(lc.nPr(5, 2), 20)
        self.assertEqual(lc.nPr(5, 5), 120)
        self.assertEqual(lc.nPr(10, 3), 720)

    def test_nCr_known_values(self):
        self.assertEqual(lc.nCr(5, 0), 1)
        self.assertEqual(lc.nCr(5, 1), 5)
        self.assertEqual(lc.nCr(5, 2), 10)
        self.assertEqual(lc.nCr(5, 5), 1)
        self.assertEqual(lc.nCr(10, 3), 120)
        self.assertEqual(lc.nCr(49, 6), 13983816)

    def test_nCr_symmetry(self):
        for n in range(0, 15):
            for r in range(0, n + 1):
                self.assertEqual(lc.nCr(n, r), lc.nCr(n, n - r))

    def test_nPr_nCr_relationship(self):
        for n in range(1, 10):
            for r in range(0, n + 1):
                self.assertEqual(lc.nPr(n, r), lc.nCr(n, r) * lc.factorial(r))

    def test_r_greater_than_n_raises(self):
        with self.assertRaises(ValueError):
            lc.nPr(3, 4)
        with self.assertRaises(ValueError):
            lc.nCr(3, 4)

    def test_cross_check_scipy(self):
        try:
            from scipy.special import comb, perm
        except ImportError:
            self.skipTest("scipy not installed")
        for n in range(0, 15):
            for r in range(0, n + 1):
                self.assertEqual(lc.nPr(n, r), int(perm(n, r)))
                self.assertEqual(lc.nCr(n, r), int(comb(n, r, exact=True)))


# graph traversals
# ----------------

class TestGraphTraversal(unittest.TestCase):

    def test_graph_from_edges(self):
        adj = lc.graph_from_edges([("A", "B"), ("A", "C"), ("B", "D")])
        self.assertEqual(adj, {"A": ["B", "C"], "B": ["D"], "C": [], "D": []})

    def test_dfs_order(self):
        adj = lc.graph_from_edges([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
        self.assertEqual(lc.dfs(adj, "A"), ["A", "B", "D", "C"])

    def test_bfs_order(self):
        adj = lc.graph_from_edges([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
        self.assertEqual(lc.bfs(adj, "A"), ["A", "B", "C", "D"])

    def test_isolated_node(self):
        adj = {"A": [], "B": ["A"]}
        self.assertEqual(lc.dfs(adj, "B"), ["B", "A"])
        self.assertEqual(lc.dfs(adj, "A"), ["A"])

    def test_cycle_does_not_loop_forever(self):
        adj = lc.graph_from_edges([("A", "B"), ("B", "A")])
        self.assertEqual(lc.dfs(adj, "A"), ["A", "B"])
        self.assertEqual(lc.bfs(adj, "A"), ["A", "B"])

    def test_missing_start_raises(self):
        with self.assertRaises(ValueError):
            lc.dfs({}, "A")
        with self.assertRaises(ValueError):
            lc.bfs({}, "A")

    def test_reachable(self):
        adj = lc.graph_from_edges([("A", "B"), ("B", "C")])
        self.assertEqual(lc.reachable(adj, "A"), ["A", "B", "C"])
        self.assertEqual(lc.reachable(adj, "C"), ["C"])


# proof validator
# ---------------

class TestProofValidator(unittest.TestCase):

    def test_valid_modus_ponens_proof(self):
        steps = [
            {"rule": "premise", "statement": "P", "refs": []},
            {"rule": "premise", "statement": "P -> Q", "refs": []},
            {"rule": "modus ponens", "statement": "Q", "refs": [1, 2]},
            {"rule": "premise", "statement": "Q -> R", "refs": []},
            {"rule": "modus ponens", "statement": "R", "refs": [3, 4]},
        ]
        reports = lc.validate_proof(steps)
        self.assertTrue(all(ok for _, ok, _ in reports))
        self.assertEqual(len(reports), 5)

    def test_valid_modus_tollens_proof(self):
        steps = [
            {"rule": "premise", "statement": "P -> Q", "refs": []},
            {"rule": "premise", "statement": "NOT Q", "refs": []},
            {"rule": "modus tollens", "statement": "NOT P", "refs": [1, 2]},
        ]
        reports = lc.validate_proof(steps)
        self.assertTrue(all(ok for _, ok, _ in reports))

    def test_wrong_conclusion_is_rejected(self):
        steps = [
            {"rule": "premise", "statement": "P", "refs": []},
            {"rule": "premise", "statement": "P -> Q", "refs": []},
            {"rule": "modus ponens", "statement": "P", "refs": [1, 2]},
        ]
        reports = lc.validate_proof(steps)
        self.assertTrue(reports[0][1] and reports[1][1])
        self.assertFalse(reports[2][1])

    def test_modus_tollens_wrong_form_is_rejected(self):
        steps = [
            {"rule": "premise", "statement": "P -> Q", "refs": []},
            {"rule": "premise", "statement": "NOT P", "refs": []},
            {"rule": "modus tollens", "statement": "NOT Q", "refs": [1, 2]},
        ]
        reports = lc.validate_proof(steps)
        self.assertFalse(reports[2][1])

    def test_unknown_rule_rejected(self):
        steps = [
            {"rule": "premise", "statement": "P", "refs": []},
            {"rule": "quantum jump", "statement": "Q", "refs": [1]},
        ]
        reports = lc.validate_proof(steps)
        self.assertFalse(reports[1][1])

    def test_future_reference_rejected(self):
        steps = [
            {"rule": "modus ponens", "statement": "Q", "refs": [1, 3]},
            {"rule": "premise", "statement": "P", "refs": []},
            {"rule": "premise", "statement": "P -> Q", "refs": []},
        ]
        reports = lc.validate_proof(steps)
        self.assertFalse(reports[0][1])

    def test_wrong_ref_count_rejected(self):
        steps = [
            {"rule": "premise", "statement": "P", "refs": []},
            {"rule": "premise", "statement": "P -> Q", "refs": []},
            {"rule": "modus ponens", "statement": "Q", "refs": [1]},
        ]
        reports = lc.validate_proof(steps)
        self.assertFalse(reports[2][1])

    def test_unparseable_statement_rejected(self):
        steps = [
            {"rule": "premise", "statement": "P AND", "refs": []},
        ]
        reports = lc.validate_proof(steps)
        self.assertFalse(reports[0][1])

    def test_reference_to_invalid_line_rejected(self):
        steps = [
            {"rule": "premise", "statement": "P AND", "refs": []},
            {"rule": "premise", "statement": "P -> Q", "refs": []},
            {"rule": "modus ponens", "statement": "Q", "refs": [1, 2]},
        ]
        reports = lc.validate_proof(steps)
        self.assertFalse(reports[2][1])

    def test_sample_files_load(self):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')
        for name in ('proof_modus_ponens.json', 'proof_modus_tollens.json'):
            steps = lc.load_proof(os.path.join(base, name))
            reports = lc.validate_proof(steps)
            self.assertTrue(all(ok for _, ok, _ in reports),
                            "sample {} should be valid".format(name))


if __name__ == '__main__':
    unittest.main()
