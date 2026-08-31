#!/usr/bin/env python3
"""Command-line front end for the Logic & Combinatorics toolkit.

    python3 cli.py truth-table "FORMULA"
    python3 cli.py factorial N
    python3 cli.py perm N R
    python3 cli.py comb N R
    python3 cli.py sets OP --a A B C [--b D E F]
    python3 cli.py graph dfs|bfs START --edges A B C D ...
    python3 cli.py proof FILE.json
    python3 cli.py verify

Set operations: union, intersection, difference, sym-diff, power-set,
product.  Graph traversals: dfs, bfs.
"""

import argparse
import sys

from logic_combinatorics import (
    LogicParseError,
    nPr,
    nCr,
    factorial,
    parse,
    format_truth_table,
    make_set,
    union,
    intersection,
    difference,
    symmetric_difference,
    power_set,
    cartesian_product,
    graph_from_edges,
    dfs,
    bfs,
    reachable,
    load_proof,
    validate_proof,
    format_proof_report,
)


def _maybe_int(value):
    """Convert a CLI argument to int when possible, else keep it a string."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def cmd_truth_table(args):
    print(format_truth_table(args.formula))


def cmd_factorial(args):
    print(factorial(args.n))


def cmd_perm(args):
    print(nPr(args.n, args.r))


def cmd_comb(args):
    print(nCr(args.n, args.r))


def cmd_sets(args):
    a = [_maybe_int(x) for x in args.a]
    b = [_maybe_int(x) for x in args.b] if args.b else []
    op = args.op.lower().replace('_', '-')
    if op in ('union', 'u'):
        result = union(a, b)
    elif op in ('intersection', 'intersect', 'i'):
        result = intersection(a, b)
    elif op in ('difference', 'diff', 'd'):
        result = difference(a, b)
    elif op in ('sym-diff', 'symmetric-difference', 'symdiff'):
        result = symmetric_difference(a, b)
    elif op in ('power-set', 'powerset', 'p'):
        result = power_set(a)
    elif op in ('product', 'cartesian', 'x'):
        if not args.b:
            sys.exit("cartesian product needs both --a and --b")
        result = cartesian_product(a, b)
    else:
        sys.exit("unknown set operation {!r}".format(args.op))
    print(result)


def cmd_graph(args):
    edges = [_maybe_int(x) for x in args.edges]
    start = _maybe_int(args.start)
    if len(edges) % 2 != 0:
        sys.exit("--edges must contain an even number of values (from to from to ...)")
    pairs = list(zip(edges[0::2], edges[1::2]))
    adj = graph_from_edges(pairs)
    if args.traversal == 'dfs':
        order = dfs(adj, start)
    elif args.traversal == 'bfs':
        order = bfs(adj, start)
    else:
        sys.exit("unknown traversal {!r}".format(args.traversal))
    print("adjacency: {}".format(adj))
    print("{} order : {}".format(args.traversal.upper(), order))
    print("reachable: {}".format(reachable(adj, start)))


def cmd_proof(args):
    steps = load_proof(args.file)
    report = validate_proof(steps)
    print(format_proof_report(report))
    ok = all(ok for _, ok, _ in report)
    sys.exit(0 if ok else 1)


def cmd_verify(args):
    """Run a battery of self-checks.

    scipy and Python's own set() are used here purely as independent
    oracles to compare against - the implementations themselves never
    import them."""
    import math
    failures = []
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        if not ok:
            failures.append(name)

    # --- factorial vs math.factorial ------------------------------------
    bad = [n for n in range(0, 51) if factorial(n) != math.factorial(n)]
    check("factorial(0..50) == math.factorial",
          not bad, "mismatches: {}".format(bad[:5]))

    # --- nPr / nCr vs scipy.special -------------------------------------
    try:
        from scipy.special import comb, perm
        bad_p = []
        bad_c = []
        for n in range(0, 12):
            for r in range(0, n + 1):
                if nPr(n, r) != int(perm(n, r)):
                    bad_p.append((n, r))
                if nCr(n, r) != int(comb(n, r, exact=True)):
                    bad_c.append((n, r))
        check("nPr matches scipy.special.perm", not bad_p, "mismatches: {}".format(bad_p[:5]))
        check("nCr matches scipy.special.comb", not bad_c, "mismatches: {}".format(bad_c[:5]))
    except ImportError:
        check("nPr / nCr vs scipy", False, "scipy not installed, skipped")

    # --- truth tables vs hand-evaluated booleans ------------------------
    tt_cases = [
        ("P AND Q", ["P", "Q"], lambda p, q: p and q),
        ("P OR NOT Q", ["P", "Q"], lambda p, q: p or (not q)),
        ("(P AND Q) OR NOT R", ["P", "Q", "R"], lambda p, q, r: (p and q) or (not r)),
        ("P -> Q", ["P", "Q"], lambda p, q: (not p) or q),
        ("P XOR Q", ["P", "Q"], lambda p, q: p != q),
        ("P <-> Q", ["P", "Q"], lambda p, q: p == q),
    ]
    for formula, names, fn in tt_cases:
        var_names, rows = __import__('logic_combinatorics').truth_table(formula)
        ok = (var_names == names)
        for vals, result in rows:
            assignment = dict(zip(names, vals))
            ok = ok and result == bool(fn(*vals))
        check("truth table: {}".format(formula), ok)

    # --- set engine vs Python set() reference ---------------------------
    A = [1, 2, 3, 2]
    B = [3, 4, 5]
    check("make_set deduplicates", make_set(A) == [1, 2, 3])
    check("union", make_set(union(A, B)) == make_set(list(set(A) | set(B))))
    check("intersection", make_set(intersection(A, B)) == make_set(list(set(A) & set(B))))
    check("difference", make_set(difference(A, B)) == make_set(list(set(A) - set(B))))
    check("symmetric difference",
          make_set(symmetric_difference(A, B)) == make_set(list(set(A) ^ set(B))))
    ps = power_set([1, 2, 3])
    check("power_set has 2^n subsets", len(ps) == 8)
    check("power_set contains empty and full set", [] in ps and [1, 2, 3] in ps)
    cp = cartesian_product([1, 2], [3, 4])
    check("cartesian product size", len(cp) == 4 and (1, 3) in cp and (2, 4) in cp)

    # --- graph traversals ----------------------------------------------
    adj = graph_from_edges([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
    check("dfs order", dfs(adj, 'A') == ['A', 'B', 'D', 'C'],
          "got {}".format(dfs(adj, 'A')))
    check("bfs order", bfs(adj, 'A') == ['A', 'B', 'C', 'D'],
          "got {}".format(bfs(adj, 'A')))
    check("reachable", make_set(reachable(adj, 'A')) == ['A', 'B', 'D', 'C'],
          "got {}".format(reachable(adj, 'A')))

    # --- proof validator on the sample proofs ---------------------------
    import os
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')
    for name in ('proof_modus_ponens.json', 'proof_modus_tollens.json'):
        steps = load_proof(os.path.join(base, name))
        ok = all(ok for _, ok, _ in validate_proof(steps))
        check("proof: {}".format(name), ok)
    bad_steps = [
        {"rule": "premise", "statement": "P", "refs": []},
        {"rule": "premise", "statement": "P -> Q", "refs": []},
        {"rule": "modus ponens", "statement": "Q", "refs": [1, 2]},
        {"rule": "premise", "statement": "Q -> R", "refs": []},
        {"rule": "modus ponens", "statement": "R", "refs": [3, 4]},
        {"rule": "modus ponens", "statement": "P", "refs": [3, 4]},
    ]
    reports = validate_proof(bad_steps)
    ok = all(ok for _, ok, _ in reports[:-1]) and reports[-1][1] is False
    check("proof validator rejects a wrong step", ok)

    width = max(len(name) for name, _, _ in checks) + 2
    for name, ok, detail in checks:
        marker = "PASS" if ok else "FAIL"
        print("{:<{w}} {}".format(marker, name, w=6))
        if not ok and detail:
            print("      {}".format(detail))
    print("-" * 60)
    print("{}/{} checks passed".format(len(checks) - len(failures), len(checks)))
    sys.exit(1 if failures else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="logic-combinatorics",
        description="Logic & Combinatorics Toolkit (Project 1)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_tt = sub.add_parser("truth-table", help="print a truth table for a formula")
    p_tt.add_argument("formula")
    p_tt.set_defaults(func=cmd_truth_table)

    p_fact = sub.add_parser("factorial", help="compute n!")
    p_fact.add_argument("n", type=int)
    p_fact.set_defaults(func=cmd_factorial)

    p_perm = sub.add_parser("perm", help="compute nPr")
    p_perm.add_argument("n", type=int)
    p_perm.add_argument("r", type=int)
    p_perm.set_defaults(func=cmd_perm)

    p_comb = sub.add_parser("comb", help="compute nCr")
    p_comb.add_argument("n", type=int)
    p_comb.add_argument("r", type=int)
    p_comb.set_defaults(func=cmd_comb)

    p_sets = sub.add_parser("sets", help="set operations engine")
    p_sets.add_argument("op", help="union|intersection|difference|"
                                   "sym-diff|power-set|product")
    p_sets.add_argument("--a", nargs="+", required=True,
                        help="set A elements")
    p_sets.add_argument("--b", nargs="+", help="set B elements")
    p_sets.set_defaults(func=cmd_sets)

    p_graph = sub.add_parser("graph", help="graph traversals")
    p_graph.add_argument("traversal", help="dfs|bfs")
    p_graph.add_argument("start", help="start node")
    p_graph.add_argument("--edges", nargs="+", required=True,
                         help="flat from/to pairs")
    p_graph.set_defaults(func=cmd_graph)

    p_proof = sub.add_parser("proof", help="validate a proof JSON file")
    p_proof.add_argument("file")
    p_proof.set_defaults(func=cmd_proof)

    p_verify = sub.add_parser("verify", help="run self-checks vs scipy")
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (LogicParseError, ValueError, TypeError) as exc:
        sys.exit("error: {}".format(exc))


if __name__ == '__main__':
    main()
