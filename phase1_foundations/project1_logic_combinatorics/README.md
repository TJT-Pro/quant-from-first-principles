# Project 1 - Logic & Combinatorics Toolkit

Command-line toolkit covering the first chunk of the roadmap: propositional
logic, sets, combinatorics, graphs, and a small proof checker.  Everything
is implemented by hand.

| Component | What it does | Uses numpy / scipy / set()? |
|---|---|---|
| Truth table generator | Parses a propositional formula and prints its truth table | No |
| Set operations | union, intersection, difference, symmetric difference, power set, cartesian product | No - not even Python's `set()` |
| Combinatorics | factorial, nPr, nCr, exact integer arithmetic | No |
| Graph traversals | DFS and BFS on a directed graph (adjacency dict) | No |
| Proof validator | Checks lines against modus ponens and modus tollens | No |

scipy and Python's own `set()` only show up as independent reference
points in the `verify` command and in the tests.  The implementations
never touch them.

---

## The math

### Propositional logic

Formulas are built from variables (`P`, `Q`, ...) and connectives.  From
strongest to weakest binding:

| Operator | Symbols | True when |
|---|---|---|
| NOT | `NOT`, `!`, `~` | its operand is False |
| AND | `AND`, `&` | both operands are True |
| XOR | `XOR`, `^` | exactly one operand is True |
| OR | `OR`, `\|` | at least one operand is True |
| IMPLIES | `->`, `=>` | not (True premise, False conclusion) |
| IFF | `<->`, `<=>` | both operands agree |

Binary operators are left-associative.  Parentheses beat everything.
Variables start with a letter or underscore and can contain letters,
digits and underscores (`P`, `rain`, `A1`).  The keywords (`and`, `or`,
...) are reserved.

A formula with `n` variables has `2^n` rows in its truth table.  Rows are
listed in binary-counting order with the leftmost variable as the most
significant bit - the usual way truth tables are drawn.

### Sets

A set is a collection of distinct objects.  Here each set is just a list
of unique values, in first-seen order, and every operation is written with
plain list membership.  Python's `set()` type is deliberately never used.

| Operation | Notation | Meaning |
|---|---|---|
| Union | `A ∪ B` | everything in A or B |
| Intersection | `A ∩ B` | only what's in both |
| Difference | `A \ B` | A minus what's in B |
| Symmetric difference | `A Δ B` | what's in exactly one of them |
| Power set | `P(A)` | all `2^|A|` subsets |
| Cartesian product | `A × B` | all ordered pairs `(x, y)` |

The power set walks the integer mask `0 .. 2^n - 1`; bit `i` of the mask
says whether element `i` is in that subset.  That guarantees `2^n`
subsets and a deterministic order.

### Combinatorics

**Factorial.** `n! = n * (n-1) * ... * 1`, and `0! = 1`.

**Permutations** - ordered selections:

    nPr = n! / (n-r)! = n * (n-1) * ... * (n-r+1)

Computed as the falling product on the right so the intermediates stay
small.

**Combinations** - unordered selections:

    nCr = n! / (r! (n-r)!)

Computed using `nCr(n, r) = nCr(n, n-r)` to shrink `r`, then the
recurrence `C <- C * (n - r + i) / i` for `i = 1..r`.  Every intermediate
value is itself a binomial coefficient, so each division is exact.  The
identity `nPr = nCr * r!` is checked in the tests.

### Graph traversals

A directed graph is an adjacency dict, `{node: [out-neighbours]}`.

- **DFS** uses a stack; it dives down each branch before backtracking.
  Returns discovery order.
- **BFS** uses a queue; it visits nodes level by level, nearest first.

Both remember what they've seen, so they terminate on cycles too.
`reachable` returns every node you can get to from a start node,
including the start.

### Proof validation

A proof is a numbered list of lines, each a premise or something derived
from earlier lines.  Two rules are checked:

| Rule | From | Infer |
|---|---|---|
| Modus ponens | `X`, `X -> Y` | `Y` |
| Modus tollens | `X -> Y`, `NOT Y` | `NOT X` |

Validation works on the parsed structure, not strings, so it's exact.
A derived line fails if its conclusion doesn't literally follow from its
two references.  References must point to earlier, valid lines.

---

## Files

- `logic_combinatorics.py` - all the implementations
- `cli.py` - the command-line interface
- `test_logic_combinatorics.py` - the tests
- `samples/proof_modus_ponens.json`, `samples/proof_modus_tollens.json`

## Setup

Python 3.11.  The toolkit itself only needs the standard library.  The
`verify` command and the cross-check tests additionally want scipy:

```bash
pip install --break-system-packages scipy numpy matplotlib
```

## Usage

Run these from inside the project folder:

```bash
python3 cli.py truth-table "(P AND Q) OR NOT R"
python3 cli.py factorial 10
python3 cli.py perm 10 3
python3 cli.py comb 49 6

python3 cli.py sets union --a 1 2 3 2 --b 3 4
python3 cli.py sets intersection --a 1 2 3 --b 2 3 4
python3 cli.py sets difference --a 1 2 3 --b 2
python3 cli.py sets sym-diff --a 1 2 --b 2 3
python3 cli.py sets power-set --a 1 2 3
python3 cli.py sets product --a 1 2 --b a b

python3 cli.py graph dfs A --edges A B A C B D C D
python3 cli.py graph bfs A --edges A B A C B D C D

python3 cli.py proof samples/proof_modus_ponens.json
python3 cli.py proof samples/proof_modus_tollens.json
```

`proof` exits `0` for a valid proof and `1` for an invalid one, so it can
be used in scripts.

## Tests

```bash
python3 -m unittest -v
```

Covers: parser precedence and errors, every connective's evaluation,
truth tables (including the roadmap's `(P AND Q) OR NOT R`), set
operations (with a guard that the engine never calls `set()`),
combinatorics against known values and the `nPr = nCr * r!` identity and
`scipy.special`, graph traversals (cycle safety included), and the proof
validator's valid and invalid cases.

## Self-verification

```bash
python3 cli.py verify
```

Cross-checks the hand-written code against `math.factorial`,
`scipy.special.perm`/`comb`, hand-evaluated truth tables, and Python's
`set()` used as a reference.  Prints PASS/FAIL per check and exits
non-zero on any failure.

## Choices worth knowing about

- Formulas are parsed into an AST, so evaluation and proof checking work
  on structure, not strings.  That's what makes the proof validator exact.
- No `set()` in the engine, per the roadmap's constraint.  A dict does
  the "seen" bookkeeping in the graph code.
- All combinatorics are exact integers - no floats anywhere.
- Output order is deterministic and documented (truth-table rows,
  power-set subsets, set membership order) so runs are reproducible.
