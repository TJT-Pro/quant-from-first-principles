from collections import deque
import itertools
import json


# ---------------------------------------------------------------------------
# propositional logic: tokenizer, parser, evaluator, truth tables
# ---------------------------------------------------------------------------

class LogicParseError(Exception):
    """The formula string could not be tokenized or parsed."""


_SYMBOLS = [('<=>', 'IFF'), ('<->', 'IFF'),
            ('=>', 'IMPLIES'), ('->', 'IMPLIES')]

_SINGLE_CHAR = {'!': 'NOT', '~': 'NOT', '&': 'AND', '|': 'OR',
                '^': 'XOR', '(': 'LPAREN', ')': 'RPAREN'}

_KEYWORDS = {'not': 'NOT', 'and': 'AND', 'or': 'OR', 'xor': 'XOR',
             'implies': 'IMPLIES', 'iff': 'IFF'}


def tokenize(text):
    """Split a formula string into (type, value) tokens."""
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isalpha() or ch == '_':
            j = i
            while j < n and (text[j].isalnum() or text[j] == '_'):
                j += 1
            word = text[i:j]
            tokens.append((_KEYWORDS.get(word.lower(), 'VAR'), word))
            i = j
            continue
        matched = False
        for symbol, tok_type in _SYMBOLS:
            if text.startswith(symbol, i):
                tokens.append((tok_type, symbol))
                i += len(symbol)
                matched = True
                break
        if matched:
            continue
        if ch in _SINGLE_CHAR:
            tokens.append((_SINGLE_CHAR[ch], ch))
            i += 1
            continue
        raise LogicParseError(
            "unexpected character {!r} at position {}".format(ch, i))
    tokens.append(('END', ''))
    return tokens


class _Parser:
    """Recursive-descent parser.  AST nodes are tuples:

    ('var', name)
    ('not', x)
    ('and', x, y)  ('or', x, y)  ('xor', x, y)
    ('implies', x, y)  ('iff', x, y)
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, tok_type):
        tok = self.advance()
        if tok[0] != tok_type:
            raise LogicParseError(
                "expected {}, found {!r}".format(tok_type, tok[1]))
        return tok

    def parse(self):
        expr = self.parse_iff()
        if self.peek()[0] != 'END':
            raise LogicParseError(
                "unexpected token {!r}".format(self.peek()[1]))
        return expr

    def parse_iff(self):
        left = self.parse_implies()
        while self.peek()[0] == 'IFF':
            self.advance()
            right = self.parse_implies()
            left = ('iff', left, right)
        return left

    def parse_implies(self):
        left = self.parse_or()
        while self.peek()[0] == 'IMPLIES':
            self.advance()
            right = self.parse_or()
            left = ('implies', left, right)
        return left

    def parse_or(self):
        left = self.parse_xor()
        while self.peek()[0] == 'OR':
            self.advance()
            right = self.parse_xor()
            left = ('or', left, right)
        return left

    def parse_xor(self):
        left = self.parse_and()
        while self.peek()[0] == 'XOR':
            self.advance()
            right = self.parse_and()
            left = ('xor', left, right)
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.peek()[0] == 'AND':
            self.advance()
            right = self.parse_not()
            left = ('and', left, right)
        return left

    def parse_not(self):
        if self.peek()[0] == 'NOT':
            self.advance()
            return ('not', self.parse_not())
        return self.parse_atom()

    def parse_atom(self):
        tok = self.peek()
        if tok[0] == 'VAR':
            self.advance()
            return ('var', tok[1])
        if tok[0] == 'LPAREN':
            self.advance()
            expr = self.parse_iff()
            self.expect('RPAREN')
            return expr
        raise LogicParseError(
            "expected a variable or '(', found {!r}".format(tok[1]))


def parse(formula):
    """Turn a formula string into an AST tuple.  Raises LogicParseError."""
    if not isinstance(formula, str):
        raise TypeError("formula must be a string, got {!r}".format(type(formula)))
    return _Parser(tokenize(formula)).parse()


def evaluate(expr, assignment):
    """Value of an AST under a {variable: bool} assignment."""
    op = expr[0]
    if op == 'var':
        name = expr[1]
        if name not in assignment:
            raise ValueError(
                "no truth value assigned to variable {!r}".format(name))
        return bool(assignment[name])
    if op == 'not':
        return not evaluate(expr[1], assignment)
    if op == 'and':
        return bool(evaluate(expr[1], assignment) and evaluate(expr[2], assignment))
    if op == 'or':
        return bool(evaluate(expr[1], assignment) or evaluate(expr[2], assignment))
    if op == 'xor':
        return bool(evaluate(expr[1], assignment) != evaluate(expr[2], assignment))
    if op == 'implies':
        return bool((not evaluate(expr[1], assignment)) or evaluate(expr[2], assignment))
    if op == 'iff':
        return bool(evaluate(expr[1], assignment) == evaluate(expr[2], assignment))
    raise ValueError("unknown AST node {!r}".format(op))


def variables(expr):
    """Distinct variable names, in order of first appearance."""
    found = []

    def walk(node):
        if node[0] == 'var':
            if node[1] not in found:
                found.append(node[1])
        else:
            for child in node[1:]:
                walk(child)

    walk(expr)
    return found


def truth_table(formula):
    """All 2^n rows for a formula.

    Returns (variable_names, rows).  Each row is a (values, result) pair,
    where values lines up with variable_names.  Rows are ordered by binary
    counting with the leftmost variable as the most significant bit, which
    is how truth tables are normally drawn.
    """
    expr = parse(formula)
    var_names = variables(expr)
    rows = []
    total = 2 ** len(var_names)
    for mask in range(total):
        assignment = {}
        for idx, name in enumerate(var_names):
            bit = (mask >> (len(var_names) - 1 - idx)) & 1
            assignment[name] = bool(bit)
        rows.append((list(assignment.values()), evaluate(expr, assignment)))
    return var_names, rows


def format_truth_table(formula):
    """Print a formula's truth table as a centred grid of text."""
    var_names, rows = truth_table(formula)
    header = list(var_names) + [formula]
    col_width = max([len(str(h)) for h in header] + [len('True'), len('False')]) + 2
    lines = []
    lines.append(''.join(str(h).center(col_width) for h in header))
    lines.append('-' * (col_width * len(header)))
    for vals, result in rows:
        cells = ['True' if v else 'False' for v in vals]
        cells.append('True' if result else 'False')
        lines.append(''.join(c.center(col_width) for c in cells))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# set operations - no set() or frozenset() anywhere
# ---------------------------------------------------------------------------

def make_set(values):
    """Drop duplicates from an iterable, keeping first-seen order.

    A list of unique values and nothing cleverer - no set type involved.
    """
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def union(a, b):
    """A union B: everything in A or in B."""
    result = make_set(a)
    for value in b:
        if value not in result:
            result.append(value)
    return result


def intersection(a, b):
    """A intersect B: elements present in both."""
    return [value for value in make_set(a) if value in b]


def difference(a, b):
    """A \\ B: elements of A that are not in B."""
    return [value for value in make_set(a) if value not in b]


def symmetric_difference(a, b):
    """A delta B: elements in exactly one of A or B."""
    return union(difference(a, b), difference(b, a))


def is_element(value, s):
    """Membership check against a list-set."""
    return value in s


def power_set(s):
    """Every subset of s - all 2^n of them.

    Walks the integer mask from 0 to 2^n-1 and lets bit i of the mask
    decide whether element i comes along.  Mask 0 gives the empty set, the
    last mask gives the whole set, so the order is deterministic.
    """
    unique = make_set(s)
    n = len(unique)
    subsets = []
    for mask in range(1 << n):
        subset = [unique[i] for i in range(n) if (mask >> i) & 1]
        subsets.append(subset)
    return subsets


def cartesian_product(a, b):
    """A x B: every ordered pair (x, y) with x in A and y in B."""
    return [(x, y) for x in make_set(a) for y in make_set(b)]


# ---------------------------------------------------------------------------
# combinatorics: factorial, permutations, combinations
# ---------------------------------------------------------------------------

def factorial(n):
    """n! = n * (n-1) * ... * 2 * 1, with 0! = 1.  Iterative, exact ints."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("factorial: n must be an integer, got {!r}".format(n))
    if n < 0:
        raise ValueError("factorial: n must be non-negative, got {}".format(n))
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _check_n_r(n, r, name):
    if isinstance(n, bool) or isinstance(r, bool) or \
            not isinstance(n, int) or not isinstance(r, int):
        raise TypeError(
            "{}: n and r must be integers, got n={!r}, r={!r}".format(name, n, r))
    if n < 0 or r < 0:
        raise ValueError(
            "{}: n and r must be non-negative, got n={}, r={}".format(name, n, r))
    if r > n:
        raise ValueError(
            "{}: r must not exceed n, got n={}, r={}".format(name, n, r))


def nPr(n, r):
    """Ordered selections: n! / (n-r)!.

    Computed as the falling product n * (n-1) * ... * (n-r+1) rather than
    as n! divided by (n-r)!, so the intermediate numbers stay small.
    Exact for every integer that fits in memory.
    """
    _check_n_r(n, r, 'nPr')
    result = 1
    for i in range(n - r + 1, n + 1):
        result *= i
    return result


def nCr(n, r):
    """Unordered selections: n! / (r! (n-r)!).

    Uses nCr(n, r) = nCr(n, n-r) to shrink r, then the recurrence
    C *= (n - r + i) / i for i = 1..r.  Every intermediate value is itself
    a binomial coefficient, so each division comes out exact.
    """
    _check_n_r(n, r, 'nCr')
    if r > n - r:
        r = n - r
    result = 1
    for i in range(1, r + 1):
        result = result * (n - r + i) // i
    return result


# ---------------------------------------------------------------------------
# graph traversals
# ---------------------------------------------------------------------------

def graph_from_edges(edges):
    """Turn a list of (from, to) pairs into an adjacency dict.

    Every node that appears gets its own key, even nodes with no outgoing
    edges.
    """
    adj = {}
    for source, target in edges:
        adj.setdefault(source, [])
        adj.setdefault(target, [])
        adj[source].append(target)
    return adj


def dfs(adj, start):
    """Depth-first traversal, returning nodes in discovery order.

    Iterative, with an explicit stack.  A dict does the "seen" bookkeeping
    instead of a set, just to stay consistent with the rest of the file.
    """
    if start not in adj:
        raise ValueError("start node {!r} is not in the graph".format(start))
    order = []
    stack = [start]
    seen = {}
    while stack:
        node = stack.pop()
        if seen.get(node):
            continue
        seen[node] = True
        order.append(node)
        for nxt in reversed(adj.get(node, [])):
            if not seen.get(nxt):
                stack.append(nxt)
    return order


def bfs(adj, start):
    """Breadth-first traversal: nodes in level order, nearest first."""
    if start not in adj:
        raise ValueError("start node {!r} is not in the graph".format(start))
    order = []
    queue = deque([start])
    seen = {start: True}
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in adj.get(node, []):
            if not seen.get(nxt):
                seen[nxt] = True
                queue.append(nxt)
    return order


def reachable(adj, start):
    """Every node reachable from start, start included."""
    return make_set(dfs(adj, start))


# ---------------------------------------------------------------------------
# proof validator: modus ponens / modus tollens
# ---------------------------------------------------------------------------

def _is_implication(node):
    return node[0] == 'implies'


def _is_negation(node):
    return node[0] == 'not'


def validate_proof(steps):
    """Check each line of a proof against the rule it claims to use.

    A step is a dict with "rule", "statement" and "refs".  Rules are
    "premise", "modus ponens" or "modus tollens".  A derived line has to
    actually follow from its two references:

        modus ponens:   X  and  X -> Y   give   Y
        modus tollens:  X -> Y  and  NOT Y  give  NOT X

    A wrong conclusion, a bad reference, or a statement that does not parse
    all fail the line.  Returns a (line, ok, message) tuple per step.
    """
    reports = []
    parsed = []  # (ok, ast_or_None) aligned with steps
    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            reports.append((i, False, "step is not a JSON object"))
            parsed.append((False, None))
            continue
        rule = str(step.get('rule', '')).strip().lower()
        statement = step.get('statement')
        refs = step.get('refs', [])

        if not statement:
            reports.append((i, False, "missing 'statement'"))
            parsed.append((False, None))
            continue

        try:
            conc_ast = parse(statement)
        except (LogicParseError, TypeError) as exc:
            reports.append((i, False, "statement does not parse: {}".format(exc)))
            parsed.append((False, None))
            continue

        if rule in ('premise', 'assumption'):
            reports.append((i, True, "premise accepted"))
            parsed.append((True, conc_ast))
            continue

        if rule not in ('modus ponens', 'mp', 'modus tollens', 'mt'):
            reports.append((i, False, "unknown rule {!r}".format(rule)))
            parsed.append((True, conc_ast))
            continue

        if not isinstance(refs, list) or len(refs) != 2:
            reports.append((i, False,
                            "rule {!r} requires exactly two references".format(rule)))
            parsed.append((True, conc_ast))
            continue

        ok_refs = True
        for r in refs:
            if not isinstance(r, int) or isinstance(r, bool):
                ok_refs = False
            elif r < 1 or r >= i:
                ok_refs = False
            elif not parsed[r - 1][0]:
                ok_refs = False
        if not ok_refs:
            reports.append((i, False,
                            "references must be earlier, valid lines "
                            "(got {})".format(refs)))
            parsed.append((True, conc_ast))
            continue

        s1 = parsed[refs[0] - 1][1]
        s2 = parsed[refs[1] - 1][1]

        if rule in ('modus ponens', 'mp'):
            matched = False
            for ante, cond in ((s1, s2), (s2, s1)):
                if _is_implication(cond) and cond[1] == ante and conc_ast == cond[2]:
                    matched = True
                    break
            if matched:
                reports.append((i, True,
                                "modus ponens on lines {} and {}".format(refs[0], refs[1])))
            else:
                reports.append((i, False,
                                "cannot derive {!r} from lines {} and {} by "
                                "modus ponens".format(statement, refs[0], refs[1])))
            parsed.append((True, conc_ast))
            continue

        if rule in ('modus tollens', 'mt'):
            matched = False
            for cond, neg in ((s1, s2), (s2, s1)):
                if _is_implication(cond) and _is_negation(neg) and \
                        neg[1] == cond[2]:
                    expected = ('not', cond[1])
                    if conc_ast == expected:
                        matched = True
                        break
            if matched:
                reports.append((i, True,
                                "modus tollens on lines {} and {}".format(refs[0], refs[1])))
            else:
                reports.append((i, False,
                                "cannot derive {!r} from lines {} and {} by "
                                "modus tollens".format(statement, refs[0], refs[1])))
            parsed.append((True, conc_ast))
            continue

    return reports


def format_proof_report(reports):
    """Render the validation report as readable text for the CLI."""
    lines = []
    for line, ok, message in reports:
        status = 'OK  ' if ok else 'FAIL'
        lines.append("{:>4}  {}  {}".format(line, status, message))
    lines.append('-' * 4 + '------')
    all_ok = all(ok for _, ok, _ in reports)
    lines.append("PROOF VALID" if all_ok else "PROOF INVALID")
    return '\n'.join(lines)


def load_proof(path):
    """Read a proof JSON file - a list of steps, or an object with a
    "steps" key - and return the steps."""
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    if isinstance(data, dict) and 'steps' in data:
        return data['steps']
    if isinstance(data, list):
        return data
    raise ValueError(
        "proof file must be a JSON list of steps, or an object with a "
        "'steps' key")
