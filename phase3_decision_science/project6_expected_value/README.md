# Project 6 - Expected Value & Kelly Position Sizer

A decision theory library built entirely by hand, from the definitions
themselves.  No numpy, scipy or pandas anywhere in the code.

| Idea | What it answers |
|---|---|
| Expected value | the probability-weighted average of a lottery |
| Decision rules | which action to pick under each criterion |
| Dominance | can an action be discarded before any rule runs? |
| Expected utility | how a risk attitude changes a lottery's worth |
| Certainty equivalent | the guaranteed amount equal to the lottery |
| Risk premium | what the agent pays to avoid the risk |
| Prospect theory | how real people value gains and losses |
| Kelly criterion | the bet size that maximizes long-run growth |

## Expected value

E[X] = sum of probability times outcome.  The same machinery gives the
variance, standard deviation and E[g(X)] for any function g.  Nothing
more than a weighted average, but it is the reference point every
decision rule and every risk measure is built on.

## Decision rules

A payoff matrix lists each action's payoff in each state of the world.
The rules differ in how they treat the states:

| Rule | Criterion | Needs probabilities |
|---|---|---|
| expected-value | maximize E[payoff] | yes |
| maximax | best of the best cases | no |
| maximin | best of the worst cases | no |
| Laplace | average over states | no |
| Hurwicz | alpha times best plus (1-alpha) times worst | no |
| minimax-regret | minimize the biggest regret | no |

Regret in state j is what the best action in that state would have paid
minus what you get, so the minimax-regret rule picks the action with
the smallest maximum regret.  An action that is at least as good in
every state and better in at least one is strictly dominant, and no
reasonable rule ever picks against it, so dominance is checked first.

## Expected utility, certainty equivalent, risk premium

The expected value rule assumes a dollar is a dollar no matter the
context.  Expected utility generalizes it: value each outcome through a
utility function u, then take the probability-weighted average E[u(X)].
For a concave u, risk is disliked: a lottery is worth less than its
expected value.

The certainty equivalent is the amount c with u(c) = E[u(X)], found by
exact inversion of the utility.  The risk premium is E[X] - c, the
amount the agent gives up to swap the lottery for a sure thing.  Both
are exact here because each utility ships its own analytic inverse.

| Utility | Form | Risk aversion |
|---|---|---|
| linear | u = x | none, A = RRA = 0 |
| log | u = ln(x) | RRA = 1 (CRRA) |
| power | u = x^gamma | RRA = 1 - gamma (CRRA) |
| exponential | u = 1 - exp(-alpha x) | A = alpha (CARA) |
| quadratic | u = x - b x^2 | increasing below the peak |

The Arrow-Pratt measures are computed analytically: absolute risk
aversion A = -u''/u' and relative risk aversion RRA = -x u''/u'.  These
are the standard quantities for comparing how risk averse an agent is.

## Prospect theory

The behavioral alternative.  Outcomes are measured against a reference
point, gains are valued x^alpha while losses are valued
-loss_aversion times (-x)^beta so losses loom larger than gains, and
each probability p is distorted through the weighting function, which
overweights small probabilities and underweights large ones.  The
defaults are the parameters estimated by Tversky and Kahneman.

## The St. Petersburg paradox

The game pays 2^k dollars when the first tail lands on toss k.  Its
expected value is infinite, so no finite price is fair by the expected
value rule.  Under log utility the expected utility is finite and the
certainty equivalent converges to 4 dollars.  The paradox is the whole
point: infinite expectation, finite price.  The game here is truncated
to n tosses and renormalized by the probability that a tail appears in
those n tosses, so the probabilities sum to one.

## The Kelly criterion

Expected value says the fair price of a lottery; the Kelly criterion
says how much of your wealth to stake on it.  For a bet that pays odds
dollars per dollar staked with win probability p, the Kelly fraction is
p - (1-p)/odds.  Betting exactly that maximizes the expected log growth
of wealth.  Betting less still grows wealth but with less variance;
betting more lowers the growth rate, and at some point it becomes
negative, so overbetting turns a winning strategy into a losing one.

For an arbitrary set of empirical returns, the growth rate of betting
fraction f is the mean of log(1 + f r) over the returns.  This is a
concave function of f, so the optimal fraction is found by bisecting
where its derivative crosses zero, which is exact rather than a grid
search.  Any fraction that can produce a return of -100% on one round
is excluded, since that bankrupts the bettor.

On the real EUR/USD returns the optimal fraction is about 0.45, and the
expected log growth is positive, so a Kelly bettor would be in the
market.  This is the direct contrast with the invest demo above, where
exponential utility with a strong risk aversion declined the same data.

## Files

- `decision_theory.py` - the library
- `cli.py` - command-line interface
- `test_decision_theory.py` - the tests
- `data/DEXUSEU_returns.csv` - the real EUR/USD daily returns from FRED

## Setup

Python 3.11.  The library needs only the standard library.  The
`verify` command needs numpy.

```bash
pip install --break-system-packages numpy
```

## Usage

```bash
python3 cli.py ev --outcomes 10,20,100 --probs 0.5,0.25,0.25
python3 cli.py decide --payoffs "2,-1,4;3,2,1;1,0,5" --probs 0.3,0.4,0.3 --rule expected-value
python3 cli.py decide --payoffs "2,-1,4;3,2,1;1,0,5" --rule minimax-regret
python3 cli.py dominates --payoffs "1,2,3;1,2,2" --a 0 --b 1
python3 cli.py ce --outcomes 10,30 --probs 0.5,0.5 --utility log
python3 cli.py ce --outcomes 10,30 --probs 0.5,0.5 --utility exponential --param 0.1
python3 cli.py risk --utility power --param 0.5 --x 100
python3 cli.py stpetersburg --tosses 20
python3 cli.py invest data/DEXUSEU_returns.csv --col return --utility exponential --param 10
python3 cli.py verify
```

A semicolon separates the actions in a payoff matrix, a comma separates
the payoffs within an action.

## The invest demo on real data

Treating each of the 6930 daily EUR/USD returns as an equiprobable
outcome of "one day in the market", the expected return is positive
(about 0.0015% per day).  Under exponential utility with alpha 10 the
certainty equivalent is negative, so a risk averse investor declines
the gamble.  This is the equity premium puzzle in miniature: positive
expectation, still not worth it after accounting for the risk.

## Tests

```bash
python3 -m unittest -v
```

52 tests.  Expected values and decision rule scores are checked against
hand computed numbers, each utility for exact inversion and risk
measures, risk premia for monotonicity in risk aversion, prospect
theory for the loss aversion effects, and the St. Petersburg game for
its unbounded expectation and finite certainty equivalent.

## Self-verification

```bash
python3 cli.py verify
```

34 checks.  Expected value and expected utility against numpy, the
decision rules against manual arithmetic, dominance, the risk aversion
measures against their defining values, the loss aversion and
probability weighting properties, the St. Petersburg exact formula, and
the expected utility of the real returns against numpy and the normal
closed form.

## Choices worth knowing about

- The truncated St. Petersburg game is normalized, so its probabilities
  sum to one.  The expected value is then exactly n/(1 - 2^-n).
- Each utility has an exact inverse, so the certainty equivalent is
  computed by inversion rather than numeric root finding, and the
  inversion test is exact.
- The prospect theory implementation weights each outcome's probability
  directly rather than cumulatively over the outcome ordering.  That is
  the simpler variant of the theory and is documented as such.
- Risk aversion measures are analytic per utility family, so A(x) and
  RRA(x) are exact rather than finite-difference approximations.
