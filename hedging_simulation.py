import math
import statistics
from typing import List, Dict, Optional

import black_scholes as bs


def reconstruct_price_path(log_returns: List[float], s0: float = 1.0) -> List[float]:
    """Rebuild a price path from log returns: S_t = S_0 * exp(sum of
    log returns up to t). Starting from an arbitrary S_0=1.0 keeps the
    analysis in percentage terms, independent of the actual EUR/USD
    level, which is irrelevant to a hedging P&L study expressed
    relative to the notional traded.
    """
    prices = [s0]
    cumulative = 0.0
    for r in log_returns:
        cumulative += r
        prices.append(s0 * math.exp(cumulative))
    return prices


def annualized_realized_volatility(log_returns: List[float], trading_days: int = 252) -> float:
    if len(log_returns) < 2:
        raise ValueError("need at least two returns to estimate a volatility")
    return statistics.pstdev(log_returns) * math.sqrt(trading_days)


def simulate_delta_hedge(prices: List[float], entry_index: int, horizon_days: int, k: float,
                          implied_vol: float, r: float, q: float,
                          option_type: str = "call") -> Dict[str, float]:
    """Simulate daily delta-hedging of a short European option, using
    the actual subsequent price path (real historical data, not a
    model simulation), with the hedge itself computed from a *fixed*
    implied_vol assumption throughout the option's life.

    The self-financing replication argument behind the Black-Scholes
    PDE is implemented directly, day by day, rather than invoked as a
    formula: sell the option for its model premium V_0, buy delta_0
    units of the underlying to hedge, and hold the remainder in a cash
    account. Each day, the cash account earns interest at the domestic
    rate r and collects the continuous yield q on the currently-held
    underlying position (the FX analogue of dividend income); the
    hedge is then rebalanced to the new delta implied by the actual
    new spot and the shrinking time to maturity, still under the same
    fixed implied_vol. At maturity, the option is settled against its
    true payoff, and the final portfolio value (cash plus remaining
    stock) minus that payoff is the hedging P&L: this is exactly zero
    if and only if the realized path behaves precisely as the
    constant-volatility model assumed, and nonzero to the extent it
    does not.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if entry_index + horizon_days >= len(prices):
        raise ValueError("not enough price history for this horizon")

    dt = 1.0 / 252
    total_t = horizon_days * dt
    price_fn = bs.call_price if option_type == "call" else bs.put_price
    delta_fn = bs.call_delta if option_type == "call" else bs.put_delta

    s = prices[entry_index]
    tau = total_t
    premium = price_fn(s, k, tau, r, implied_vol, q)
    delta = delta_fn(s, k, tau, r, implied_vol, q)
    cash = premium - delta * s

    for day in range(1, horizon_days + 1):
        s_new = prices[entry_index + day]
        tau_new = total_t - day * dt

        cash *= math.exp(r * dt)
        cash += delta * q * s * dt

        if tau_new > 1e-9:
            delta_new = delta_fn(s_new, k, tau_new, r, implied_vol, q)
        else:
            if option_type == "call":
                delta_new = 1.0 if s_new > k else 0.0
            else:
                delta_new = -1.0 if s_new < k else 0.0

        cash -= (delta_new - delta) * s_new
        delta = delta_new
        s = s_new

    payoff = max(s - k, 0.0) if option_type == "call" else max(k - s, 0.0)
    final_portfolio = cash + delta * s
    pnl = final_portfolio - payoff

    window_returns = []
    for i in range(entry_index, entry_index + horizon_days):
        window_returns.append(math.log(prices[i + 1] / prices[i]))
    realized_vol = annualized_realized_volatility(window_returns)

    return {
        "pnl": pnl,
        "premium": premium,
        "realized_vol": realized_vol,
        "implied_vol": implied_vol,
    }


def rolling_hedge_backtest(prices: List[float], horizon_days: int, implied_vol: float,
                            r: float, q: float, option_type: str = "call",
                            step: Optional[int] = None) -> List[Dict[str, float]]:
    """Run `simulate_delta_hedge` across many non-overlapping (by
    default) entry points spanning the available price history, each
    entered at-the-money (strike equal to the spot at entry).
    Non-overlapping windows are used by default so that the resulting
    P&L observations are close to independent, which matters for
    interpreting their average and spread honestly.
    """
    if step is None:
        step = horizon_days
    results = []
    entry = 0
    while entry + horizon_days < len(prices):
        k = prices[entry]
        result = simulate_delta_hedge(prices, entry, horizon_days, k, implied_vol, r, q, option_type)
        result["entry_index"] = entry
        results.append(result)
        entry += step
    return results
