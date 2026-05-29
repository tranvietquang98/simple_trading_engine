from __future__ import annotations

import os
import json
import logging
import pandas as pd
from dataclasses import asdict

from strategy_engine import (
    StrategyConfig,
    month_end_rebalance_mask,
    target_weights_on_date,
    weights_to_target_shares,
    shares_to_orders,
    log_rebalance_jsonl,
)
from paper_portfolio import (
    PaperState,
    load_state,
    save_state,
    state_to_series_shares,
    mark_to_market,
    apply_orders_fill,
    accrue_interest,
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

INITIAL_CASH = 1000.0
PAPER_FEE_BPS = 5.0      # 5 bps realistic execution slippage
RISK_FREE_RATE = 0.05    # 5% annualized rate for cash drag offset

BASE_CFG = dict(
    benchmark="SPY",
    lookback_short=63,
    lookback_long=126,
    w_short=0.5,
    w_long=0.5,
    ma_window=200,
    risk_off_exposure=0.2,
    cost_bps=0.0,
    strict_no_same_close=True,
)

STRATS = {
    "top5": StrategyConfig(**BASE_CFG, top_n=5),
    "top10": StrategyConfig(**BASE_CFG, top_n=10),
}

STATE_PATHS = {k: f"state/paper_state_{k}.json" for k in STRATS}
LOG_PATHS = {k: f"logs/paper_log_{k}.jsonl" for k in STRATS}


def load_prices_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def ensure_state(path: str) -> PaperState:
    if os.path.exists(path):
        return load_state(path)
    return PaperState(cash=float(INITIAL_CASH), shares={}, last_date=None, equity=float(INITIAL_CASH))


def log_mtm_jsonl(path: str, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def run_one_strategy_one_day(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    asof: pd.Timestamp,
    name: str,
    cfg: StrategyConfig,
) -> dict:
    state_path = STATE_PATHS[name]
    log_path = LOG_PATHS[name]
    state = ensure_state(state_path)

    # 1. Accrue Interest on Cash
    if state.last_date:
        days_elapsed = (asof.date() - pd.to_datetime(state.last_date).date()).days
        if days_elapsed > 0:
            accrue_interest(state, days_elapsed, RISK_FREE_RATE)

    # 2. Execute Pending Orders on Today's OPEN
    if state.pending_orders:
        prices_open = open_px.loc[asof].astype(float)
        pending = pd.Series(state.pending_orders, index=open_px.columns).fillna(0.0)
        apply_orders_fill(state, pending, prices_open, PAPER_FEE_BPS)
        
        state.pending_orders = None
        state.pending_asof = None
        logging.info(f"[{name}] Filled pending orders at {asof.date()} Open.")

    # 3. Mark-to-Market on Today's CLOSE
    prices_close = close.loc[asof].astype(float)
    eq_close = mark_to_market(state, prices_close)

    idx = close.index
    reb_mask = month_end_rebalance_mask(idx)
    i = idx.get_loc(asof)

    cal_month_end = (asof + pd.offsets.MonthEnd(0))
    last_bday = (cal_month_end - pd.tseries.offsets.BDay(0))
    is_reb = bool(reb_mask[i]) and (asof.normalize() >= last_bday.normalize())

    summary = {
        "strategy": name,
        "asof": str(asof.date()),
        "is_rebalance": is_reb,
        "equity_after": float(eq_close),
        "cash_after": float(state.cash),
    }

    # 4. Generate New Signals on Today's CLOSE (to be executed tomorrow)
    if is_reb:
        w_target = target_weights_on_date(close, asof=asof, cfg=cfg)
        target_shares = weights_to_target_shares(
            weights=w_target,
            prices=prices_close,
            total_value=eq_close,
            allow_fractional=True,
        )

        current_shares = state_to_series_shares(state, close.columns)
        orders = shares_to_orders(current_shares, target_shares)

        state.pending_orders = orders.to_dict()
        state.pending_asof = str(asof.date())

        exec_info = {"traded_notional": float((orders.abs() * prices_close).sum()), "fees": 0.0}

        summary |= {
            "traded_notional": float(exec_info["traded_notional"]),
            "fees": float(exec_info["fees"]),
            "gross_exposure": float(w_target.abs().sum()),
            "nonzero_assets": int((w_target.abs() > 1e-12).sum()),
        }

        log_rebalance_jsonl(
            path=log_path, asof=asof, weights=w_target, prices=prices_close,
            target_shares=target_shares, orders=orders, cfg=cfg, extra=summary,
        )
        logging.info(f"[{name}] Rebalance calculated. Orders queued for next open.")
    else:
        log_mtm_jsonl(log_path, {"type": "mtm", **summary})

    state.last_date = str(asof.date())
    save_state(state_path, state)

    return summary


def run_parallel_for_date(close: pd.DataFrame, open_px: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for name, cfg in STRATS.items():
        rows.append(run_one_strategy_one_day(close, open_px, asof, name, cfg))
    return pd.DataFrame(rows).set_index("strategy")


if __name__ == "__main__":
    close = load_prices_from_csv("data/close_prices.csv")
    open_px = load_prices_from_csv("data/open_prices.csv")

    asof = close.index[-1]
    logging.info(f"Running paper execution for date: {asof.date()}")

    out = run_parallel_for_date(close, open_px, asof)

    cols = ["asof", "is_rebalance", "equity_after", "cash_after"]
    extra = [c for c in ["traded_notional", "fees", "gross_exposure", "nonzero_assets"] if c in out.columns]
    
    print("\n" + out[cols + extra].to_string(float_format=lambda x: f"{x:,.2f}"))