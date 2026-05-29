from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PaperState:
    cash: float
    shares: Dict[str, float]          
    last_date: Optional[str] = None   
    equity: float = 0.0               
    pending_orders: Optional[Dict[str, float]] = None
    pending_asof: Optional[str] = None  


def state_to_series_shares(state: PaperState, universe: pd.Index) -> pd.Series:
    s = pd.Series(0.0, index=universe, dtype=float)
    for k, v in state.shares.items():
        if k in s.index:
            s.loc[k] = float(v)
    return s


def mark_to_market(state: PaperState, prices: pd.Series) -> float:
    sh = pd.Series(state.shares, dtype=float).reindex(prices.index).fillna(0.0)
    eq = float(state.cash + np.dot(sh.values, prices.values))
    state.equity = eq
    return eq


def accrue_interest(state: PaperState, days_elapsed: int, annual_rate: float = 0.05) -> float:
    """Accrue simple interest on cash balance."""
    if days_elapsed > 0 and state.cash > 0:
        interest = state.cash * (annual_rate / 365.0) * days_elapsed
        state.cash += interest
        return interest
    return 0.0


def apply_orders_fill(
    state: PaperState,
    orders: pd.Series,
    fill_prices: pd.Series,
    fee_bps: float = 0.0,
) -> Dict[str, float]:
    orders = orders.reindex(fill_prices.index).fillna(0.0)
    fill_prices = fill_prices.astype(float)

    traded_notional = float(np.sum(np.abs(orders.values) * fill_prices.values))
    fees = traded_notional * (fee_bps / 10000.0)

    cash_delta = -float(np.dot(orders.values, fill_prices.values)) - fees
    state.cash += cash_delta

    for sym, dq in orders.items():
        if abs(dq) < 1e-12:
            continue
        state.shares[sym] = float(state.shares.get(sym, 0.0) + dq)
        if abs(state.shares[sym]) < 1e-10:
            state.shares.pop(sym, None)

    return {"traded_notional": traded_notional, "fees": float(fees), "cash_delta": cash_delta}


def load_state(path: str) -> PaperState:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return PaperState(**d)


def save_state(path: str, state: PaperState) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)