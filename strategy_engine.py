from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    benchmark: str = "SPY"
    lookback_short: int = 63
    lookback_long: int = 126
    w_short: float = 0.5
    w_long: float = 0.5
    top_n: int = 10

    ma_window: int = 200
    risk_off_exposure: float = 0.2
    cost_bps: float = 5.0  

    strict_no_same_close: bool = True


def month_end_rebalance_mask(index: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(index, index=index)
    month_ends = s.resample("ME").last().values
    return index.isin(month_ends)


def precompute_score_matrix(close: pd.DataFrame, cfg: StrategyConfig) -> np.ndarray:
    ret = close.pct_change()
    ret_s = close / close.shift(cfg.lookback_short) - 1.0
    ret_l = close / close.shift(cfg.lookback_long) - 1.0

    vol_s = ret.rolling(cfg.lookback_short).std(ddof=0) * np.sqrt(252)
    vol_l = ret.rolling(cfg.lookback_long).std(ddof=0) * np.sqrt(252)

    vol_s = vol_s.clip(lower=1e-8)
    vol_l = vol_l.clip(lower=1e-8)

    score = cfg.w_short * (ret_s / vol_s) + cfg.w_long * (ret_l / vol_l)
    return score.to_numpy()


def precompute_regime_multiplier(close: pd.DataFrame, cfg: StrategyConfig) -> np.ndarray:
    px = close[cfg.benchmark]
    ma = px.rolling(cfg.ma_window).mean()
    risk_on = (px >= ma)
    mult = np.where(risk_on.to_numpy(), 1.0, float(cfg.risk_off_exposure))
    mult = np.where(np.isfinite(ma.to_numpy()), mult, 1.0)
    return mult.astype(float)


def top_n_equal_weight(score_row: np.ndarray, top_n: int) -> np.ndarray:
    n = score_row.shape[0]
    w = np.zeros(n, dtype=float)
    if top_n <= 0:
        return w

    s = np.array(score_row, dtype=float)
    s[~np.isfinite(s)] = -np.inf
    if np.all(np.isneginf(s)):
        return w

    k = min(top_n, n)
    idx_k = np.argpartition(s, -k)[-k:]
    idx_k = idx_k[np.argsort(s[idx_k])[::-1]]
    w[idx_k] = 1.0 / k
    return w


def target_weights_on_date(
    close: pd.DataFrame,
    asof: pd.Timestamp,
    cfg: StrategyConfig,
    score_matrix: Optional[np.ndarray] = None,
    regime_mult: Optional[np.ndarray] = None,
    reb_mask: Optional[np.ndarray] = None,
) -> pd.Series:
    close = close.sort_index()
    idx = close.index

    if score_matrix is None:
        score_matrix = precompute_score_matrix(close, cfg)
    if regime_mult is None:
        regime_mult = precompute_regime_multiplier(close, cfg)
    if reb_mask is None:
        reb_mask = month_end_rebalance_mask(idx)

    i = idx.get_loc(asof)
    if not reb_mask[i]:
        raise ValueError("asof is not a rebalance date under the current calendar")

    # FIX: Since pending orders are executed the following day at the Open, 
    # using signal calculated at Close `i` introduces NO lookahead bias.
    j = i 

    w_sig = top_n_equal_weight(score_matrix[j, :], cfg.top_n)
    w = w_sig * float(regime_mult[j])
    return pd.Series(w, index=close.columns)


def weights_to_target_shares(
    weights: pd.Series,
    prices: pd.Series,
    total_value: float,
    allow_fractional: bool = True,
) -> pd.Series:
    weights = weights.reindex(prices.index).fillna(0.0)
    prices = prices.astype(float)

    dollar = weights * float(total_value)
    shares = dollar / prices.replace(0.0, np.nan)

    shares = shares.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if not allow_fractional:
        shares = np.floor(shares)

    return shares


def shares_to_orders(current_shares: pd.Series, target_shares: pd.Series) -> pd.Series:
    cs = current_shares.reindex(target_shares.index).fillna(0.0)
    return (target_shares - cs).astype(float)


def log_rebalance_jsonl(
    path: str,
    asof: pd.Timestamp,
    weights: pd.Series,
    prices: pd.Series,
    target_shares: pd.Series,
    orders: pd.Series,
    cfg: StrategyConfig,
    extra: Optional[dict] = None,
) -> None:
    rec = {
        "asof": str(asof),
        "config": asdict(cfg),
        "weights": weights.dropna().to_dict(),
        "prices": prices.dropna().to_dict(),
        "target_shares": target_shares.dropna().to_dict(),
        "orders": orders.dropna().to_dict(),
    }
    if extra:
        rec["extra"] = extra
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")