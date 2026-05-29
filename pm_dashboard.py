from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------
# CONFIG / PATHS
# -----------------
CLOSE_CSV = r"data\close_prices.csv"

LOG_TOP5 = r"logs\paper_log_top5.jsonl"
LOG_TOP10 = r"logs\paper_log_top10.jsonl"

STATE_TOP5 = r"state\paper_state_top5.json"
STATE_TOP10 = r"state\paper_state_top10.json"

BENCH = "SPY"
MA_WINDOW = 200

ROLL_WIN = 252  # rolling 1Y window for Sharpe/vol


# -----------------
# Helpers
# -----------------
def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_daily_ledger(rows: list[dict]) -> pd.DataFrame:
    """
    Supports:
      - MTM lines: {"type":"mtm", ... flat ...}
      - Rebalance lines from log_rebalance_jsonl: {..., "extra": {...}}
    Returns daily df indexed by asof with at least:
      equity_after, equity_before, is_rebalance, traded_notional, fees
    """
    out = []
    for r in rows:
        if r.get("type") == "mtm":
            rec = dict(r)
        else:
            rec = {"asof": r.get("asof")}
            extra = r.get("extra", {}) or {}
            rec.update(extra)
            # fallbacks
            rec["is_rebalance"] = rec.get("is_rebalance", True)
            rec["traded_notional"] = rec.get("traded_notional", 0.0)
            rec["fees"] = rec.get("fees", 0.0)

        if rec.get("asof") is None:
            continue
        out.append(rec)

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out)
    df["asof"] = pd.to_datetime(df["asof"])
    df = df.sort_values("asof").drop_duplicates("asof", keep="last").set_index("asof")

    for c in ["equity_after", "equity_before", "cash_after", "cash_before",
              "is_rebalance", "traded_notional", "fees"]:
        if c not in df.columns:
            df[c] = np.nan

    df["is_rebalance"] = df["is_rebalance"].fillna(False).astype(bool)
    df["traded_notional"] = df["traded_notional"].fillna(0.0).astype(float)
    df["fees"] = df["fees"].fillna(0.0).astype(float)

    # fill equity_after if missing
    df["equity_after"] = df["equity_after"].fillna(df["equity_before"])
    return df


def load_close(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def scaled_equity_from_prices(px: pd.Series, start_value: float = 1000.0) -> pd.Series:
    px = px.dropna()
    if len(px) == 0:
        return px
    return (px / px.iloc[0] * start_value).rename(px.name)


def compute_metrics_from_equity(equity: pd.Series, ann: int = 252) -> dict:
    equity = equity.dropna()
    if len(equity) < 3:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}

    ret = equity.pct_change().dropna()
    ann_return = (1.0 + ret).prod() ** (ann / len(ret)) - 1.0
    ann_vol = ret.std(ddof=0) * np.sqrt(ann)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
    }


def drawdown(equity: pd.Series) -> pd.Series:
    eq = equity.dropna()
    if len(eq) == 0:
        return eq
    peak = eq.cummax()
    return (eq / peak - 1.0).rename("drawdown")


def rolling_sharpe(equity: pd.Series, win: int = 252, ann: int = 252) -> pd.Series:
    eq = equity.dropna()
    if len(eq) < win + 2:
        return pd.Series(index=eq.index, dtype=float, name="roll_sharpe")
    r = eq.pct_change()
    mu = r.rolling(win).mean() * ann
    sig = r.rolling(win).std(ddof=0) * np.sqrt(ann)
    out = (mu / sig).replace([np.inf, -np.inf], np.nan)
    return out.rename("roll_sharpe")


def monthly_returns(equity: pd.Series) -> pd.Series:
    eq = equity.dropna()
    if len(eq) == 0:
        return eq
    # Use "ME" to avoid pandas freq issues
    m = eq.resample("ME").last().pct_change()
    return m.rename("mret")


def monthly_table(mrets: pd.DataFrame, last_n_months: int = 24) -> pd.DataFrame:
    """
    mrets: columns = strategies, index = month-end timestamps
    returns a pivot-like table year x month with percentages.
    """
    x = mrets.copy().tail(last_n_months)
    if x.dropna(how="all").empty:
        return pd.DataFrame()

    tmp = x.copy()
    tmp.index = pd.to_datetime(tmp.index)
    tmp["Year"] = tmp.index.year
    tmp["Month"] = tmp.index.month

    # build MultiIndex columns: strategy-month
    out = {}
    for col in [c for c in tmp.columns if c not in ["Year", "Month"]]:
        p = tmp.pivot(index="Year", columns="Month", values=col)
        out[col] = p

    return out  # dict[strategy]->DataFrame(year x month)


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(x: float) -> str:
    return "NaN" if (x is None or not np.isfinite(x)) else f"{100*x:,.2f}%"


def fmt_num(x: float) -> str:
    return "NaN" if (x is None or not np.isfinite(x)) else f"{x:,.4f}"


# -----------------
# Dashboard
# -----------------
def main():
    # Load data
    close = load_close(CLOSE_CSV)
    if BENCH not in close.columns:
        raise ValueError(f"Benchmark {BENCH} not found in {CLOSE_CSV}")

    # Logs -> ledgers
    df5 = extract_daily_ledger(read_jsonl(LOG_TOP5))
    df10 = extract_daily_ledger(read_jsonl(LOG_TOP10))

    if df5.empty or df10.empty:
        print("No paper logs found yet. Run run_paper_parallel.py at least once.")
        return

    # Align on common dates
    common = df5.index.intersection(df10.index)
    df5 = df5.loc[common]
    df10 = df10.loc[common]

    eq5 = df5["equity_after"].astype(float).rename("top5")
    eq10 = df10["equity_after"].astype(float).rename("top10")

    # Benchmark equity scaled to 1000 starting on first common date
    bench_px = close[BENCH].reindex(common).dropna()
    bench_eq = scaled_equity_from_prices(bench_px, start_value=1000.0).rename("SPY")

    # Regime (risk-on/off) based on MA200 of benchmark
    bench_ma = close[BENCH].rolling(MA_WINDOW).mean().reindex(common)
    regime = (bench_px >= bench_ma).astype(float)  # 1 risk-on, 0 risk-off
    # If MA not ready, regime will be False; treat as NaN
    regime = regime.where(bench_ma.notna(), np.nan).rename("risk_on")

    # ---- Print Summary Table (PM header) ----
    m5 = compute_metrics_from_equity(eq5)
    m10 = compute_metrics_from_equity(eq10)
    mspy = compute_metrics_from_equity(bench_eq)

    summary = pd.DataFrame([m5, m10, mspy], index=["top5", "top10", "SPY"])
    summary = summary[["ann_return", "ann_vol", "sharpe", "max_dd"]]

    # Active metrics vs SPY (approx using daily returns where possible)
    r5 = eq5.pct_change()
    r10 = eq10.pct_change()
    rb = bench_eq.pct_change()
    active5 = (r5 - rb).dropna()
    active10 = (r10 - rb).dropna()

    def active_stats(ar: pd.Series) -> dict:
        if len(ar) < 3:
            return {"ir": np.nan, "active_vol": np.nan, "active_return": np.nan}
        active_return = (1 + ar).prod() ** (252 / len(ar)) - 1
        active_vol = ar.std(ddof=0) * np.sqrt(252)
        ir = active_return / active_vol if active_vol > 0 else np.nan
        return {"active_return": float(active_return), "active_vol": float(active_vol), "ir": float(ir)}

    a5 = active_stats(active5)
    a10 = active_stats(active10)

    print("\n==================== PM DASHBOARD ====================")
    print("Date range:", common.min().date(), "->", common.max().date())
    print("\n=== Performance Summary (annualized, from paper equity) ===")
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\n=== Active vs SPY (annualized approx) ===")
    print("top5  active_return:", fmt_pct(a5["active_return"]), "active_vol:", fmt_pct(a5["active_vol"]), "IR:", fmt_num(a5["ir"]))
    print("top10 active_return:", fmt_pct(a10["active_return"]), "active_vol:", fmt_pct(a10["active_vol"]), "IR:", fmt_num(a10["ir"]))

    # ---- Latest state / holdings ----
    s5 = load_state(STATE_TOP5)
    s10 = load_state(STATE_TOP10)

    def print_state(name: str, s: dict):
        print(f"\n=== {name} Latest State ===")
        if not s:
            print("No state file found.")
            return
        print("cash:", s.get("cash"))
        print("equity:", s.get("equity"))
        sh = s.get("shares", {})
        if sh:
            # Sort by absolute market value using last available prices
            last_px = close.loc[common.max()].astype(float)
            items = []
            for sym, qty in sh.items():
                if sym in last_px.index:
                    items.append((sym, float(qty), float(qty) * float(last_px[sym])))
            items.sort(key=lambda x: abs(x[2]), reverse=True)
            print("holdings (sym, shares, $value):")
            for sym, qty, val in items:
                print(f"  {sym:>5}  {qty:>12.6f}  {val:>12.2f}")
        else:
            print("holdings: (none)")

        if s.get("pending_orders"):
            print("PENDING orders (to execute next run day):")
            for sym, qty in s["pending_orders"].items():
                if abs(float(qty)) > 1e-12:
                    print(f"  {sym:>5}: {float(qty):+.6f} shares")
        else:
            print("pending_orders: none")

    print_state("top5", s5)
    print_state("top10", s10)

    # ---- Plots ----
    # 1) Equity curves
    plt.figure(figsize=(10, 4))
    eq5.plot(label="top5")
    eq10.plot(label="top10")
    bench_eq.reindex(common).plot(label="SPY (scaled)")
    plt.title("Equity Curves (Paper) vs SPY")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Drawdowns
    plt.figure(figsize=(10, 3))
    drawdown(eq5).plot(label="top5")
    drawdown(eq10).plot(label="top10")
    drawdown(bench_eq.reindex(common)).plot(label="SPY")
    plt.title("Drawdown Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Rolling Sharpe (if enough history)
    rs5 = rolling_sharpe(eq5, win=ROLL_WIN)
    rs10 = rolling_sharpe(eq10, win=ROLL_WIN)
    rsb = rolling_sharpe(bench_eq.reindex(common), win=ROLL_WIN)

    plt.figure(figsize=(10, 3))
    if rs5.notna().sum() == 0 and rs10.notna().sum() == 0 and rsb.notna().sum() == 0:
        plt.text(0.5, 0.5, f"Not enough history for {ROLL_WIN}-day rolling Sharpe yet.",
                 ha="center", va="center")
        plt.axis("off")
    else:
        rs5.plot(label="top5")
        rs10.plot(label="top10")
        rsb.plot(label="SPY")
        plt.legend()
    plt.title(f"Rolling Sharpe ({ROLL_WIN} trading days)")
    plt.tight_layout()
    plt.show()

    # 4) Turnover proxy + fees on rebalance days
    def turnover_proxy(df: pd.DataFrame) -> pd.Series:
        x = df[df["is_rebalance"]].copy()
        if x.empty:
            return pd.Series(dtype=float)
        denom = x["equity_before"].replace(0.0, np.nan)
        return (x["traded_notional"] / denom).replace([np.inf, -np.inf], np.nan).dropna()

    tp5 = turnover_proxy(df5).rename("top5")
    tp10 = turnover_proxy(df10).rename("top10")

    plt.figure(figsize=(10, 3))
    if len(tp5) == 0 and len(tp10) == 0:
        plt.text(0.5, 0.5, "No rebalance trades logged yet.", ha="center", va="center")
        plt.axis("off")
    else:
        if len(tp5):
            tp5.plot(marker="o", linewidth=1, label="top5")
        if len(tp10):
            tp10.plot(marker="o", linewidth=1, label="top10")
        plt.legend()
    plt.title("Rebalance Turnover Proxy (traded_notional / equity_before)")
    plt.tight_layout()
    plt.show()

    # 5) Regime overlay plot (risk-on/off)
    plt.figure(figsize=(10, 2.8))
    if regime.notna().sum() == 0:
        plt.text(0.5, 0.5, "MA regime not ready yet (need MA_WINDOW history).", ha="center", va="center")
        plt.axis("off")
    else:
        regime.plot(drawstyle="steps-post")
        plt.ylim(-0.1, 1.1)
        plt.yticks([0, 1], ["risk_off", "risk_on"])
        plt.title(f"Regime (SPY >= MA{MA_WINDOW})")
    plt.tight_layout()
    plt.show()

    # ---- Monthly table (printed) ----
    mr = pd.concat([
        monthly_returns(eq5).rename("top5"),
        monthly_returns(eq10).rename("top10"),
        monthly_returns(bench_eq.reindex(common)).rename("SPY")
    ], axis=1)

    print("\n=== Monthly Returns (last 24 months) ===")
    if mr.dropna(how="all").empty or mr.shape[0] < 2:
        print("Not enough history yet to compute monthly returns (need at least 2 month-end points).")
    else:
        tables = monthly_table(mr, last_n_months=24)  # dict strategy -> year x month table
        for strat, tab in tables.items():
            print(f"\n[{strat}]")
            # show month columns 1..12 with % formatting
            tab2 = tab.copy() * 100.0
            tab2 = tab2.reindex(columns=range(1, 13))
            tab2.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            print(tab2.to_string(float_format=lambda x: f"{x:6.2f}%"))

    print("\n================== END DASHBOARD ==================\n")


if __name__ == "__main__":
    main()