from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLOSE_CSV = "data/close_prices.csv"
BENCH = "SPY"
LOG_TOP5 = "logs/paper_log_top5.jsonl"
LOG_TOP10 = "logs/paper_log_top10.jsonl"


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_daily_ledger(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        if r.get("type") == "mtm":
            rec = r.copy()
        else:
            rec = {}
            rec["asof"] = r.get("asof")
            extra = r.get("extra", {}) or {}
            for k, v in extra.items():
                rec[k] = v

            rec["is_rebalance"] = rec.get("is_rebalance", True)
            rec["traded_notional"] = rec.get("traded_notional", 0.0)
            rec["fees"] = rec.get("fees", 0.0)

        if "asof" not in rec or rec["asof"] is None:
            continue
        out.append(rec)

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

    if df["equity_after"].isna().any():
        df["equity_after"] = df["equity_after"].fillna(df["equity_before"])

    return df


def load_benchmark_equity(csv_path: str, bench: str, dates: pd.DatetimeIndex) -> pd.Series:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    if bench not in df.columns:
        raise ValueError(f"{bench} not found in {csv_path}")
    px = df[bench].reindex(dates).dropna()
    eq = px / px.iloc[0] * 1000.0  
    return eq.rename(bench)


def compute_metrics_from_equity(equity: pd.Series, ann_factor: int = 252) -> dict:
    equity = equity.dropna()
    if len(equity) < 3:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}

    ret = equity.pct_change().dropna()
    ann_return = (1.0 + ret).prod() ** (ann_factor / len(ret)) - 1.0
    ann_vol = ret.std(ddof=0) * np.sqrt(ann_factor)
    
    # Division by zero safety
    sharpe = ann_return / ann_vol if ann_vol > 1e-8 else 0.0

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
    }


def plot_equity(top5_eq: pd.Series, top10_eq: pd.Series, bench_eq: pd.Series):
    plt.figure(figsize=(10, 4))
    top5_eq.plot(label="top5")
    top10_eq.plot(label="top10")
    bench_eq.plot(label="SPY (scaled)")
    plt.title("Paper Equity Curves (with SPY benchmark)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_drawdown(eq: pd.Series, label: str):
    peak = eq.cummax()
    dd = eq / peak - 1.0
    plt.figure(figsize=(10, 3))
    dd.plot()
    plt.title(f"Drawdown: {label}")
    plt.tight_layout()
    plt.show()


def plot_turnover_proxy(df5: pd.DataFrame, df10: pd.DataFrame):
    def proxy(df: pd.DataFrame) -> pd.Series:
        x = df[df["is_rebalance"]].copy()
        denom = x["equity_before"].replace(0.0, np.nan)
        p = (x["traded_notional"] / denom).replace([np.inf, -np.inf], np.nan).dropna()
        return p

    p5 = proxy(df5)
    p10 = proxy(df10)

    plt.figure(figsize=(10, 3))
    if len(p5):
        p5.plot(label="top5", marker="o", linewidth=1)
    if len(p10):
        p10.plot(label="top10", marker="o", linewidth=1)
    plt.title("Rebalance Turnover Proxy (traded_notional / equity_before)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def monthly_return_table(eq: pd.Series) -> pd.Series:
    mret = eq.resample("ME").last().pct_change()
    return mret


def main():
    rows5 = read_jsonl(LOG_TOP5)
    rows10 = read_jsonl(LOG_TOP10)

    df5 = extract_daily_ledger(rows5)
    df10 = extract_daily_ledger(rows10)

    common = df5.index.intersection(df10.index)
    df5c = df5.loc[common]
    df10c = df10.loc[common]

    eq5 = df5c["equity_after"].astype(float).rename("top5")
    eq10 = df10c["equity_after"].astype(float).rename("top10")
    bench_eq = load_benchmark_equity(CLOSE_CSV, BENCH, common)

    plot_equity(eq5, eq10, bench_eq)
    plot_drawdown(eq5, "top5")
    plot_drawdown(eq10, "top10")
    plot_turnover_proxy(df5c, df10c)

    m_top5 = compute_metrics_from_equity(eq5)
    m_top10 = compute_metrics_from_equity(eq10)
    m_spy = compute_metrics_from_equity(bench_eq)

    summary = pd.DataFrame([m_top5, m_top10, m_spy], index=["top5", "top10", "SPY"])
    summary = summary[["ann_return", "ann_vol", "sharpe", "max_dd"]]

    print("\n=== Paper Performance Summary (from realized equity) ===")
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))

    m5 = monthly_return_table(eq5).rename("top5")
    m10 = monthly_return_table(eq10).rename("top10")
    mb = monthly_return_table(bench_eq).rename("SPY")

    mtbl = pd.concat([m5, m10, mb], axis=1)

    print("\n=== Monthly Returns (last 12 months) ===")
    if mtbl.dropna(how="all").empty:
        print("Not enough history yet to compute monthly returns (need at least 2 month-end points).")
    else:
        print((mtbl.tail(12) * 100).to_string(float_format=lambda x: f"{x:,.2f}%"))


if __name__ == "__main__":
    main()
