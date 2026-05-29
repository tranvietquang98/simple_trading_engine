from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import yfinance as yf


def fetch_ohlc(
    tickers: list[str],
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (close_df, open_df) indexed by Date with columns = tickers.
    """
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        group_by="column",
        auto_adjust=auto_adjust,
        threads=True,
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy() if "Close" in df.columns.get_level_values(0) else pd.DataFrame()
        open_px = df["Open"].copy() if "Open" in df.columns.get_level_values(0) else pd.DataFrame()
    else:
        # single ticker
        if "Close" not in df.columns or "Open" not in df.columns:
            raise ValueError("Missing 'Close' or 'Open' in yfinance response.")
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
        open_px = df[["Open"]].rename(columns={"Open": tickers[0]})

    for df_target in [close, open_px]:
        if not df_target.empty:
            df_target.index = pd.to_datetime(df_target.index)
            df_target.sort_index(inplace=True)
            for t in tickers:
                if t not in df_target.columns:
                    df_target[t] = np.nan

    return close[tickers], open_px[tickers]


def load_existing(csv_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def merge_and_clean(
    old: pd.DataFrame | None,
    new: pd.DataFrame,
    *,
    ffill_limit: int = 2,
    max_nan_frac_per_row: float = 0.2,
) -> pd.DataFrame:
    if old is None:
        out = new.copy()
    else:
        cols = sorted(set(old.columns).union(new.columns))
        out = pd.concat([old.reindex(columns=cols), new.reindex(columns=cols)], axis=0)
    out = out[~out.index.duplicated(keep="last")].sort_index()

    if ffill_limit is not None and ffill_limit > 0:
        out = out.ffill(limit=ffill_limit)

    nan_frac = out.isna().mean(axis=1)
    out = out.loc[nan_frac <= max_nan_frac_per_row]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=str, required=True, help="Comma-separated, e.g. SPY,QQQ,TLT")
    ap.add_argument("--csv_close", type=str, default="data/close_prices.csv")
    ap.add_argument("--csv_open", type=str, default="data/open_prices.csv")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--no_auto_adjust", action="store_true")
    ap.add_argument("--ffill_limit", type=int, default=2)
    ap.add_argument("--max_nan_frac", type=float, default=0.2)
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise ValueError("No tickers provided.")

    old_close = load_existing(args.csv_close)
    old_open = load_existing(args.csv_open)

    end = args.end or (datetime.utcnow().date() + timedelta(days=1)).isoformat()
    
    if args.start is None:
        if old_close is not None and len(old_close) > 0:
            start = (old_close.index[-1].date() - timedelta(days=5)).isoformat()
        else:
            start = (datetime.utcnow().date() - timedelta(days=3650)).isoformat()
    else:
        start = args.start

    auto_adjust = not args.no_auto_adjust

    new_close, new_open = fetch_ohlc(tickers, start=start, end=end, auto_adjust=auto_adjust)
    
    out_close = merge_and_clean(old_close, new_close, ffill_limit=args.ffill_limit, max_nan_frac_per_row=args.max_nan_frac)
    out_open = merge_and_clean(old_open, new_open, ffill_limit=args.ffill_limit, max_nan_frac_per_row=args.max_nan_frac)

    out_close.reset_index().rename(columns={"index": "Date"}).to_csv(args.csv_close, index=False)
    out_open.reset_index().rename(columns={"index": "Date"}).to_csv(args.csv_open, index=False)
    
    print(f"Wrote {args.csv_close} & {args.csv_open}: {len(out_close)} rows. Range: {out_close.index.min().date()} to {out_close.index.max().date()}")


if __name__ == "__main__":
    main()