from __future__ import annotations
import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo
import pandas as pd
from thesis.config import (
    DATABENTO_API_KEY,
    DATABENTO_RAW_DIR,
    ensure_dirs,
    require_api_key,
)
# ============================================================
# CONFIG
# ============================================================

OPT_DATASET = "OPRA.PILLAR"
OPT_CBBO_SCHEMA = "cbbo-1m"

EQ_DATASET = "EQUS.SUMMARY"
EQ_OHLCV_SCHEMA = "ohlcv-1d"

TICKERS: list[str] = [
    "NVDA",
    "NVDL",
]

DEFAULT_LOOKBACK_MINUTES = 1
TICKER_LOOKBACK_OVERRIDES: dict[str, int] = {
    "NVDA": 1,
}

# Use either DATES or DATE_START + DATE_END
DATES: list[str] = []
DATE_START = "2025-01-01"
DATE_END = "2025-12-31"

NY = ZoneInfo("America/New_York")
CLOSE_TIME_NY = dt.time(16, 0)
QUOTE_LOOKAHEAD_MINUTES = 1

OUTPUT_DIR = DATABENTO_RAW_DIR
PARQUET_FILENAME = "opra_cbbo_close_snapshots.parquet"
EXCEL_MAX_DATA_ROWS = 1_048_575

# Max parallel CBBO date fetches per ticker
CBBO_MAX_WORKERS = 3

# Retry settings for transient server errors (503, 504)
RETRY_MAX_ATTEMPTS = 5
RETRY_INITIAL_WAIT_SECONDS = 5


# ============================================================
# HELPERS
# ============================================================

def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        item = str(raw).strip().upper()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def resolve_tickers() -> list[str]:
    tickers = dedupe_preserve_order(TICKERS)
    if not tickers:
        raise SystemExit("Populate TICKERS at the top of the script.")
    return tickers


def build_dates(explicit: list[str], start: str | None, end: str | None) -> list[dt.date]:
    if explicit:
        out = [dt.date.fromisoformat(s.strip()) for s in explicit if s.strip()]
    elif start and end:
        out = [d.date() for d in pd.bdate_range(start, end)]
    else:
        raise SystemExit("Provide either DATES or DATE_START + DATE_END.")
    return sorted(dict.fromkeys(out))


def get_lookback(ticker: str) -> int:
    return TICKER_LOOKBACK_OVERRIDES.get(ticker, DEFAULT_LOOKBACK_MINUTES)


def utc_day_bounds(d: dt.date) -> tuple[dt.datetime, dt.datetime]:
    start = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc)
    return start, start + dt.timedelta(days=1)


def close_time_utc(d: dt.date) -> dt.datetime:
    return dt.datetime.combine(d, CLOSE_TIME_NY, tzinfo=NY).astimezone(dt.timezone.utc)


def cbbo_window_utc(d: dt.date, lookback_min: int) -> tuple[dt.datetime, dt.datetime, dt.datetime]:
    close_utc = close_time_utc(d)
    return (
        close_utc - dt.timedelta(minutes=lookback_min),
        close_utc,
        close_utc + dt.timedelta(minutes=QUOTE_LOOKAHEAD_MINUTES),
    )


def normalize_df(df: pd.DataFrame, context: str) -> pd.DataFrame:
    index_name = df.index.name
    df = df.reset_index()

    if "index" in df.columns:
        if "ts_recv" not in df.columns:
            df = df.rename(columns={"index": "ts_recv"})
        else:
            df = df.drop(columns=["index"])

    if "ts_recv" not in df.columns:
        for candidate in [index_name, "ts_event", "ts_record"]:
            if candidate and candidate in df.columns:
                df["ts_recv"] = df[candidate]
                break
        else:
            raise KeyError(f"[{context}] No usable timestamp column. Columns: {df.columns.tolist()}")

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    return df


def to_dt_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def safe_sheet_name(name: str) -> str:
    invalid = set('[]:*?/\\')
    cleaned = "".join(ch if ch not in invalid else "_" for ch in str(name))
    return (cleaned.strip() or "Sheet1")[:31]


def clean_for_save(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "expiration" in df.columns:
        df["expiration"] = df["expiration"].apply(
            lambda x: x.isoformat() if isinstance(x, dt.date) else str(x) if pd.notna(x) else None
        )
    return df


def sort_output(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["trade_date", "expiration", "symbol"] if c in df.columns]
    if not cols:
        return df
    return df.sort_values(cols, kind="stable").reset_index(drop=True)


# ============================================================
# OCC SYMBOL PARSER
# ============================================================

def parse_occ_symbol(symbol: str) -> tuple[dt.date | None, str | None, float | None]:
    s = str(symbol)
    if len(s) < 21:
        return None, None, None

    try:
        exp_str = s[6:12]
        instrument_class = s[12]
        strike_raw = s[13:21]

        expiration = dt.date(
            2000 + int(exp_str[:2]),
            int(exp_str[2:4]),
            int(exp_str[4:6]),
        )
        strike_price = int(strike_raw) / 1000.0

        if instrument_class not in ("C", "P"):
            instrument_class = None

        return expiration, instrument_class, strike_price
    except (TypeError, ValueError, IndexError):
        return None, None, None


def add_occ_columns(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    parsed = df[symbol_col].astype(str).apply(parse_occ_symbol)
    out = df.copy()
    out["expiration"] = parsed.str[0]
    out["instrument_class"] = parsed.str[1]
    out["strike_price"] = parsed.str[2]
    return out


# ============================================================
# DATA FETCH
# ============================================================

OUTPUT_COLUMNS = [
    "underlying",
    "trade_date",
    "symbol",
    "expiration",
    "instrument_class",
    "strike_price",
    "ts_recv",
    "ts_event",
    "close_time_utc",
    "staleness_seconds",
    "bid_px_00",
    "ask_px_00",
    "bid_sz_00",
    "ask_sz_00",
    "mid_px",
    "spread",
    "price",
    "size",
    "underlying_price_unadj",
]


def fetch_underlying_prices_for_ticker(
    client: Any,
    ticker: str,
    dates: list[dt.date],
) -> dict[dt.date, float]:
    range_start = utc_day_bounds(dates[0])[0]
    range_end = utc_day_bounds(dates[-1])[1]

    store = client.timeseries.get_range(
        dataset=EQ_DATASET,
        schema=EQ_OHLCV_SCHEMA,
        symbols=[ticker],
        stype_in="raw_symbol",
        start=range_start,
        end=range_end,
    )

    df = store.to_df(price_type="float")
    df = normalize_df(df, f"{ticker} ohlcv-1d")

    if df.empty:
        return {}

    if "close" not in df.columns:
        raise KeyError(f"[{ticker} ohlcv-1d] Missing 'close'. Columns: {df.columns.tolist()}")

    df["ts_recv"] = to_dt_utc(df["ts_recv"])
    df["trade_date"] = df["ts_recv"].dt.date
    df = df.sort_values("ts_recv").drop_duplicates(subset=["trade_date"], keep="last")

    out: dict[dt.date, float] = {}
    for _, row in df.iterrows():
        out[row["trade_date"]] = float(row["close"])

    return out


def fetch_cbbo_for_ticker_date(
    client: Any,
    ticker: str,
    d: dt.date,
    lookback_min: int,
) -> pd.DataFrame:
    win_start, close_utc, win_end = cbbo_window_utc(d, lookback_min)
    parent = f"{ticker}.OPT"

    last_exc = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            store = client.timeseries.get_range(
                dataset=OPT_DATASET,
                schema=OPT_CBBO_SCHEMA,
                symbols=[parent],
                stype_in="parent",
                start=win_start,
                end=win_end,
            )
            break  # success
        except Exception as e:
            err_str = str(e)
            is_retryable = "504" in err_str or "503" in err_str
            if is_retryable and attempt < RETRY_MAX_ATTEMPTS:
                wait = RETRY_INITIAL_WAIT_SECONDS * attempt
                print(f"[{ticker}]   {d} attempt {attempt} failed (504/503), "
                      f"retrying in {wait}s...")
                time.sleep(wait)
                last_exc = e
            else:
                raise
    else:
        raise last_exc  # type: ignore[misc]

    df = store.to_df(price_type="float")
    df = normalize_df(df, f"{ticker} cbbo {d}")

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if "symbol" not in df.columns:
        if "raw_symbol" in df.columns:
            df["symbol"] = df["raw_symbol"]
        else:
            raise KeyError(f"[{ticker} cbbo {d}] Missing 'symbol' and 'raw_symbol'.")

    df["ts_recv"] = to_dt_utc(df["ts_recv"])
    df = df[df["ts_recv"] <= close_utc].copy()

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    dedupe_col = "instrument_id" if "instrument_id" in df.columns else "symbol"
    df = df.sort_values("ts_recv").drop_duplicates(subset=[dedupe_col], keep="last").copy()

    if "ts_event" in df.columns:
        df["ts_event"] = to_dt_utc(df["ts_event"]).dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        df["ts_event"] = pd.NaT

    for col in ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "price", "size"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["mid_px"] = (df["bid_px_00"] + df["ask_px_00"]) / 2.0
    df["spread"] = df["ask_px_00"] - df["bid_px_00"]
    df["staleness_seconds"] = (close_utc - df["ts_recv"]).dt.total_seconds()
    df["ts_recv"] = df["ts_recv"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["close_time_utc"] = close_utc.replace(tzinfo=None)

    df = add_occ_columns(df, "symbol")
    df["underlying"] = ticker
    df["trade_date"] = d.isoformat()
    df["underlying_price_unadj"] = pd.NA

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[OUTPUT_COLUMNS].copy()

# ============================================================
# EXCEL EXPORT
# ============================================================

def write_ticker_excel(df: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
    path = output_dir / f"{ticker}_opra_cbbo_snapshots.xlsx"
    n_rows = len(df)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if n_rows <= EXCEL_MAX_DATA_ROWS:
            df.to_excel(writer, sheet_name=safe_sheet_name(ticker), index=False)
        else:
            n_chunks = (n_rows + EXCEL_MAX_DATA_ROWS - 1) // EXCEL_MAX_DATA_ROWS
            for i in range(n_chunks):
                start_row = i * EXCEL_MAX_DATA_ROWS
                end_row = min((i + 1) * EXCEL_MAX_DATA_ROWS, n_rows)
                chunk = df.iloc[start_row:end_row]
                sheet_name = safe_sheet_name(f"{ticker}_{i + 1}")
                chunk.to_excel(writer, sheet_name=sheet_name, index=False)

    return path


# ============================================================
# PER-TICKER PIPELINE
# ============================================================

def process_ticker(
    client: Any,
    ticker: str,
    dates: list[dt.date],
    parquet_dir: Path,
) -> tuple[str | None, list[str]]:
    """
    Download CBBO + underlying data for one ticker, save parquet.
    CBBO dates are fetched concurrently using CBBO_MAX_WORKERS threads.
    Returns (ticker_name_or_None, list_of_error_strings).
    """
    lookback = get_lookback(ticker)
    print(f"\n{'=' * 70}")
    print(f"{ticker} | lookback={lookback} min | {len(dates)} dates | "
          f"CBBO workers={CBBO_MAX_WORKERS}")
    print(f"{'=' * 70}")

    errors: list[str] = []

    # --- underlying prices (non-fatal if it fails) ---
    print(f"[{ticker}] Fetching underlying prices...")
    try:
        underlying_prices = fetch_underlying_prices_for_ticker(client, ticker, dates)
        print(f"[{ticker}] Underlying dates returned: {len(underlying_prices)}")
    except Exception as e:
        err_msg = f"underlying: {type(e).__name__}: {e}"
        print(f"[{ticker}] *** ERROR fetching underlying prices: {type(e).__name__}: {e}")
        errors.append(err_msg)
        underlying_prices = {}

    # --- CBBO: concurrent date fetches ---
    ticker_frames: list[pd.DataFrame] = []
    total_dates = len(dates)

    workers = min(CBBO_MAX_WORKERS, total_dates)
    print(f"[{ticker}] Fetching CBBO for {total_dates} dates "
          f"({workers} concurrent workers)...")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_cbbo_for_ticker_date, client, ticker, d, lookback): d
            for d in dates
        }
        completed = 0
        for fut in as_completed(futures):
            d = futures[fut]
            completed += 1
            try:
                day_df = fut.result()
            except Exception as e:
                errors.append(f"CBBO {d}: {type(e).__name__}: {e}")
                continue

            if day_df.empty:
                continue

            day_df["underlying_price_unadj"] = underlying_prices.get(d)
            ticker_frames.append(day_df)

            # Progress every 10 dates or at the end
            if completed % 10 == 0 or completed == total_dates:
                cbbo_errors = sum(1 for e in errors if e.startswith("CBBO"))
                print(f"[{ticker}]   progress: {completed}/{total_dates} "
                      f"({len(ticker_frames)} with data, {cbbo_errors} errors)")

    cbbo_errors = [e for e in errors if e.startswith("CBBO")]
    if cbbo_errors:
        print(f"[{ticker}] {len(cbbo_errors)} date(s) had CBBO errors:")
        for err in cbbo_errors:
            print(f"[{ticker}]   *** {err}")

    if not ticker_frames:
        print(f"[{ticker}] No CBBO data assembled.")
        return None, errors

    ticker_df = pd.concat(ticker_frames, ignore_index=True)
    ticker_df = sort_output(clean_for_save(ticker_df))

    parquet_path = parquet_dir / f"{ticker}.parquet"
    ticker_df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"[{ticker}] Saved parquet: {parquet_path} ({len(ticker_df):,} rows)")

    return ticker, errors


# ============================================================
# MAIN
# ============================================================

def main(
    *,
    api_key: str | None = None,
    output_dir: str | Path | None = None,
) -> None:
    import databento as db

    tickers = resolve_tickers()
    dates = build_dates(DATES, DATE_START, DATE_END)
    api_key = require_api_key("DATABENTO_API_KEY", api_key or DATABENTO_API_KEY)
    output_root = Path(output_dir) if output_dir is not None else OUTPUT_DIR

    print(f"Tickers: {tickers}")
    print(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    client = db.Historical(key=api_key)

    ensure_dirs()
    output_root.mkdir(parents=True, exist_ok=True)
    parquet_dir = output_root / "parquet_parts"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    excel_dir = output_root / "excel"
    excel_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PHASE 1: Download each ticker sequentially → parquet only
    # ------------------------------------------------------------------
    completed_tickers: list[str] = []
    all_errors: dict[str, list[str]] = {}

    for ticker in tickers:
        result, errors = process_ticker(client, ticker, dates, parquet_dir)
        if errors:
            all_errors[ticker] = errors
        if result:
            completed_tickers.append(result)

    # ------------------------------------------------------------------
    # PHASE 2: Build master parquet from parts
    # ------------------------------------------------------------------
    if completed_tickers:
        parts = [
            pd.read_parquet(parquet_dir / f"{t}.parquet")
            for t in completed_tickers
            if (parquet_dir / f"{t}.parquet").exists()
        ]
        if parts:
            master = pd.concat(parts, ignore_index=True)
            master = sort_output(clean_for_save(master))
            master_path = output_root / PARQUET_FILENAME
            master.to_parquet(master_path, index=False, engine="pyarrow")
            print(f"\nSaved master parquet: {master_path} ({len(master):,} rows)")
    else:
        print("\nNo tickers completed.")
        return

    # ------------------------------------------------------------------
    # PHASE 3: Generate Excel files from saved parquets
    # ------------------------------------------------------------------
    print(f"\nGenerating Excel files for {len(completed_tickers)} ticker(s)...")
    for t in completed_tickers:
        pq_path = parquet_dir / f"{t}.parquet"
        if not pq_path.exists():
            continue
        try:
            df = pd.read_parquet(pq_path)
            excel_path = write_ticker_excel(df, t, excel_dir)
            print(f"[{t}] Saved Excel: {excel_path} ({len(df):,} rows)")
        except Exception as e:
            print(f"[{t}] *** ERROR writing Excel: {type(e).__name__}: {e}")
            all_errors.setdefault(t, []).append(f"Excel write: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # PHASE 4: Error summary
    # ------------------------------------------------------------------
    if all_errors:
        print(f"\n{'=' * 70}")
        print("ERROR SUMMARY")
        print(f"{'=' * 70}")
        for ticker, errs in all_errors.items():
            print(f"\n  [{ticker}] — {len(errs)} error(s):")
            for err in errs:
                print(f"    • {err}")
    else:
        print("\nNo errors encountered.")

    print(f"\nOutput written to: {output_root}")


if __name__ == "__main__":
    main()
