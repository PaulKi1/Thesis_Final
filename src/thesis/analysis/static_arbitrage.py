from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from thesis.config import DEAMERICANIZED_DIR, FENGLER_DIR, ensure_dirs


# ---------------------------
# User settings
# ---------------------------

WORKBOOK_PATH: Optional[Path] = None
SHEET_NAME: Optional[str] = None

PRICE_TOLERANCE = 1e-8
CALENDAR_TOLERANCE = 1e-8
BLACK_INVERSION_TOLERANCE = 1e-8
TOTAL_VARIANCE_UPPER = 25.0

# ---------------------------
# Workbook loading
# ---------------------------

def find_default_workbooks(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(input_dir.glob(pattern))

def _pick_sheet_name(workbook_path: str, preferred: Optional[str]) -> str:
    xls = pd.ExcelFile(workbook_path)
    if preferred is not None:
        if preferred not in xls.sheet_names:
            raise ValueError(f"Sheet {preferred!r} not found. Available sheets: {xls.sheet_names}")
        return preferred
    if "quote_level_output" in xls.sheet_names:
        return "quote_level_output"
    if "deamericanized" in xls.sheet_names:
        return "deamericanized"
    return xls.sheet_names[0]



def _load_workbook(workbook_path: str, sheet_name: Optional[str]) -> Tuple[pd.DataFrame, str]:
    chosen_sheet = _pick_sheet_name(workbook_path, sheet_name)
    header = pd.read_excel(workbook_path, sheet_name=chosen_sheet, nrows=0)
    columns = list(header.columns)

    candidate_columns = [
        "trade_date",
        "expiration",
        "instrument_class",
        "strike_price",
        "mid_px_eu",
        "discount_factor",
        "forward_price",
        "Moneyness",
        "tau",
        "slice_id",
        "repaired_call_eu",
        "repaired_total_variance",
        "support_class",
        "inside_global_core_band",
    ]
    usecols = [c for c in candidate_columns if c in columns]
    if not usecols:
        raise ValueError(f"None of the required columns were found in sheet {chosen_sheet!r}.")

    df = pd.read_excel(workbook_path, sheet_name=chosen_sheet, usecols=usecols)
    return df, chosen_sheet


# ---------------------------
# Input normalisation
# ---------------------------


def _normalise_input(df: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "expiration", "strike_price", "discount_factor", "forward_price", "tau"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Workbook is missing required columns: {missing}")

    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce")

    numeric_cols = [
        "strike_price",
        "discount_factor",
        "forward_price",
        "tau",
        "Moneyness",
        "mid_px_eu",
        "repaired_call_eu",
        "repaired_total_variance",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["trade_date", "expiration", "strike_price", "discount_factor", "forward_price", "tau"])
    out = out.loc[(out["discount_factor"] > 0) & (out["forward_price"] > 0) & (out["tau"] > 0)].copy()

    if "Moneyness" not in out.columns or out["Moneyness"].isna().all():
        out["Moneyness"] = out["strike_price"] / out["forward_price"]
    else:
        missing_kappa = out["Moneyness"].isna()
        out.loc[missing_kappa, "Moneyness"] = out.loc[missing_kappa, "strike_price"] / out.loc[missing_kappa, "forward_price"]

    if "slice_id" not in out.columns:
        out["slice_id"] = (
            out["trade_date"].dt.strftime("%Y-%m-%d")
            + "__"
            + out["expiration"].dt.strftime("%Y-%m-%d")
        )
    else:
        out["slice_id"] = out["slice_id"].fillna(
            out["trade_date"].dt.strftime("%Y-%m-%d")
            + "__"
            + out["expiration"].dt.strftime("%Y-%m-%d")
        )

    out = out.drop_duplicates().sort_values(["trade_date", "expiration", "strike_price"]).reset_index(drop=True)
    return out


# ---------------------------
# Mode detection and call construction
# ---------------------------


def _is_repaired_quote_level_workbook(df: pd.DataFrame) -> bool:
    return "repaired_call_eu" in df.columns and df["repaired_call_eu"].notna().any()



def _build_effective_calls_from_raw(day: pd.DataFrame) -> pd.DataFrame:
    required = {"instrument_class", "mid_px_eu", "Moneyness"}
    missing = sorted(required.difference(day.columns))
    if missing:
        raise ValueError(f"Raw workbook is missing required columns for call construction: {missing}")

    work = day.copy()
    work["instrument_class"] = work["instrument_class"].astype(str).str.upper()

    rows: List[Dict[str, object]] = []
    grouped = work.groupby(["slice_id", "strike_price"], sort=True, dropna=False)
    for (_, strike), grp in grouped:
        grp = grp.sort_values("instrument_class")
        first = grp.iloc[0]

        df = float(first["discount_factor"])
        forward = float(first["forward_price"])
        tau = float(first["tau"])
        kappa = float(first["Moneyness"])

        call_values = grp.loc[grp["instrument_class"] == "C", "mid_px_eu"].dropna()
        put_values = grp.loc[grp["instrument_class"] == "P", "mid_px_eu"].dropna()

        raw_call = float(call_values.iloc[0]) if not call_values.empty else np.nan
        raw_put = float(put_values.iloc[0]) if not put_values.empty else np.nan
        put_converted_call = raw_put + df * (forward - float(strike)) if np.isfinite(raw_put) else np.nan

        if kappa < 1.0 and np.isfinite(raw_put):
            effective_call = put_converted_call
        elif kappa >= 1.0 and np.isfinite(raw_call):
            effective_call = raw_call
        elif np.isfinite(raw_call):
            effective_call = raw_call
        elif np.isfinite(raw_put):
            effective_call = put_converted_call
        else:
            continue

        rows.append(
            {
                "trade_date": first["trade_date"],
                "expiration": first["expiration"],
                "slice_id": first["slice_id"],
                "tau": tau,
                "strike_price": float(strike),
                "kappa": kappa,
                "discount_factor": df,
                "forward_price": forward,
                "effective_call": float(effective_call),
                "total_variance": np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["expiration", "strike_price"]).reset_index(drop=True)



def _build_effective_calls_from_repaired(day: pd.DataFrame) -> pd.DataFrame:
    work = day.copy()

    if "support_class" in work.columns:
        work = work.loc[work["support_class"].astype(str) != "outside_core_band"].copy()
    elif "inside_global_core_band" in work.columns:
        work = work.loc[work["inside_global_core_band"].fillna(False)].copy()

    work = work.loc[work["repaired_call_eu"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    keep = [
        "trade_date",
        "expiration",
        "slice_id",
        "tau",
        "strike_price",
        "Moneyness",
        "discount_factor",
        "forward_price",
        "repaired_call_eu",
    ]
    if "repaired_total_variance" in work.columns:
        keep.append("repaired_total_variance")

    work = work[keep].drop_duplicates(subset=["slice_id", "strike_price"])
    out = work.rename(
        columns={
            "Moneyness": "kappa",
            "repaired_call_eu": "effective_call",
            "repaired_total_variance": "total_variance",
        }
    )
    if "total_variance" not in out.columns:
        out["total_variance"] = np.nan
    return out.sort_values(["expiration", "strike_price"]).reset_index(drop=True)



def _build_effective_calls(day: pd.DataFrame, repaired_mode: bool) -> pd.DataFrame:
    if repaired_mode:
        return _build_effective_calls_from_repaired(day)
    return _build_effective_calls_from_raw(day)


# ---------------------------
# Low-level Black helpers
# ---------------------------


def _black_forward_call_from_total_variance(
    strike: float,
    forward: float,
    discount_factor: float,
    total_variance: float,
) -> float:
    intrinsic = discount_factor * max(forward - strike, 0.0)
    if total_variance <= 0.0:
        return intrinsic
    sqrt_w = math.sqrt(total_variance)
    log_fk = math.log(forward / strike)
    d1 = log_fk / sqrt_w + 0.5 * sqrt_w
    d2 = d1 - sqrt_w
    return discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))



def _implied_total_variance_from_call_price(
    price: float,
    strike: float,
    forward: float,
    discount_factor: float,
    tol: float,
    upper: float,
) -> float:
    lower = max(discount_factor * (forward - strike), 0.0)
    upper_price = discount_factor * forward

    if price <= lower + tol:
        return 0.0
    if price >= upper_price - tol:
        return upper

    def objective(w: float) -> float:
        return _black_forward_call_from_total_variance(strike, forward, discount_factor, w) - price

    lo = 1e-16
    hi = max(1.0, upper)
    flo = objective(lo)
    fhi = objective(hi)
    expansions = 0
    while flo * fhi > 0 and expansions < 10:
        hi *= 2.0
        fhi = objective(hi)
        expansions += 1
    if flo * fhi > 0:
        raise RuntimeError("Unable to bracket implied total variance.")
    return float(brentq(objective, lo, hi, xtol=tol, rtol=tol, maxiter=200))


# ---------------------------
# Static arbitrage counts
# ---------------------------


def _attach_bounds_and_total_variance(calls: pd.DataFrame) -> pd.DataFrame:
    out = calls.copy()
    out["lower_bound"] = np.maximum(out["discount_factor"] * (out["forward_price"] - out["strike_price"]), 0.0)
    out["upper_bound"] = out["discount_factor"] * out["forward_price"]

    cleaned: List[float] = []
    total_variances: List[float] = []

    for row in out.itertuples(index=False):
        price = float(row.effective_call)
        lower = float(row.lower_bound)
        upper = float(row.upper_bound)
        strike = float(row.strike_price)
        forward = float(row.forward_price)
        discount_factor = float(row.discount_factor)
        existing_tv = getattr(row, "total_variance", np.nan)

        cleaned_price = price
        if price < lower:
            cleaned_price = lower
        elif price > upper:
            cleaned_price = upper
        cleaned.append(cleaned_price)

        if np.isfinite(existing_tv):
            total_variances.append(float(existing_tv))
        else:
            try:
                total_variances.append(
                    _implied_total_variance_from_call_price(
                        price=cleaned_price,
                        strike=strike,
                        forward=forward,
                        discount_factor=discount_factor,
                        tol=BLACK_INVERSION_TOLERANCE,
                        upper=TOTAL_VARIANCE_UPPER,
                    )
                )
            except Exception:
                total_variances.append(np.nan)

    out["effective_call_clean"] = cleaned
    out["total_variance"] = total_variances
    return out



def _count_strike_side_arbitrage(calls: pd.DataFrame) -> Dict[str, int]:
    counts = {
        "lower_bound": 0,
        "upper_bound": 0,
        "vertical_monotonicity": 0,
        "vertical_slope_bound": 0,
        "butterfly": 0,
    }

    for _, grp in calls.groupby("slice_id", sort=True):
        grp = grp.sort_values("strike_price").drop_duplicates(subset=["strike_price"])
        if grp.empty:
            continue

        strikes = grp["strike_price"].to_numpy(dtype=float)
        prices = grp["effective_call"].to_numpy(dtype=float)
        lower = grp["lower_bound"].to_numpy(dtype=float)
        upper = grp["upper_bound"].to_numpy(dtype=float)
        discount_factor = float(grp["discount_factor"].iloc[0])

        counts["lower_bound"] += int(np.sum(prices < lower - PRICE_TOLERANCE))
        counts["upper_bound"] += int(np.sum(prices > upper + PRICE_TOLERANCE))

        if len(grp) >= 2:
            strike_diff = np.diff(strikes)
            valid = strike_diff > 0
            if np.any(valid):
                slopes = np.diff(prices)[valid] / strike_diff[valid]
                counts["vertical_monotonicity"] += int(np.sum(slopes > PRICE_TOLERANCE))
                counts["vertical_slope_bound"] += int(np.sum(slopes < -discount_factor - PRICE_TOLERANCE))

        if len(grp) >= 3:
            for i in range(1, len(grp) - 1):
                k1, k2, k3 = strikes[i - 1], strikes[i], strikes[i + 1]
                c1, c2, c3 = prices[i - 1], prices[i], prices[i + 1]
                denom = k3 - k1
                if denom <= 0:
                    continue
                weight = (k3 - k2) / denom
                butterfly_value = weight * c1 + (1.0 - weight) * c3 - c2
                if butterfly_value < -PRICE_TOLERANCE:
                    counts["butterfly"] += 1

    return counts



def _count_calendar_arbitrage(calls: pd.DataFrame) -> int:
    calendar_count = 0

    slices = []
    for _, grp in calls.groupby("slice_id", sort=True):
        grp = grp.sort_values("kappa").drop_duplicates(subset=["kappa"])
        grp = grp.loc[np.isfinite(grp["total_variance"])].copy()
        if len(grp) < 2:
            continue
        slices.append(grp)

    slices.sort(key=lambda g: float(g["tau"].iloc[0]))
    if len(slices) < 2:
        return 0

    for short_grp, long_grp in zip(slices[:-1], slices[1:]):
        short_kappa = short_grp["kappa"].to_numpy(dtype=float)
        long_kappa = long_grp["kappa"].to_numpy(dtype=float)
        short_tv = short_grp["total_variance"].to_numpy(dtype=float)
        long_tv = long_grp["total_variance"].to_numpy(dtype=float)

        overlap_low = max(np.min(short_kappa), np.min(long_kappa))
        overlap_high = min(np.max(short_kappa), np.max(long_kappa))
        if overlap_high <= overlap_low:
            continue

        grid = np.union1d(
            short_kappa[(short_kappa >= overlap_low) & (short_kappa <= overlap_high)],
            long_kappa[(long_kappa >= overlap_low) & (long_kappa <= overlap_high)],
        )
        if grid.size == 0:
            continue

        short_interp = np.interp(grid, short_kappa, short_tv)
        long_interp = np.interp(grid, long_kappa, long_tv)
        calendar_count += int(np.sum(long_interp < short_interp - CALENDAR_TOLERANCE))

    return calendar_count



def count_static_arbitrage(calls: pd.DataFrame) -> Dict[str, int]:
    calls = _attach_bounds_and_total_variance(calls)

    counts = _count_strike_side_arbitrage(calls)
    counts["european_bounds_total"] = counts["lower_bound"] + counts["upper_bound"]
    counts["vertical_total"] = counts["vertical_monotonicity"] + counts["vertical_slope_bound"]
    counts["calendar_total_variance"] = _count_calendar_arbitrage(calls)
    counts["total_violations"] = (
        counts["lower_bound"]
        + counts["upper_bound"]
        + counts["vertical_monotonicity"]
        + counts["vertical_slope_bound"]
        + counts["butterfly"]
        + counts["calendar_total_variance"]
    )
    return counts


# ---------------------------
# Public runner
# ---------------------------


def run_static_arb_report(workbook_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    workbook_path = str(Path(workbook_path))
    df, chosen_sheet = _load_workbook(workbook_path, sheet_name)
    df = _normalise_input(df)
    repaired_mode = _is_repaired_quote_level_workbook(df)

    summary_rows: List[Dict[str, object]] = []
    trade_dates = sorted(df["trade_date"].dropna().unique())

    for trade_date in trade_dates:
        day = df.loc[df["trade_date"] == trade_date].copy()
        calls = _build_effective_calls(day, repaired_mode=repaired_mode)
        if calls.empty:
            summary_rows.append(
                {
                    "trade_date": pd.Timestamp(trade_date),
                    "n_effective_quotes": 0,
                    "n_expirations": 0,
                    "lower_bound": 0,
                    "upper_bound": 0,
                    "european_bounds_total": 0,
                    "vertical_monotonicity": 0,
                    "vertical_slope_bound": 0,
                    "vertical_total": 0,
                    "butterfly": 0,
                    "calendar_total_variance": 0,
                    "total_violations": 0,
                }
            )
            continue

        counts = count_static_arbitrage(calls)
        summary_rows.append(
            {
                "trade_date": pd.Timestamp(trade_date),
                "n_effective_quotes": int(len(calls)),
                "n_expirations": int(calls["expiration"].nunique()),
                **counts,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("trade_date").reset_index(drop=True)

    print(f"Workbook: {workbook_path}")
    print(f"Sheet: {chosen_sheet}")
    print(f"Mode: {'repaired quote-level workbook' if repaired_mode else 'raw workbook'}")
    print()
    if summary.empty:
        print("No usable rows found.")
        return summary

    print(summary.to_string(index=False))
    print()

    numeric_cols = [
        "n_effective_quotes",
        "n_expirations",
        "lower_bound",
        "upper_bound",
        "european_bounds_total",
        "vertical_monotonicity",
        "vertical_slope_bound",
        "vertical_total",
        "butterfly",
        "calendar_total_variance",
        "total_violations",
    ]
    totals = summary[numeric_cols].sum(numeric_only=True)
    totals_row = pd.DataFrame([{"trade_date": "TOTAL", **totals.to_dict()}])
    print(totals_row.to_string(index=False))

    return summary


def main(stage: str = "post_fengler") -> dict[str, pd.DataFrame]:
    """
    Run static arbitrage checks across all workbooks in the relevant stage folder.

    Parameters
    ----------
    stage : {"pre_fengler", "post_fengler"}
        "pre_fengler"  -> reads deamericanized workbooks from DEAMERICANIZED_DIR
                          (sheet name: "deamericanized")
        "post_fengler" -> reads Fengler workbooks from FENGLER_DIR
                          (sheet name: "Sheet1")

    Returns
    -------
    dict mapping workbook stem -> per-trade-date summary DataFrame
    """
    ensure_dirs()

    if stage == "pre_fengler":
        input_dir = DEAMERICANIZED_DIR
        sheet_name = "deamericanized"
        pattern = "*.xlsx"
    elif stage == "post_fengler":
        input_dir = FENGLER_DIR
        sheet_name = "Sheet1"
        pattern = "*_combined_quote_level.xlsx"
    else:
        raise ValueError(f"Unknown stage: {stage!r}. Use 'pre_fengler' or 'post_fengler'.")

    workbooks = find_default_workbooks(input_dir, pattern)
    if not workbooks:
        raise FileNotFoundError(
            f"No workbooks matching {pattern!r} found in {input_dir} (stage: {stage})"
        )

    return {
        workbook.stem: run_static_arb_report(str(workbook), sheet_name)
        for workbook in workbooks
    }


if __name__ == "__main__":
    main()