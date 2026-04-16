
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thesis.config import FENGLER_DIR, FIGURES_DIR, TABLES_DIR, ensure_dirs


# ============================================================
# USER SETTINGS
# Edit only this block.
# ============================================================

INPUT_FOLDER = FENGLER_DIR
SHEET_NAME: str | None = None

# Reference ETF / unleveraged product in the paper notation (beta = 1).
REFERENCE_PRODUCT = "NVDA"

# Assumption: if TARGET_DATE is unset or missing, use the latest trade date
# available across the reference product and at least one leveraged product.
TARGET_DATE: str | None = None

# Add every product you want to load here.
# Keep the reference product at beta = 1.
BETA_BY_PRODUCT = {
    "NVDA": 1.0,
    "NVDL": 2.0,
    "TSLA": 1.0,
    "TSLR": 2.0,
    "TSLT": 2.0
}

# Optional exact filename-to-product override if a filename contains both
# the LETF ticker and the reference ticker and simple filename inference
# picks the wrong one.
FILE_PRODUCT_HINTS = {
    # "NVDL_opra_cbbo_snapshots_with_cleaned_sheet_NVDA_underlying.xlsx": "NVDL",
}

ONLY_OTM = True

# The paper plots IVs against log(strike / current (L)ETF spot).
USE_PAPER_SPOT_LOG_MONEYNESS = True

# Use de-Americanized European mids already present in the sheet.
PRICE_COLUMN = "repaired_price_for_original_class"

OUTPUT_TABLE_DIR = TABLES_DIR
OUTPUT_FIGURE_DIR = FIGURES_DIR / "moneyness_scaling"
OUTPUT_WORKBOOK = OUTPUT_TABLE_DIR / "moneyness_scaling_outputs.xlsx"
ALL_ROWS_FILE = OUTPUT_TABLE_DIR / "moneyness_scaling_all_rows_enriched.csv"
FIGURE2_DATA_FILE = OUTPUT_TABLE_DIR / "figure2_data.csv"
FIGURE4_DATA_FILE = OUTPUT_TABLE_DIR / "figure4_data.csv"

# ============================================================
# End of user settings
# ============================================================

REQUIRED_COLUMNS = [
    "trade_date",
    "symbol",
    "expiration",
    "instrument_class",
    "strike_price",
    "underlying_price",
    "discount_factor",
    "carry",
    "forward_price",
    "tau",
    PRICE_COLUMN,
    "Log_Moneyness",
]


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black76_price(
    option_type: str,
    forward: float,
    strike: float,
    tau: float,
    discount_factor: float,
    sigma: float,
) -> float:
    option_type = option_type.upper()
    if tau <= 0.0:
        intrinsic = max(forward - strike, 0.0) if option_type == "C" else max(strike - forward, 0.0)
        return discount_factor * intrinsic

    if sigma <= 0.0:
        intrinsic = max(forward - strike, 0.0) if option_type == "C" else max(strike - forward, 0.0)
        return discount_factor * intrinsic

    vol_sqrt_t = sigma * math.sqrt(tau)
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * tau) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if option_type == "C":
        return discount_factor * (forward * normal_cdf(d1) - strike * normal_cdf(d2))
    if option_type == "P":
        return discount_factor * (strike * normal_cdf(-d2) - forward * normal_cdf(-d1))

    raise ValueError(f"Unsupported option type: {option_type}")


def implied_vol_black76(
    option_type: str,
    forward: float,
    strike: float,
    tau: float,
    discount_factor: float,
    option_price: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    option_type = option_type.upper()

    if not np.isfinite(forward) or not np.isfinite(strike) or not np.isfinite(tau):
        return np.nan
    if not np.isfinite(discount_factor) or not np.isfinite(option_price):
        return np.nan
    if forward <= 0.0 or strike <= 0.0 or tau <= 0.0 or discount_factor <= 0.0:
        return np.nan
    if option_price <= 0.0:
        return np.nan

    intrinsic = discount_factor * (
        max(forward - strike, 0.0) if option_type == "C" else max(strike - forward, 0.0)
    )
    upper_bound = discount_factor * (forward if option_type == "C" else strike)

    if option_price < intrinsic - 1e-7 or option_price > upper_bound + 1e-7:
        return np.nan
    if abs(option_price - intrinsic) <= tol:
        return 0.0

    low = 1e-8
    high = 1.0
    price_high = black76_price(option_type, forward, strike, tau, discount_factor, high)

    while price_high < option_price and high < 20.0:
        high *= 2.0
        price_high = black76_price(option_type, forward, strike, tau, discount_factor, high)

    if price_high < option_price:
        return np.nan

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price_mid = black76_price(option_type, forward, strike, tau, discount_factor, mid)

        if abs(price_mid - option_price) <= tol:
            return mid

        if price_mid < option_price:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def infer_product_from_filename(file_path: Path) -> Optional[str]:
    if file_path.name in FILE_PRODUCT_HINTS:
        return FILE_PRODUCT_HINTS[file_path.name]

    tokens = [token for token in re.split(r"[^A-Z0-9]+", file_path.stem.upper()) if token]
    matches_in_order = [token for token in tokens if token in BETA_BY_PRODUCT]

    if matches_in_order:
        return matches_in_order[0]

    return None


def infer_product_from_symbols(df: pd.DataFrame) -> Optional[str]:
    if "symbol" not in df.columns:
        return None

    roots = (
        df["symbol"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.extract(r"^([A-Z]+)")[0]
        .dropna()
    )

    if roots.empty:
        return None

    roots = roots[roots.isin(BETA_BY_PRODUCT)]
    if roots.empty:
        return None

    counts = roots.value_counts()
    return str(counts.index[0])


def infer_product(file_path: Path, df: pd.DataFrame) -> Optional[str]:
    product = infer_product_from_filename(file_path)
    if product is not None:
        return product

    return infer_product_from_symbols(df)


def flatten_axes(axes) -> list:
    if hasattr(axes, "flatten"):
        return list(axes.flatten())
    if isinstance(axes, (list, tuple)):
        return list(axes)
    return [axes]


def _pick_sheet_name(file_path: Path) -> str | int:
    workbook = pd.ExcelFile(file_path)
    if SHEET_NAME is not None:
        if SHEET_NAME not in workbook.sheet_names:
            raise ValueError(f"Sheet {SHEET_NAME!r} not found in {file_path.name}")
        return SHEET_NAME
    if "quote_level_output" in workbook.sheet_names:
        return "quote_level_output"
    return workbook.sheet_names[0]


def safe_read_input_file(file_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_excel(file_path, sheet_name=_pick_sheet_name(file_path))
    except Exception:
        return None

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return None

    product = infer_product(file_path, df)
    if product is None:
        print(f"Skipping {file_path.name}: could not infer product ticker.")
        return None

    if product not in BETA_BY_PRODUCT:
        print(f"Skipping {file_path.name}: product {product} is not in BETA_BY_PRODUCT.")
        return None

    df = df.copy()
    df["product"] = product
    df["beta"] = float(BETA_BY_PRODUCT[product])
    df["source_file"] = file_path.name
    return df


def add_plot_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["instrument_class"] = out["instrument_class"].astype(str).str.upper().str[0]

    valid_inputs = (
        out["instrument_class"].isin(["C", "P"])
        & np.isfinite(out["strike_price"])
        & np.isfinite(out["underlying_price"])
        & np.isfinite(out["forward_price"])
        & np.isfinite(out["discount_factor"])
        & np.isfinite(out["tau"])
        & np.isfinite(out[PRICE_COLUMN])
        & np.isfinite(out["beta"])
        & (out["strike_price"] > 0.0)
        & (out["underlying_price"] > 0.0)
        & (out["forward_price"] > 0.0)
        & (out["discount_factor"] > 0.0)
        & (out["tau"] > 0.0)
        & (out[PRICE_COLUMN] > 0.0)
        & (out["beta"] != 0.0)
    )

    out["LM_paper_exact"] = np.nan
    out.loc[valid_inputs, "LM_paper_exact"] = np.log(
        out.loc[valid_inputs, "strike_price"] / out.loc[valid_inputs, "underlying_price"]
    )

    if USE_PAPER_SPOT_LOG_MONEYNESS:
        out["LM_plot"] = out["LM_paper_exact"]
    else:
        out["LM_plot"] = out["Log_Moneyness"]

    out["is_otm"] = False
    call_otm = (out["instrument_class"] == "C") & (out["strike_price"] > out["underlying_price"])
    put_otm = (out["instrument_class"] == "P") & (out["strike_price"] < out["underlying_price"])
    out.loc[call_otm | put_otm, "is_otm"] = True

    out["std_bs_iv"] = np.nan
    calc_rows = out.index[valid_inputs]

    for idx in calc_rows:
        row = out.loc[idx]
        out.at[idx, "std_bs_iv"] = implied_vol_black76(
            option_type=row["instrument_class"],
            forward=float(row["forward_price"]),
            strike=float(row["strike_price"]),
            tau=float(row["tau"]),
            discount_factor=float(row["discount_factor"]),
            option_price=float(row[PRICE_COLUMN]),
        )

    out["iv_norm"] = out["std_bs_iv"] / out["beta"].abs()

    out["plot_eligible"] = valid_inputs & np.isfinite(out["LM_plot"]) & np.isfinite(out["iv_norm"])
    if ONLY_OTM:
        out["plot_eligible"] = out["plot_eligible"] & out["is_otm"]

    return out


def collapse_curve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    group_cols = ["trade_date", "product", "expiration", "LM_plot"]
    keep_cols = ["tau", "time_to_expiration", "discount_factor", "carry", "beta", "underlying_price", "forward_price"]

    agg_map = {col: "first" for col in keep_cols if col in df.columns}
    agg_map["iv_norm"] = "mean"

    collapsed = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(agg_map)
        .sort_values(["trade_date", "product", "expiration", "LM_plot"])
        .reset_index(drop=True)
    )
    return collapsed


def prepare_interp_curve(curve: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    curve = curve[["LM_plot", "iv_norm"]].dropna().copy()
    if curve.empty:
        return np.array([]), np.array([])

    curve = curve.groupby("LM_plot", as_index=False)["iv_norm"].mean()
    curve = curve.sort_values("LM_plot")
    return curve["LM_plot"].to_numpy(dtype=float), curve["iv_norm"].to_numpy(dtype=float)


def build_figure2_data(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    ref = panel[panel["product"] == REFERENCE_PRODUCT].copy()
    letf = panel[panel["product"] != REFERENCE_PRODUCT].copy()

    records: list[dict] = []

    for trade_date in sorted(panel["trade_date"].dropna().unique()):
        ref_day = ref[ref["trade_date"] == trade_date]

        if ref_day.empty:
            continue

        for product in sorted(letf.loc[letf["trade_date"] == trade_date, "product"].dropna().unique()):
            letf_day = letf[(letf["trade_date"] == trade_date) & (letf["product"] == product)]

            common_expiries = sorted(set(ref_day["expiration"].dropna()) & set(letf_day["expiration"].dropna()))

            for expiry in common_expiries:
                ref_curve = ref_day[ref_day["expiration"] == expiry]
                letf_curve = letf_day[letf_day["expiration"] == expiry]

                x_ref, y_ref = prepare_interp_curve(ref_curve)
                if len(x_ref) < 2:
                    continue

                x_min = x_ref.min()
                x_max = x_ref.max()

                for _, row in letf_curve.iterrows():
                    x = float(row["LM_plot"])
                    if x < x_min or x > x_max:
                        continue

                    iv_ref_interp = float(np.interp(x, x_ref, y_ref))
                    if iv_ref_interp <= 0.0 or not np.isfinite(iv_ref_interp):
                        continue

                    records.append(
                        {
                            "trade_date": trade_date,
                            "product": product,
                            "reference_product": REFERENCE_PRODUCT,
                            "expiration": expiry,
                            "tau": row["tau"],
                            "time_to_expiration": row.get("time_to_expiration", np.nan),
                            "beta": row["beta"],
                            "LM_plot": x,
                            "iv_norm": row["iv_norm"],
                            "iv_ref_interp": iv_ref_interp,
                            "iv_ratio_ref": float(row["iv_norm"]) / iv_ref_interp,
                        }
                    )

    return pd.DataFrame.from_records(records)


def build_figure4_data(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    ref = panel[panel["product"] == REFERENCE_PRODUCT].copy()
    letf = panel[panel["product"] != REFERENCE_PRODUCT].copy()

    records: list[dict] = []

    for trade_date in sorted(panel["trade_date"].dropna().unique()):
        ref_day = ref[ref["trade_date"] == trade_date]
        letf_day = letf[letf["trade_date"] == trade_date]

        if ref_day.empty or letf_day.empty:
            continue

        common_expiries = sorted(set(ref_day["expiration"].dropna()) & set(letf_day["expiration"].dropna()))

        for expiry in common_expiries:
            ref_curve = ref_day[ref_day["expiration"] == expiry].copy()
            if ref_curve.empty:
                continue

            avg_ref_iv = float(ref_curve["iv_norm"].mean())

            for _, row in ref_curve.iterrows():
                records.append(
                    {
                        "trade_date": trade_date,
                        "product": REFERENCE_PRODUCT,
                        "reference_product": REFERENCE_PRODUCT,
                        "expiration": expiry,
                        "tau": row["tau"],
                        "time_to_expiration": row.get("time_to_expiration", np.nan),
                        "beta": 1.0,
                        "iv_norm": row["iv_norm"],
                        "avg_ref_iv": avg_ref_iv,
                        "LM_ref_axis": row["LM_plot"],
                    }
                )

            letf_same_expiry = letf_day[letf_day["expiration"] == expiry].copy()
            if letf_same_expiry.empty:
                continue

            valid_scale = (
                np.isfinite(letf_same_expiry["LM_plot"])
                & np.isfinite(letf_same_expiry["discount_factor"])
                & np.isfinite(letf_same_expiry["carry"])
                & np.isfinite(letf_same_expiry["tau"])
                & (letf_same_expiry["discount_factor"] > 0.0)
                & (letf_same_expiry["tau"] > 0.0)
            )
            letf_same_expiry = letf_same_expiry[valid_scale]

            for _, row in letf_same_expiry.iterrows():
                beta = float(row["beta"])
                tau = float(row["tau"])
                rT = -math.log(float(row["discount_factor"]))
                cT = float(row["carry"]) * tau

                lm_ref_axis = (
                    float(row["LM_plot"])
                    + (beta - 1.0) * rT
                    + cT
                    + 0.5 * beta * (beta - 1.0) * (avg_ref_iv ** 2) * tau
                ) / beta

                records.append(
                    {
                        "trade_date": trade_date,
                        "product": row["product"],
                        "reference_product": REFERENCE_PRODUCT,
                        "expiration": expiry,
                        "tau": tau,
                        "time_to_expiration": row.get("time_to_expiration", np.nan),
                        "beta": beta,
                        "iv_norm": row["iv_norm"],
                        "avg_ref_iv": avg_ref_iv,
                        "LM_ref_axis": lm_ref_axis,
                    }
                )

    return pd.DataFrame.from_records(records)


def make_figure2_plots(panel: pd.DataFrame, figure2_data: pd.DataFrame, plot_folder: Path) -> None:
    if panel.empty or figure2_data.empty:
        print("Figure 2 plots skipped: no ETF/LETF common expiries after filtering.")
        return

    ref = panel[panel["product"] == REFERENCE_PRODUCT].copy()
    letf = panel[panel["product"] != REFERENCE_PRODUCT].copy()

    for trade_date in sorted(figure2_data["trade_date"].dropna().unique()):
        ref_day = ref[ref["trade_date"] == trade_date]

        for product in sorted(figure2_data.loc[figure2_data["trade_date"] == trade_date, "product"].dropna().unique()):
            letf_day = letf[(letf["trade_date"] == trade_date) & (letf["product"] == product)]
            ratio_day = figure2_data[(figure2_data["trade_date"] == trade_date) & (figure2_data["product"] == product)]

            common_expiries = sorted(ratio_day["expiration"].dropna().unique())
            if not common_expiries:
                continue

            n_panels = len(common_expiries) + 1
            ncols = min(3, n_panels)
            nrows = math.ceil(n_panels / ncols)

            fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows))
            axes_list = flatten_axes(axes)

            for ax_idx, expiry in enumerate(common_expiries):
                ax = axes_list[ax_idx]

                ref_curve = ref_day[ref_day["expiration"] == expiry]
                letf_curve = letf_day[letf_day["expiration"] == expiry]
                ttm_values = ratio_day.loc[ratio_day["expiration"] == expiry, "time_to_expiration"].dropna()
                days = int(ttm_values.iloc[0]) if not ttm_values.empty else int(round(float(ref_curve["tau"].iloc[0]) * 365))

                ax.scatter(ref_curve["LM_plot"], ref_curve["iv_norm"], label=REFERENCE_PRODUCT, s=18)
                ax.scatter(letf_curve["LM_plot"], letf_curve["iv_norm"], label=product, s=18)
                ax.set_title(f"{days} days")
                ax.set_xlabel("Log moneyness")
                ax.set_ylabel("Normalized IV")
                ax.legend()

            ratio_ax = axes_list[len(common_expiries)]
            for expiry in common_expiries:
                ratio_curve = ratio_day[ratio_day["expiration"] == expiry]
                ttm_values = ratio_curve["time_to_expiration"].dropna()
                days = int(ttm_values.iloc[0]) if not ttm_values.empty else int(round(float(ratio_curve["tau"].iloc[0]) * 365))

                ratio_ax.scatter(ratio_curve["LM_plot"], ratio_curve["iv_ratio_ref"], label=str(days), s=18)

            ratio_ax.set_title(f"Ratios {product}:{REFERENCE_PRODUCT}")
            ratio_ax.set_xlabel("Log moneyness")
            ratio_ax.set_ylabel("IV ratio")
            ratio_ax.legend(title="Days")

            for extra_ax in axes_list[n_panels:]:
                extra_ax.axis("off")

            fig.tight_layout()

            trade_label = pd.Timestamp(trade_date).strftime("%Y%m%d")
            outfile = plot_folder / f"figure2_{trade_label}_{product}.png"
            fig.savefig(outfile, dpi=150, bbox_inches="tight")
            plt.close(fig)


def make_figure4_plots(figure4_data: pd.DataFrame, plot_folder: Path) -> None:
    if figure4_data.empty:
        print("Figure 4 plots skipped: no ETF/LETF common expiries after filtering.")
        return

    for trade_date in sorted(figure4_data["trade_date"].dropna().unique()):
        day = figure4_data[figure4_data["trade_date"] == trade_date]

        for expiry in sorted(day["expiration"].dropna().unique()):
            expiry_slice = day[day["expiration"] == expiry]
            ref_curve = expiry_slice[expiry_slice["product"] == REFERENCE_PRODUCT]
            letf_products = sorted(expiry_slice.loc[expiry_slice["product"] != REFERENCE_PRODUCT, "product"].dropna().unique())

            if ref_curve.empty or not letf_products:
                continue

            n_panels = len(letf_products)
            ncols = min(3, n_panels)
            nrows = math.ceil(n_panels / ncols)

            fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows))
            axes_list = flatten_axes(axes)

            ttm_values = expiry_slice["time_to_expiration"].dropna()
            days = int(ttm_values.iloc[0]) if not ttm_values.empty else int(round(float(expiry_slice["tau"].iloc[0]) * 365))

            for ax_idx, product in enumerate(letf_products):
                ax = axes_list[ax_idx]
                product_curve = expiry_slice[expiry_slice["product"] == product]

                ax.scatter(ref_curve["LM_ref_axis"], ref_curve["iv_norm"], label=REFERENCE_PRODUCT, s=18)
                ax.scatter(product_curve["LM_ref_axis"], product_curve["iv_norm"], label=product, s=18)
                ax.set_title(f"{product} | {days} days")
                ax.set_xlabel(f"{REFERENCE_PRODUCT} log moneyness axis")
                ax.set_ylabel("Normalized IV")
                ax.legend()

            for extra_ax in axes_list[n_panels:]:
                extra_ax.axis("off")

            fig.tight_layout()

            trade_label = pd.Timestamp(trade_date).strftime("%Y%m%d")
            outfile = plot_folder / f"figure4_{trade_label}_{days}d.png"
            fig.savefig(outfile, dpi=150, bbox_inches="tight")
            plt.close(fig)


def save_outputs(
    enriched: pd.DataFrame,
    figure2_data: pd.DataFrame,
    figure4_data: pd.DataFrame,
) -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_WORKBOOK, engine="openpyxl") as writer:
        enriched.to_excel(writer, sheet_name="all_rows_enriched", index=False)
        figure2_data.to_excel(writer, sheet_name="figure2_data", index=False)
        figure4_data.to_excel(writer, sheet_name="figure4_data", index=False)

    enriched.to_csv(ALL_ROWS_FILE, index=False)
    figure2_data.to_csv(FIGURE2_DATA_FILE, index=False)
    figure4_data.to_csv(FIGURE4_DATA_FILE, index=False)


def resolve_target_date(enriched: pd.DataFrame) -> pd.Timestamp:
    work = enriched.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])

    eligible_dates: list[pd.Timestamp] = []
    for trade_date, day in work.groupby("trade_date", sort=True):
        products = set(day["product"].dropna())
        if REFERENCE_PRODUCT in products and any(product != REFERENCE_PRODUCT for product in products):
            eligible_dates.append(pd.Timestamp(trade_date))

    if not eligible_dates:
        raise RuntimeError(
            "No trade date contains both the reference product and at least one leveraged product."
        )

    if TARGET_DATE is not None:
        explicit_date = pd.Timestamp(TARGET_DATE)
        if explicit_date in eligible_dates:
            return explicit_date
        print(
            f"TARGET_DATE {explicit_date.date()} not found in the Fengler outputs; "
            f"using {eligible_dates[-1].date()} instead."
        )

    return eligible_dates[-1]


def main() -> None:
    ensure_dirs()
    if REFERENCE_PRODUCT not in BETA_BY_PRODUCT:
        raise ValueError("REFERENCE_PRODUCT must be present in BETA_BY_PRODUCT.")
    if float(BETA_BY_PRODUCT[REFERENCE_PRODUCT]) != 1.0:
        raise ValueError("The reference product must have beta = 1.0.")
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"INPUT_FOLDER does not exist: {INPUT_FOLDER}")

    xlsx_files = sorted(INPUT_FOLDER.glob("*_combined_quote_level.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(
            f"No Fengler combined quote-level workbooks matching '*_combined_quote_level.xlsx' in {INPUT_FOLDER}"
        )

    loaded_frames: list[pd.DataFrame] = []

    for file_path in xlsx_files:
        if file_path.resolve() == OUTPUT_WORKBOOK.resolve():
            continue

        df = safe_read_input_file(file_path)
        if df is None:
            continue

        loaded_frames.append(df)
        product = df["product"].iloc[0]
        print(f"Loaded {file_path.name} as product {product} (beta={BETA_BY_PRODUCT[product]}).")

    if not loaded_frames:
        raise RuntimeError("No valid input files were loaded.")

    raw = pd.concat(loaded_frames, ignore_index=True)
    enriched = add_plot_columns(raw)
    target_date = resolve_target_date(enriched)
    enriched["trade_date"] = pd.to_datetime(enriched["trade_date"])
    enriched = enriched[enriched["trade_date"] == target_date].copy()

    plot_panel = enriched[enriched["plot_eligible"]].copy()
    plot_panel = collapse_curve_duplicates(plot_panel)

    figure2_data = build_figure2_data(plot_panel)
    figure4_data = build_figure4_data(plot_panel)
    if figure2_data.empty or figure4_data.empty:
        raise RuntimeError(
            "Stage 08 could not build both figure2_data and figure4_data from the Fengler outputs."
        )

    figure2_folder = OUTPUT_FIGURE_DIR / "figure2_plots"
    figure4_folder = OUTPUT_FIGURE_DIR / "figure4_plots"
    figure2_folder.mkdir(parents=True, exist_ok=True)
    figure4_folder.mkdir(parents=True, exist_ok=True)

    make_figure2_plots(plot_panel, figure2_data, figure2_folder)
    make_figure4_plots(figure4_data, figure4_folder)

    save_outputs(enriched, figure2_data, figure4_data)

    print()
    print(f"Trade date used: {target_date.date()}")
    print(f"Rows loaded: {len(enriched):,}")
    print(f"Rows kept for plotting: {len(plot_panel):,}")
    print(f"Figure 2 points: {len(figure2_data):,}")
    print(f"Figure 4 points: {len(figure4_data):,}")
    print(f"Outputs written to: {OUTPUT_TABLE_DIR} and {OUTPUT_FIGURE_DIR}")


if __name__ == "__main__":
    main()
