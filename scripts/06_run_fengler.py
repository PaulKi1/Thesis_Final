from __future__ import annotations

from pathlib import Path

import pandas as pd

from thesis.config import DEAMERICANIZED_DIR, FENGLER_DIR, ensure_dirs
from thesis.fengler.repair import (
    FenglerConfig,
    repair_fengler_surface_single_date,
    save_outputs_to_excel,
)


def get_unique_trade_dates(excel_path: Path, sheet_name: str = "deamericanized") -> list[pd.Timestamp]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=["trade_date"])
    return sorted(pd.to_datetime(df["trade_date"]).dropna().dt.normalize().unique())


def infer_ticker(excel_path: Path, sheet_name: str = "deamericanized") -> str:
    preview = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=["underlying"], nrows=25)
    values = [str(value).strip().upper() for value in preview["underlying"].dropna().unique()]
    if not values:
        raise ValueError(f"Could not infer ticker from {excel_path}")
    return values[0]


def run_fengler_all_dates(
    excel_path: Path,
    output_folder: Path,
    config: FenglerConfig,
) -> pd.DataFrame:
    output_folder.mkdir(parents=True, exist_ok=True)
    all_dates = get_unique_trade_dates(excel_path, sheet_name=config.sheet_name)
    combined_frames: list[pd.DataFrame] = []

    for trade_date in all_dates:
        results = repair_fengler_surface_single_date(
            excel_path=str(excel_path),
            trade_date=trade_date,
            config=config,
        )
        date_label = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        save_outputs_to_excel(results, str(output_folder / f"{date_label}.xlsx"))
        combined_frames.append(results["quote_level_output"].copy())
        print(f"  Repaired {date_label}")

    if not combined_frames:
        return pd.DataFrame()

    combined = pd.concat(combined_frames, ignore_index=True)
    sort_cols = [
        column
        for column in ["trade_date", "expiration", "strike_price", "instrument_class"]
        if column in combined.columns
    ]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    return combined


def main() -> None:
    ensure_dirs()

    # This is the part that can change params of the smoother
    config = FenglerConfig(
        pre_smoother_smoothing=0.5,
        fixed_lambda=1,
    )

    input_workbooks = sorted(DEAMERICANIZED_DIR.glob("*.xlsx"))
    if not input_workbooks:
        raise FileNotFoundError(f"No deamericanized workbooks found in {DEAMERICANIZED_DIR}")

    for input_path in input_workbooks:
        ticker = infer_ticker(input_path, sheet_name=config.sheet_name)
        output_folder = FENGLER_DIR / f"{ticker}_by_date"
        combined_output = FENGLER_DIR / f"{ticker}_combined_quote_level.xlsx"

        print("\n" + "=" * 80)
        print(f"Processing {ticker}")
        print(f"Input:  {input_path.name}")
        print("=" * 80)

        combined = run_fengler_all_dates(
            excel_path=input_path,
            output_folder=output_folder,
            config=config,
        )
        if combined.empty:
            print(f"No quote-level output generated for {ticker}")
            continue

        with pd.ExcelWriter(combined_output, engine="openpyxl") as writer:
            combined.to_excel(writer, index=False)
        print(f"\nSaved combined {ticker} quote-level output to: {combined_output}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
