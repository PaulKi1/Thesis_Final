from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import pandas as pd
from thesis.deamericanization.pipeline import run_pipeline

TICKER_BRANCH = {
    "NVDA": "underlying",
    "TSLA": "underlying",
    "NVDL": "letf",
    "NVDD": "letf",
    "NVDQ": "letf",
    "TSLR": "letf",
    "TSLS": "letf",
    "TSLT": "letf",
}

# Annualized carry floors in percent.
# This is expense ratio + swap spread
# Based on Annual report 2025, since 2024 is not available
EXPENSE_RATIOS_PCT: dict[str, float] = {
    "NVDA": 0.00, # UL
    "NVDL": 4.06, # 1.06% + 300bps
}


def _coerce_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        s = str(value).strip().replace("\u00a0", "")
        if s == "":
            return float("nan")
        s = s.replace(" ", "")
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        elif "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "")
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")
        return float(s)


def _detect_column(df: pd.DataFrame, candidates: list[str], *, label: str) -> str:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(f"Could not infer {label} column from DataFrame. Tried: {candidates}")


def _normalize_letf_carry_floor_lookup(
    letf_carry_floors_pct: Mapping[str, float] | pd.DataFrame | None,
    *,
    ticker_col: str | None = None,
    value_col: str | None = None,
) -> dict[str, float]:
    if letf_carry_floors_pct is None:
        return {}

    if isinstance(letf_carry_floors_pct, Mapping):
        return {
            str(k).strip().upper(): _coerce_float(v) / 100.0
            for k, v in letf_carry_floors_pct.items()
            if str(k).strip() != ""
        }

    if not isinstance(letf_carry_floors_pct, pd.DataFrame):
        raise TypeError("letf_carry_floors_pct must be a mapping, a DataFrame, or None.")

    carry_df = letf_carry_floors_pct.copy()
    ticker_name = ticker_col or _detect_column(
        carry_df,
        ["underlying", "ticker", "symbol"],
        label="ticker",
    )
    value_name = value_col or _detect_column(
        carry_df,
        ["carry_pct", "expense_ratio_pct", "carry", "expense_ratio"],
        label="carry",
    )

    out: dict[str, float] = {}
    for _, row in carry_df[[ticker_name, value_name]].dropna(how="any").iterrows():
        ticker = str(row[ticker_name]).strip().upper()
        if ticker == "":
            continue
        out[ticker] = _coerce_float(row[value_name]) / 100.0
    return out


def _format_cutoff_label(cutoff_days: float | None) -> str:
    if cutoff_days is None:
        return "na"
    cutoff = _coerce_float(cutoff_days)
    if cutoff.is_integer():
        return str(int(cutoff))
    return str(cutoff).replace(".", "p")


def run_folder(
    input_folder: str | Path,
    *,
    output_folder: str | Path | None = None,
    ticker_branch: dict[str, str] | None = None,
    input_sheet: str | None = None,
    letf_carry_floors_pct: Mapping[str, float] | pd.DataFrame | None = None,
    letf_carry_ticker_col: str | None = None,
    letf_carry_value_col: str | None = None,
    letf_test_with_and_without_long_end: bool = False,
    letf_exclude_long_end_from_fit: bool = False,
    letf_long_end_cutoff_days: float | None = None,
    **pipeline_kwargs,
) -> None:
    input_folder = Path(input_folder).expanduser().resolve()
    output_folder = Path(output_folder).expanduser().resolve() if output_folder else input_folder / "deamericanized_out"
    output_folder.mkdir(parents=True, exist_ok=True)

    ticker_branch = {
        str(k).strip().upper(): str(v).strip().lower()
        for k, v in (ticker_branch or TICKER_BRANCH).items()
    }
    carry_floor_lookup = _normalize_letf_carry_floor_lookup(
        letf_carry_floors_pct,
        ticker_col=letf_carry_ticker_col,
        value_col=letf_carry_value_col,
    )

    if (letf_test_with_and_without_long_end or letf_exclude_long_end_from_fit) and (
        letf_long_end_cutoff_days is None or _coerce_float(letf_long_end_cutoff_days) <= 0.0
    ):
        raise ValueError("letf_long_end_cutoff_days must be positive when testing or excluding the long end.")

    cutoff_label = _format_cutoff_label(letf_long_end_cutoff_days)

    for path in sorted(input_folder.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue

        df = pd.read_excel(path, sheet_name=0 if input_sheet is None else input_sheet)
        if isinstance(df, dict):
            df = next(iter(df.values()))

        lookup = {str(c).strip().lower(): c for c in df.columns}
        if "underlying" not in lookup:
            print(f"SKIP {path.name}: missing 'underlying' column")
            continue

        ticker_col = lookup["underlying"]
        available = {str(x).strip().upper() for x in df[ticker_col].dropna().unique()}

        for ticker in sorted(available & set(ticker_branch)):
            branch = ticker_branch[ticker]
            run_specs = [(False, "")]
            if branch == "letf":
                if letf_test_with_and_without_long_end:
                    run_specs = [
                        (False, "_with_long_end"),
                        (True, f"_without_long_end_gt{cutoff_label}d"),
                    ]
                elif letf_exclude_long_end_from_fit:
                    run_specs = [(True, f"_without_long_end_gt{cutoff_label}d")]

            for exclude_long_end, suffix in run_specs:
                out_path = output_folder / f"{path.stem}_{ticker}_{branch}{suffix}.xlsx"
                try:
                    per_run_kwargs = dict(pipeline_kwargs)
                    if branch == "letf":
                        per_run_kwargs.setdefault("letf_carry_floor", carry_floor_lookup.get(ticker))
                        per_run_kwargs.setdefault("letf_monotone_mode", "none")
                        per_run_kwargs["letf_exclude_long_end_from_fit"] = bool(exclude_long_end)
                        per_run_kwargs["letf_long_end_cutoff_days"] = letf_long_end_cutoff_days

                    run_pipeline(
                        df,
                        branch=branch,
                        ticker=ticker,
                        input_sheet=input_sheet,
                        output_path=out_path,
                        **per_run_kwargs,
                    )
                    print(f"OK   {path.name} -> {out_path.name}")
                except Exception as exc:
                    print(f"FAIL {path.name} [{ticker}/{branch}{suffix}]: {exc}")


if __name__ == "__main__":
    from thesis.config import INTERIM_DIR, DEAMERICANIZED_DIR, ensure_dirs

    ensure_dirs()

    run_folder(
        input_folder=INTERIM_DIR,
        output_folder=DEAMERICANIZED_DIR,
        letf_carry_floors_pct=EXPENSE_RATIOS_PCT,
        letf_test_with_and_without_long_end=False,
        letf_long_end_cutoff_days=365,
        letf_monotone_mode="none",
    )
