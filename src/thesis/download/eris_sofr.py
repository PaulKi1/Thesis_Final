from __future__ import annotations
import argparse
import calendar
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import pandas as pd

ACTIVE_BASE_URL = "https://files.erisfutures.com/ftp"
ARCHIVE_BASE_URL = "https://files.erisfutures.com/ftp/archives"
FILE_TEMPLATE = "Eris_{yyyymmdd}_EOD_DiscountFactors_SOFR.csv"
EXPECTED_COLUMNS = {
    "Date",
    "DiscountFactor",
    "SpotRate (Actual360 Continuous)",
    "ForwardRate",
    "ValueDate",
    "MaturityDate",
}


@dataclass
class DownloadResult:
    trade_date: date
    status: str
    url: Optional[str]
    local_path: Optional[Path]
    error: Optional[str] = None


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_calendar_days(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def archive_month_folder(d: date) -> str:
    return f"{d.year}/{d.month:02d}-{calendar.month_name[d.month]}"


def file_name_for_trade_date(trade_date: date) -> str:
    return FILE_TEMPLATE.format(yyyymmdd=trade_date.strftime("%Y%m%d"))


def candidate_urls(trade_date: date) -> list[str]:
    filename = file_name_for_trade_date(trade_date)
    archive_url = f"{ARCHIVE_BASE_URL}/{archive_month_folder(trade_date)}/{filename}"
    active_url = f"{ACTIVE_BASE_URL}/{filename}"

    # Archive first works well for historical pulls such as all of 2025.
    return [archive_url, active_url]


def try_download_file(trade_date: date, raw_dir: Path, timeout_seconds: int = 30) -> DownloadResult:
    filename = file_name_for_trade_date(trade_date)
    destination = raw_dir / filename

    if destination.exists():
        return DownloadResult(
            trade_date=trade_date,
            status="cached",
            url=None,
            local_path=destination,
            error=None,
        )

    errors: list[str] = []

    for url in candidate_urls(trade_date):
        try:
            with urlopen(url, timeout=timeout_seconds) as response, destination.open("wb") as fh:
                shutil.copyfileobj(response, fh)
            return DownloadResult(
                trade_date=trade_date,
                status="downloaded",
                url=url,
                local_path=destination,
                error=None,
            )
        except HTTPError as exc:
            errors.append(f"{url} -> HTTP {exc.code}")
        except URLError as exc:
            errors.append(f"{url} -> {exc.reason}")
        except Exception as exc:  # pragma: no cover - keeps the script robust in user envs
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")

    if destination.exists():
        destination.unlink(missing_ok=True)

    return DownloadResult(
        trade_date=trade_date,
        status="missing",
        url=None,
        local_path=None,
        error=" | ".join(errors) if errors else "No candidate URL succeeded.",
    )


def normalize_curve_file(local_path: Path, trade_date: date, max_tenor_days: int) -> pd.DataFrame:
    df = pd.read_csv(local_path)

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Unexpected file format in {local_path.name}. Missing columns: {sorted(missing_cols)}"
        )

    df["curve_date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    if df["curve_date"].isna().any():
        bad_rows = int(df["curve_date"].isna().sum())
        raise ValueError(f"Could not parse {bad_rows} date values in {local_path.name}.")

    df["trade_date"] = pd.Timestamp(trade_date)
    df["days_to_expiry"] = (df["curve_date"] - df["trade_date"]).dt.days

    df = df.loc[
        (df["days_to_expiry"] >= 0) & (df["days_to_expiry"] <= max_tenor_days)
    ].copy()

    df = df.rename(
        columns={
            "DiscountFactor": "discount_factor",
            "SpotRate (Actual360 Continuous)": "spot_rate_act360_continuous",
        }
    )

    out = df[
        [
            "trade_date",
            "curve_date",
            "days_to_expiry",
            "discount_factor",
            "spot_rate_act360_continuous",
        ]
    ].copy()

    out["source_file"] = local_path.name
    return out.sort_values(["trade_date", "curve_date"]).reset_index(drop=True)


def build_lookup_table(
    start_date: date,
    end_date: date,
    output_dir: Path,
    max_tenor_days: int = 1095,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_log: list[dict] = []
    frames: list[pd.DataFrame] = []

    for trade_date in iter_calendar_days(start_date, end_date):
        result = try_download_file(trade_date=trade_date, raw_dir=raw_dir)
        download_log.append(
            {
                "trade_date": trade_date.isoformat(),
                "status": result.status,
                "url": result.url,
                "local_path": str(result.local_path) if result.local_path else None,
                "error": result.error,
            }
        )

        if result.local_path is None:
            continue

        try:
            frames.append(normalize_curve_file(result.local_path, trade_date, max_tenor_days))
        except Exception as exc:
            download_log[-1]["status"] = "parse_error"
            download_log[-1]["error"] = str(exc)

    log_df = pd.DataFrame(download_log)
    log_df.to_csv(output_dir / "download_log.csv", index=False)

    if not frames:
        raise RuntimeError(
            "No Eris SOFR discount-factor files were downloaded for the requested date range. "
            "Check the date range, internet access, or the public file paths."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.drop_duplicates(subset=["trade_date", "curve_date", "source_file"])
        .sort_values(["trade_date", "curve_date"])
        .reset_index(drop=True)
    )

    combined_csv = output_dir / "sofr_discount_curve_lookup.csv"
    combined.to_csv(combined_csv, index=False)

    # Optional one-sheet Excel export. If openpyxl is not installed, the CSV is still enough.
    combined_xlsx = output_dir / "sofr_discount_curve_lookup.xlsx"
    try:
        with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
            combined.to_excel(writer, sheet_name="curve_lookup", index=False)
    except ImportError:
        pass

    return combined, log_df


def map_discount_factor_to_option_chain(
    option_chain: pd.DataFrame,
    curve_lookup: pd.DataFrame,
    trade_date_col: str = "trade_date",
    expiry_date_col: str = "expiration_date",
) -> pd.DataFrame:
    """
    Exact-date lookup helper for a listed option chain.

    Expected behavior:
    - trade_date in the option chain matches the trade_date in the curve table
    - expiration_date in the option chain matches the curve_date in the curve table

    Because the Eris discount-factor file already contains one row per calendar
    date, this is usually an exact merge, not an interpolation problem.
    """
    opt = option_chain.copy()
    curve = curve_lookup.copy()

    opt[trade_date_col] = pd.to_datetime(opt[trade_date_col]).dt.normalize()
    opt[expiry_date_col] = pd.to_datetime(opt[expiry_date_col]).dt.normalize()
    curve["trade_date"] = pd.to_datetime(curve["trade_date"]).dt.normalize()
    curve["curve_date"] = pd.to_datetime(curve["curve_date"]).dt.normalize()

    merged = opt.merge(
        curve[
            [
                "trade_date",
                "curve_date",
                "days_to_expiry",
                "discount_factor",
                "spot_rate_act360_continuous",
            ]
        ],
        left_on=[trade_date_col, expiry_date_col],
        right_on=["trade_date", "curve_date"],
        how="left",
    )

    merged = merged.drop(columns=["curve_date"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Eris EOD SOFR discount-factor files, save raw files, and "
            "create one combined lookup table for option-expiry discounting."
        )
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder to save raw and combined files. Defaults to config.ERIS_SOFR_DIR.",
    )
    parser.add_argument(
        "--max-tenor-days",
        type=int,
        default=1095,
        help="Keep only dates from trade date through this many calendar days. Default: 1095.",
    )
    args = parser.parse_args()

    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    if end_date < start_date:
        raise ValueError("end-date must be on or after start-date")

    # Default to config path if not specified
    if args.output_dir is None:
        from thesis.config import ERIS_SOFR_DIR, ensure_dirs
        ensure_dirs()
        output_dir = ERIS_SOFR_DIR
    else:
        output_dir = args.output_dir

    combined, log_df = build_lookup_table(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        max_tenor_days=args.max_tenor_days,
    )

    successful_days = int(log_df[log_df["status"].isin(["downloaded", "cached"])] .shape[0])
    print(f"Saved raw files to: {output_dir / 'raw'}")
    print(f"Saved combined lookup CSV to: {output_dir / 'sofr_discount_curve_lookup.csv'}")
    if (output_dir / "sofr_discount_curve_lookup.xlsx").exists():
        print(f"Saved combined lookup XLSX to: {output_dir / 'sofr_discount_curve_lookup.xlsx'}")
    print(f"Saved download log to: {output_dir / 'download_log.csv'}")
    print(f"Downloaded or reused {successful_days} daily curve file(s).")
    print(f"Combined lookup rows: {len(combined):,}")


if __name__ == "__main__":
    main()
