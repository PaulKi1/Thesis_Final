"""Download Eris SOFR discount curves for the thesis date range.

Outputs go to data/raw/eris_sofr/ (configured in thesis.config).
Adjust the date range below to match your sample period.
"""
from thesis.config import ERIS_SOFR_DIR, ensure_dirs
from thesis.download.eris_sofr import build_lookup_table, parse_iso_date

# Adjust these to your thesis sample period
START_DATE = "2025-01-01"
END_DATE = "2025-01-03"
MAX_TENOR_DAYS = 1095

if __name__ == "__main__":
    ensure_dirs()

    combined, log_df = build_lookup_table(
        start_date=parse_iso_date(START_DATE),
        end_date=parse_iso_date(END_DATE),
        output_dir=ERIS_SOFR_DIR,
        max_tenor_days=MAX_TENOR_DAYS,
    )

    successful = int(log_df[log_df["status"].isin(["downloaded", "cached"])].shape[0])
    print(f"\nDownloaded or reused {successful} daily curve file(s).")
    print(f"Combined lookup rows: {len(combined):,}")
    print(f"Output: {ERIS_SOFR_DIR}")