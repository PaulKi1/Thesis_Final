from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from thesis.config import DEAMERICANIZED_DIR, FIGURES_DIR, TABLES_DIR, ensure_dirs

MATURITY_ORDER = ["<=1M", "1-3M", "3-6M", ">6M"]
MONEYNESS_ORDER = ["Deep ITM", "ITM", "ATM", "OTM", "Deep OTM"]


def _find_one(pattern: str) -> Path:
    matches = sorted(DEAMERICANIZED_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {DEAMERICANIZED_DIR}")
    return matches[0]


def assign_maturity_bucket(tau: float) -> str:
    if tau <= 1 / 12:
        return "<=1M"
    if tau <= 0.25:
        return "1-3M"
    if tau <= 0.5:
        return "3-6M"
    return ">6M"


def assign_moneyness_bucket(row: pd.Series) -> str:
    m = row["Moneyness"]
    inst = str(row["instrument_class"]).upper()

    if inst == "C":
        if m <= 0.90:
            return "Deep ITM"
        if m <= 0.95:
            return "ITM"
        if m < 1.05:
            return "ATM"
        if m < 1.10:
            return "OTM"
        return "Deep OTM"

    if m >= 1.10:
        return "Deep ITM"
    if m >= 1.05:
        return "ITM"
    if m > 0.95:
        return "ATM"
    if m > 0.90:
        return "OTM"
    return "Deep OTM"


def load_default_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    nvda_file = _find_one("*NVDA*underlying*.xlsx")
    nvdl_file = _find_one("*NVDL*letf*.xlsx")
    df_nvda = pd.read_excel(nvda_file, sheet_name="deamericanized")
    df_nvdl = pd.read_excel(nvdl_file, sheet_name="deamericanized")
    return df_nvda, df_nvdl


def prepare_eep_frames(
    df_nvda: pd.DataFrame,
    df_nvdl: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_nvda = df_nvda.copy()
    df_nvdl = df_nvdl.copy()

    df_nvda["EEP"] = df_nvda["mid_px"] - df_nvda["mid_px_eu"]
    df_nvda["EEP_pct"] = (df_nvda["EEP"] / df_nvda["underlying_price"]) * 100
    df_nvdl["EEP"] = df_nvdl["mid_px"] - df_nvdl["mid_px_eu"]
    df_nvdl["EEP_pct"] = (df_nvdl["EEP"] / df_nvdl["underlying_price"]) * 100

    df_nvda_informative = df_nvda[df_nvda["instrument_class"] == "P"].copy()
    df_nvdl_informative = df_nvdl.copy()

    for frame in (df_nvda_informative, df_nvdl_informative):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["maturity_bucket"] = frame["tau"].apply(assign_maturity_bucket)
        frame["moneyness_bucket"] = frame.apply(assign_moneyness_bucket, axis=1)

    nvda_daily = df_nvda_informative.groupby("trade_date", as_index=False)["EEP_pct"].mean()
    nvda_daily = nvda_daily.rename(columns={"trade_date": "date", "EEP_pct": "avg_EEP_pct"})
    nvda_daily["ticker"] = "NVDA (Puts only)"

    nvdl_daily = df_nvdl_informative.groupby("trade_date", as_index=False)["EEP_pct"].mean()
    nvdl_daily = nvdl_daily.rename(columns={"trade_date": "date", "EEP_pct": "avg_EEP_pct"})
    nvdl_daily["ticker"] = "NVDL (Calls & Puts)"

    return df_nvda_informative, df_nvdl_informative, nvda_daily, nvdl_daily


def save_plot(nvda_daily: pd.DataFrame, nvdl_daily: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(
        nvda_daily["date"],
        nvda_daily["avg_EEP_pct"],
        label="NVDA (Puts only)",
        color="#2E86AB",
        linewidth=1.5,
        alpha=0.9,
    )
    ax.plot(
        nvdl_daily["date"],
        nvdl_daily["avg_EEP_pct"],
        label="NVDL (Calls & Puts)",
        color="#E94F37",
        linewidth=1.5,
        alpha=0.9,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Average EEP (% of Underlying)", fontsize=12)
    ax.set_title(
        "Model-Implied Early Exercise Premium Proxy Over Time\n"
        r"$\widehat{\mathrm{EEP}}_i = (Q^{Am}_i - \widetilde{Q}^{Eu}_i) / S_i \times 100\%$",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = FIGURES_DIR / "eep_diagnostics_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_statistics(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n{'=' * 80}")
    print(f"EEP STATISTICS FOR {name} (% of Underlying)")
    print(f"{'=' * 80}")
    print("\nOverall EEP Statistics (%):")
    print(f"  Count:            {len(df):,}")
    print(f"  Mean:             {df['EEP_pct'].mean():.4f}%")
    print(f"  Median:           {df['EEP_pct'].median():.4f}%")
    print(f"  Std Dev:          {df['EEP_pct'].std():.4f}%")
    print(f"  95th Percentile:  {df['EEP_pct'].quantile(0.95):.4f}%")
    print(f"  99th Percentile:  {df['EEP_pct'].quantile(0.99):.4f}%")
    print(f"  Max:              {df['EEP_pct'].max():.4f}%")

    print("\nBy Maturity Bucket (%):")
    print(f"  {'Bucket':<10} {'Count':>10} {'Median':>12} {'95th Pctl':>12} {'99th Pctl':>12}")
    print(f"  {'-' * 56}")
    for bucket in MATURITY_ORDER:
        subset = df[df["maturity_bucket"] == bucket]["EEP_pct"]
        if len(subset) > 0:
            print(
                f"  {bucket:<10} {len(subset):>10,} {subset.median():>11.4f}% "
                f"{subset.quantile(0.95):>11.4f}% {subset.quantile(0.99):>11.4f}%"
            )

    print("\nBy Moneyness Bucket (%):")
    print(f"  {'Bucket':<12} {'Count':>10} {'Median':>12} {'95th Pctl':>12} {'99th Pctl':>12}")
    print(f"  {'-' * 58}")
    for bucket in MONEYNESS_ORDER:
        subset = df[df["moneyness_bucket"] == bucket]["EEP_pct"]
        if len(subset) > 0:
            print(
                f"  {bucket:<12} {len(subset):>10,} {subset.median():>11.4f}% "
                f"{subset.quantile(0.95):>11.4f}% {subset.quantile(0.99):>11.4f}%"
            )

    pivot_median = df.pivot_table(
        values="EEP_pct",
        index="moneyness_bucket",
        columns="maturity_bucket",
        aggfunc="median",
    ).reindex(index=MONEYNESS_ORDER, columns=MATURITY_ORDER)
    print("\nMedian EEP (%) by Maturity x Moneyness:")
    print(pivot_median.round(4).to_string())

    pivot_95 = df.pivot_table(
        values="EEP_pct",
        index="moneyness_bucket",
        columns="maturity_bucket",
        aggfunc=lambda x: x.quantile(0.95),
    ).reindex(index=MONEYNESS_ORDER, columns=MATURITY_ORDER)
    print("\n95th Percentile EEP (%) by Maturity x Moneyness:")
    print(pivot_95.round(4).to_string())

    return pivot_median, pivot_95


def save_tables(
    nvda_daily: pd.DataFrame,
    nvdl_daily: pd.DataFrame,
    nvda_median: pd.DataFrame,
    nvda_95: pd.DataFrame,
    nvdl_median: pd.DataFrame,
    nvdl_95: pd.DataFrame,
) -> None:
    pd.concat([nvda_daily, nvdl_daily], ignore_index=True).to_csv(
        TABLES_DIR / "eep_daily_comparison.csv",
        index=False,
    )
    nvda_median.to_csv(TABLES_DIR / "eep_nvda_median.csv")
    nvda_95.to_csv(TABLES_DIR / "eep_nvda_p95.csv")
    nvdl_median.to_csv(TABLES_DIR / "eep_nvdl_median.csv")
    nvdl_95.to_csv(TABLES_DIR / "eep_nvdl_p95.csv")


def main() -> None:
    ensure_dirs()
    df_nvda, df_nvdl = load_default_inputs()
    df_nvda_informative, df_nvdl_informative, nvda_daily, nvdl_daily = prepare_eep_frames(df_nvda, df_nvdl)
    figure_path = save_plot(nvda_daily, nvdl_daily)

    print("\n" + "#" * 80)
    print("# MODEL-IMPLIED EARLY EXERCISE PREMIUM (EEP) ANALYSIS")
    print("# EEP_i = (Q^Am_i - Q~^Eu_i) / S_i x 100%")
    print("#" * 80)
    print("\nNote: For NVDA (ordinary underlying), only PUTS are informative because")
    print("      calls are unchanged by construction (EEP = 0 mechanically).")
    print("      For NVDL (LETF), both calls and puts are informative.")

    nvda_median, nvda_95 = compute_statistics(df_nvda_informative, "NVDA (Puts Only)")
    nvdl_median, nvdl_95 = compute_statistics(df_nvdl_informative, "NVDL (Calls & Puts)")

    print(f"\n{'=' * 80}")
    print("NVDL BREAKDOWN BY INSTRUMENT CLASS (% of Underlying)")
    print(f"{'=' * 80}")
    for inst_class, inst_name in [("C", "Calls"), ("P", "Puts")]:
        subset = df_nvdl_informative[df_nvdl_informative["instrument_class"] == inst_class]
        print(f"\n{inst_name}:")
        print(f"  Count:            {len(subset):,}")
        print(f"  Mean:             {subset['EEP_pct'].mean():.4f}%")
        print(f"  Median:           {subset['EEP_pct'].median():.4f}%")
        print(f"  95th Percentile:  {subset['EEP_pct'].quantile(0.95):.4f}%")

    print(f"\n{'=' * 80}")
    print("SUMMARY COMPARISON (% of Underlying)")
    print(f"{'=' * 80}")
    print(f"\n{'Metric':<25} {'NVDA (Puts)':<20} {'NVDL (All)':<20}")
    print(f"{'-' * 65}")
    print(f"{'Sample Size':<25} {len(df_nvda_informative):>15,} {len(df_nvdl_informative):>15,}")
    print(
        f"{'Mean EEP':<25} {df_nvda_informative['EEP_pct'].mean():>14.4f}% "
        f"{df_nvdl_informative['EEP_pct'].mean():>14.4f}%"
    )
    print(
        f"{'Median EEP':<25} {df_nvda_informative['EEP_pct'].median():>14.4f}% "
        f"{df_nvdl_informative['EEP_pct'].median():>14.4f}%"
    )
    print(
        f"{'95th Percentile EEP':<25} {df_nvda_informative['EEP_pct'].quantile(0.95):>14.4f}% "
        f"{df_nvdl_informative['EEP_pct'].quantile(0.95):>14.4f}%"
    )
    print(
        f"{'Max EEP':<25} {df_nvda_informative['EEP_pct'].max():>14.4f}% "
        f"{df_nvdl_informative['EEP_pct'].max():>14.4f}%"
    )

    save_tables(nvda_daily, nvdl_daily, nvda_median, nvda_95, nvdl_median, nvdl_95)
    print(f"\nSaved figure: {figure_path}")
    print(f"Saved tables to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
