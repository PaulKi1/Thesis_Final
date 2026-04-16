"""Run static arbitrage on the Fengler-repaired surface (AFTER repair).

Reads from data/processed/fengler/ (sheet: "Sheet1")
Writes to data/outputs/tables/ with _post_fengler suffix.
"""
from thesis.analysis.static_arbitrage import main
from thesis.config import TABLES_DIR, ensure_dirs


if __name__ == "__main__":
    ensure_dirs()
    summaries = main(stage="post_fengler")
    for name, summary in summaries.items():
        out_path = TABLES_DIR / f"{name}_static_arbitrage_post_fengler.csv"
        summary.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")