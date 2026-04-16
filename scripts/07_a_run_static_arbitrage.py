# scripts/06_run_static_arbitrage_pre_fengler.py
from thesis.analysis.static_arbitrage import main
from thesis.config import TABLES_DIR, ensure_dirs

if __name__ == "__main__":
    ensure_dirs()
    summaries = main(stage="pre_fengler")
    for name, summary in summaries.items():
        out_path = TABLES_DIR / f"{name}_static_arbitrage_pre_fengler.csv"
        summary.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")