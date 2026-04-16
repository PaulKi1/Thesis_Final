"""Clean raw Databento data into the interim format for downstream stages.

Reads from data/raw/databento/
Writes to data/interim/
"""
from thesis.config import DATABENTO_RAW_DIR, INTERIM_DIR, ensure_dirs
from thesis.preprocessing.cleaning import main

if __name__ == "__main__":
    ensure_dirs()
    main()