"""Download option data from Databento.

Outputs go to data/raw/databento/ (configured in thesis.config).
"""
from thesis.config import DATABENTO_RAW_DIR, DATABENTO_API_KEY, ensure_dirs, require_api_key
from thesis.download.databento import main

if __name__ == "__main__":
    ensure_dirs()
    require_api_key("DATABENTO_API_KEY", DATABENTO_API_KEY)
    main()