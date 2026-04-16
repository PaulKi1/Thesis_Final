"""
Central configuration for the thesis project.

All paths and environment variables are defined here. Import from this module
instead of hardcoding paths anywhere else in the project.

Usage:
    from thesis.config import DEAMERICANIZED_DIR, DATABENTO_API_KEY, ensure_dirs

    ensure_dirs()  # call once at the start of a script
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = two levels up from this file (src/thesis/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env file at project root
load_dotenv(PROJECT_ROOT / ".env")

# -----------------------------------------------------------------------------
# Data directories
# -----------------------------------------------------------------------------
# DATA_ROOT can be overridden in .env if you want to store data on another drive.
DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data"))

RAW_DIR = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"
PROCESSED_DIR = DATA_ROOT / "processed"
OUTPUTS_DIR = DATA_ROOT / "outputs"

# Stage-specific subdirectories
DATABENTO_RAW_DIR = RAW_DIR / "databento"
ERIS_SOFR_DIR = RAW_DIR / "eris_sofr"
DEAMERICANIZED_DIR = PROCESSED_DIR / "deamericanized"
FENGLER_DIR = PROCESSED_DIR / "fengler"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# -----------------------------------------------------------------------------
# API keys and secrets (loaded from .env, never hardcoded)
# -----------------------------------------------------------------------------
DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create all data directories if they don't already exist."""
    dirs = [
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        OUTPUTS_DIR,
        DATABENTO_RAW_DIR,
        ERIS_SOFR_DIR,
        DEAMERICANIZED_DIR,
        FENGLER_DIR,
        FIGURES_DIR,
        TABLES_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def require_api_key(name: str, value: str | None) -> str:
    """Raise a clear error if a required API key is missing."""
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to your .env file at {PROJECT_ROOT / '.env'}"
        )
    return value
