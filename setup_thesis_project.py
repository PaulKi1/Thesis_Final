"""
Thesis project scaffolding script.

Run this from the ROOT of an empty (or mostly empty) project folder. It will:
  1. Create the full folder structure for a multi-stage thesis pipeline
  2. Add __init__.py files to make src/thesis a proper Python package
  3. Add .gitkeep files so empty data directories get tracked by git
  4. Generate starter versions of:
       - config.py          (central path / env handling)
       - .env.example       (template for environment variables)
       - .env               (your local config, gitignored)
       - .gitignore
       - README.md
       - requirements.txt
       - pyproject.toml
  5. Optionally move existing .py files from the project root into the
     appropriate subdirectories (prompts before doing anything)

Usage:
    python setup_thesis_project.py
    python setup_thesis_project.py --no-move   # skip the file-moving prompt
    python setup_thesis_project.py --force     # overwrite existing starter files

Safe to re-run: it will not overwrite existing files unless --force is passed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIRECTORIES = [
    "src/thesis/download",
    "src/thesis/preprocessing",
    "src/thesis/deamericanization",
    "src/thesis/fengler",
    "src/thesis/analysis",
    "scripts",
    "data/raw/databento",
    "data/raw/eris_sofr",
    "data/interim",
    "data/processed/deamericanized",
    "data/processed/fengler",
    "data/outputs/figures",
    "data/outputs/tables",
    "notebooks",
    "tests",
]

# Folders that should be importable Python packages
PACKAGE_DIRS = [
    "src/thesis",
    "src/thesis/download",
    "src/thesis/preprocessing",
    "src/thesis/deamericanization",
    "src/thesis/fengler",
    "src/thesis/analysis",
    "tests",
]

# Folders that should stay tracked by git even when empty
GITKEEP_DIRS = [
    "data/raw/databento",
    "data/raw/eris_sofr",
    "data/interim",
    "data/processed/deamericanized",
    "data/processed/fengler",
    "data/outputs/figures",
    "data/outputs/tables",
    "notebooks",
]

# Mapping from existing filenames -> target location inside src/thesis
# If your filenames differ, adjust this dict.
FILE_MOVES = {
    "databento_download.py": "src/thesis/download/databento.py",
    "download_eris_sofr_discount_curves.py": "src/thesis/download/eris_sofr.py",
    "cleaning.py": "src/thesis/preprocessing/cleaning.py",
    "deamericanization.py": "src/thesis/deamericanization/pipeline.py",
    "runner_deamericanization.py": "scripts/04_run_deamericanization.py",
    "fengler_repair.py": "src/thesis/fengler/repair.py",
    "Runner_fengler.py": "scripts/06_run_fengler.py",
    "eep_diagnostics.py": "src/thesis/analysis/eep_diagnostics.py",
    "moneyness_scaling_rmse.py": "src/thesis/analysis/moneyness_scaling_rmse.py",
    "static_arbitrage_report.py": "src/thesis/analysis/static_arbitrage.py",
    "minimal_single_stock_letf_plots.py": "src/thesis/analysis/moneyness_scaling.py",
}

# ---------------------------------------------------------------------------
# File contents
# ---------------------------------------------------------------------------

CONFIG_PY = '''"""
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
'''

ENV_EXAMPLE = """# Copy this file to .env and fill in your values.
# The .env file is gitignored and should never be committed.

# Databento API key (https://databento.com)
DATABENTO_API_KEY=your-databento-api-key-here

# Optional: override the data directory (e.g., to store data on a different drive)
# DATA_ROOT=D:/thesis_data
"""

ENV_FILE = """# Local environment configuration. DO NOT COMMIT THIS FILE.
# Fill in your actual values below.

DATABENTO_API_KEY=replace-me-with-your-actual-key

# DATA_ROOT=D:/thesis_data
"""

GITIGNORE = """# -----------------------------------------------------------------------------
# Secrets
# -----------------------------------------------------------------------------
.env
*.key
*.pem

# -----------------------------------------------------------------------------
# Data (large files, downloads, outputs)
# -----------------------------------------------------------------------------
data/raw/**/*
data/interim/**/*
data/processed/**/*
data/outputs/**/*
!data/**/.gitkeep

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.venv/
venv/
env/
ENV/

# -----------------------------------------------------------------------------
# IDEs
# -----------------------------------------------------------------------------
.idea/
.vscode/
*.swp
*.swo

# -----------------------------------------------------------------------------
# Jupyter
# -----------------------------------------------------------------------------
.ipynb_checkpoints/
*.ipynb_checkpoints

# -----------------------------------------------------------------------------
# OS
# -----------------------------------------------------------------------------
.DS_Store
Thumbs.db
desktop.ini

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
.pytest_cache/
.coverage
htmlcov/
.tox/
"""

README = """# De-Americanization Thesis Project

Pipeline for processing OPRA option data through de-Americanization, Fengler
surface repair, and downstream analysis (EEP diagnostics, moneyness scaling,
static arbitrage checks).

## Quick start

### 1. Set up environment

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the thesis package in editable mode so imports work cleanly
pip install -e .
```

### 2. Configure secrets

```bash
# Copy the template and fill in your API keys
cp .env.example .env
# Then edit .env with your Databento API key
```

### 3. Run the pipeline

Scripts are numbered in execution order:

```bash
python scripts/01_download_databento.py
python scripts/02_download_sofr_curves.py
python scripts/03_clean_data.py
python scripts/04_run_deamericanization.py
python scripts/06_run_fengler.py
python scripts/06_run_analysis.py
```

## Project structure

```
.
├── src/thesis/              # Importable package with reusable logic
│   ├── config.py            # Central path + env handling
│   ├── download/            # Data acquisition (Databento, Eris SOFR)
│   ├── preprocessing/       # Cleaning
│   ├── deamericanization/   # De-Americanization pipeline
│   ├── fengler/             # Fengler surface repair
│   └── analysis/            # Downstream analysis and plots
├── scripts/                 # Thin entry-point runners
├── data/
│   ├── raw/                 # Downloaded source data (gitignored)
│   ├── interim/             # Cleaned but not yet processed (gitignored)
│   ├── processed/           # Deamericanized + Fengler output (gitignored)
│   └── outputs/             # Figures, tables, final artifacts (gitignored)
├── notebooks/               # Exploratory Jupyter notebooks
└── tests/                   # Unit tests
```

## Pipeline stages

1. **Download** — Pull OPRA option data from Databento and SOFR discount
   curves from Eris Futures.
2. **Preprocessing** — Clean raw data into a consistent schema.
3. **De-Americanization** — Convert American option prices into pseudo-European
   prices using the method in `deamericanization/pipeline.py`.
4. **Fengler repair** — Apply Fengler's arbitrage-free surface repair to the
   de-Americanized prices.
5. **Analysis** — Compute EEP diagnostics, moneyness scaling tests, static
   arbitrage reports, and produce plots.

## Paths and configuration

All paths are defined in `src/thesis/config.py` and driven by environment
variables in `.env`. To change where data lives, set `DATA_ROOT` in `.env`.
Never hardcode paths inside scripts.

## Notes

- This project uses editable installs (`pip install -e .`), which means any
  edits to `src/thesis/` take effect immediately without reinstalling.
- The `data/` directory is gitignored — you will need to regenerate downloads
  on a fresh clone.
"""

REQUIREMENTS = """pandas>=2.0
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
openpyxl>=3.1
python-dotenv>=1.0
databento>=0.30
"""

PYPROJECT = """[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "thesis"
version = "0.1.0"
description = "De-Americanization thesis pipeline"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "openpyxl>=3.1",
    "python-dotenv>=1.0",
    "databento>=0.30",
]

[tool.setuptools.packages.find]
where = ["src"]
"""

# Files to generate (destination -> content, overwrite_warning)
GENERATED_FILES = {
    "src/thesis/config.py": CONFIG_PY,
    ".env.example": ENV_EXAMPLE,
    ".env": ENV_FILE,
    ".gitignore": GITIGNORE,
    "README.md": README,
    "requirements.txt": REQUIREMENTS,
    "pyproject.toml": PYPROJECT,
}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def create_directories(root: Path) -> None:
    print("\n[1/4] Creating directories...")
    for rel in DIRECTORIES:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ok  {rel}")


def create_init_files(root: Path) -> None:
    print("\n[2/4] Creating __init__.py files...")
    for rel in PACKAGE_DIRS:
        path = root / rel / "__init__.py"
        if path.exists():
            print(f"  --  {rel}/__init__.py (already exists)")
            continue
        path.write_text("", encoding="utf-8")
        print(f"  ok  {rel}/__init__.py")


def create_gitkeep_files(root: Path) -> None:
    print("\n[3/4] Creating .gitkeep files for empty data dirs...")
    for rel in GITKEEP_DIRS:
        path = root / rel / ".gitkeep"
        if path.exists():
            print(f"  --  {rel}/.gitkeep (already exists)")
            continue
        path.write_text("", encoding="utf-8")
        print(f"  ok  {rel}/.gitkeep")


def create_starter_files(root: Path, force: bool) -> None:
    print("\n[4/4] Creating starter config files...")
    for rel, content in GENERATED_FILES.items():
        path = root / rel
        if path.exists() and not force:
            print(f"  --  {rel} (already exists, use --force to overwrite)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"  ok  {rel}")


def move_existing_files(root: Path) -> None:
    """Offer to move any existing .py files from the root into the package."""
    candidates = {
        src: dst for src, dst in FILE_MOVES.items()
        if (root / src).exists()
    }
    if not candidates:
        print("\n[optional] No existing files to move.")
        return

    print("\n[optional] Found existing files at the project root:")
    for src, dst in candidates.items():
        print(f"  {src}  ->  {dst}")

    response = input("\nMove these files into the package? [y/N] ").strip().lower()
    if response != "y":
        print("  Skipped.")
        return

    for src, dst in candidates.items():
        src_path = root / src
        dst_path = root / dst
        if dst_path.exists():
            print(f"  --  {dst} already exists, skipping {src}")
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        print(f"  ok  moved {src} -> {dst}")


def print_next_steps(root: Path) -> None:
    print("\n" + "=" * 70)
    print("Setup complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Edit .env and add your Databento API key")
    print("  2. Create a virtualenv:       python -m venv .venv")
    print("  3. Activate it (Windows):     .venv\\Scripts\\activate")
    print("     Activate it (Mac/Linux):   source .venv/bin/activate")
    print("  4. Install dependencies:      pip install -r requirements.txt")
    print("  5. Install the package:       pip install -e .")
    print("  6. Replace hardcoded paths in your scripts with imports from")
    print("     thesis.config (see README.md for examples)")
    print("\nIn PyCharm:")
    print("  - Right-click 'src' -> Mark Directory as -> Sources Root")
    print("  - Right-click 'data' -> Mark Directory as -> Excluded")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold the thesis project structure.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (defaults to current working directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing starter files (config.py, .gitignore, etc).",
    )
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="Skip the prompt to move existing .py files into the package.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    print(f"Setting up thesis project in: {root}")

    if not root.exists():
        print(f"ERROR: Project root does not exist: {root}")
        return 1

    create_directories(root)
    create_init_files(root)
    create_gitkeep_files(root)
    create_starter_files(root, force=args.force)

    if not args.no_move:
        move_existing_files(root)

    print_next_steps(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())