# Thesis Paul Kielhorn

Below is a quick running code if replication is desired. All of this is the code used for my Thesis: "Empirical Analysis of Implied Volatility Scaling in Single-Stock Leveraged ETF Options"
As a summary, **scripts** contains the simple runners. The codes are in **src** / can be accessed easily from the imported packages for each runner in **scripts**. 
The data has been obtained from databento. Replication therefore requires an API key (cost for replication are less than $2. Databento gives out $125 as free credit initially not sure if this is still done)

Note: Chapters 4-5 of the main paper (Leung and Sircar 2015) were also used for analysis but are not in this repo as they were not included in the thesis. Generative AI has been used for the development of this codebase

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows cmd
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

If `pip install -e .` fails on Windows with a `distutils` / `setuptools` assertion error, use:

```powershell
$env:SETUPTOOLS_USE_DISTUTILS = "stdlib"
pip install -e .
```

### 3. Configure `.env`

Create `.env` in the project root and add at least:

```env
DATABENTO_API_KEY=your_key_here
```

Optional:

```env
DATA_ROOT=path/to/custom/data/root
```

All paths are resolved through `src/thesis/config.py`.

### 4. Run the pipeline

Run the numbered scripts in order:

```bash
python scripts/01_download_databento.py
python scripts/02_download_sofr_curves.py
python scripts/03_clean_data.py
python scripts/04_run_deamericanization.py
python scripts/05_run_eep_diagnostics.py
python scripts/06_run_fengler.py
python scripts/07_run_static_arbitrage.py
python scripts/08_run_moneyness_scaling.py
python scripts/09_run_rmse_moneyness_scaling.py
```

## Pipeline Stages

### 01. Databento download

- Reads `DATABENTO_API_KEY` from `.env`
- Downloads raw OPRA / underlying data
- Writes to `data/raw/databento/`

### 02. Eris SOFR download

- Downloads SOFR discount-factor files and builds a lookup table
- Writes to `data/raw/eris_sofr/`

### 03. Cleaning

- Reads Databento Excel outputs plus the SOFR lookup table
- Writes cleaned workbooks to `data/interim/`

### 04. De-Americanization

- Reads cleaned workbooks from `data/interim/`
- Writes de-Americanized workbooks to `data/processed/deamericanized/`

### 05. EEP diagnostics

- Reads de-Americanized workbooks
- Writes figures to `data/outputs/figures/`
- Writes tables to `data/outputs/tables/`

### 06. Fengler repair

- Reads de-Americanized workbooks
- Writes per-date repaired workbooks and combined quote-level outputs to `data/processed/fengler/`

### 07. Static arbitrage report

- Reads Fengler combined quote-level workbooks
- Writes summary CSVs to `data/outputs/tables/`

### 08. Moneyness scaling

- Reads Fengler combined quote-level workbooks
- Writes plots to `data/outputs/figures/moneyness_scaling/`
- Writes `figure2_data.csv` and `figure4_data.csv` to `data/outputs/tables/`

### 09. RMSE moneyness scaling test

- Reads `figure2_data.csv` and `figure4_data.csv` from stage 08
- Fails clearly if those files are missing
- Writes outputs to:
  `data/outputs/tables/moneyness_scaling_rmse/`
  `data/outputs/figures/moneyness_scaling_rmse/`

## Project Layout

```text
.
|-- pyproject.toml
|-- README.md
|-- .env
|-- src/
|   `-- thesis/
|       |-- __init__.py
|       |-- config.py
|       |-- analysis/
|       |-- deamericanization/
|       |-- download/
|       |-- fengler/
|       `-- preprocessing/
|-- scripts/
|   |-- 01_download_databento.py
|   |-- 02_download_sofr_curves.py
|   |-- 03_clean_data.py
|   |-- 04_run_deamericanization.py
|   |-- 05_run_eep_diagnostics.py
|   |-- 06_run_fengler.py
|   |-- 07_run_static_arbitrage.py
|   |-- 08_run_moneyness_scaling.py
|   `-- 09_run_rmse_moneyness_scaling.py
`-- data/
    |-- raw/
    |-- interim/
    |-- processed/
    `-- outputs/
```

## Notes

- The scripts are intended to be thin entry points. Reusable logic lives under `src/thesis/`.
- Do not hardcode paths in scripts or modules. Import them from `thesis.config`.
- Stage 08 produces the CSV inputs required by stage 09.
- The `data/` directory is generated data and should be treated as pipeline output, not source code.
