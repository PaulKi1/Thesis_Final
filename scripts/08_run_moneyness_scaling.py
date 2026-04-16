"""Run moneyness scaling analysis from Fengler quote-level workbooks."""

from thesis.analysis.moneyness_scaling import main
from thesis.config import ensure_dirs


if __name__ == "__main__":
    ensure_dirs()
    main()
