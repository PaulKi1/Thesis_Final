"""Run RMSE diagnostics for the moneyness scaling outputs."""

from thesis.analysis.moneyness_scaling_rmse import run_strict_scaling_tests
from thesis.config import ensure_dirs


if __name__ == "__main__":
    ensure_dirs()
    run_strict_scaling_tests()
