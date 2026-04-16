"""
Run EEP diagnostics from deamericanized workbooks. Needs both the underlying and LETF
"""


from thesis.analysis.eep_diagnostics import main
from thesis.config import ensure_dirs


if __name__ == "__main__":
    ensure_dirs()
    main()
