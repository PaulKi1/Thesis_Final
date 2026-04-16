from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

from thesis.config import FIGURES_DIR, TABLES_DIR, ensure_dirs

# Stage 09 consumes the CSVs produced by stage 08.
FIGURE2_FILE = TABLES_DIR / "figure2_data.csv"
FIGURE4_FILE = TABLES_DIR / "figure4_data.csv"
OUTPUT_TABLE_DIR = TABLES_DIR / "moneyness_scaling_rmse"
OUTPUT_FIGURE_DIR = FIGURES_DIR / "moneyness_scaling_rmse"

N_GRID_POINTS = 50
MIN_GRID_WIDTH = 0.05  # Minimum overlap width to keep a cell


# ============================================================
# DATA LOADING
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [path for path in [FIGURE2_FILE, FIGURE4_FILE] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Stage 09 requires figure2_data.csv and figure4_data.csv from stage 08. "
            f"Missing: {missing_text}"
        )
    f2 = pd.read_csv(FIGURE2_FILE, parse_dates=['trade_date', 'expiration'])
    f4 = pd.read_csv(FIGURE4_FILE, parse_dates=['trade_date', 'expiration'])
    return f2, f4


# ============================================================
# STRICT COMMON-GRID COMPUTATION
# ============================================================

def compute_rmse_on_grid(
    x_a: np.ndarray, y_a: np.ndarray,
    x_b: np.ndarray, y_b: np.ndarray,
    grid: np.ndarray,
) -> dict:
    """
    Compute RMSE between two curves on a specified grid.
    Both curves are interpolated onto the grid.
    """
    # Sort curves
    sort_a = np.argsort(x_a)
    x_a, y_a = x_a[sort_a], y_a[sort_a]

    sort_b = np.argsort(x_b)
    x_b, y_b = x_b[sort_b], y_b[sort_b]

    # Interpolate both onto grid
    y_a_interp = np.interp(grid, x_a, y_a)
    y_b_interp = np.interp(grid, x_b, y_b)

    # Compute metrics
    errors = y_a_interp - y_b_interp

    return {
        'rmse': np.sqrt(np.mean(errors**2)),
        'mae': np.mean(np.abs(errors)),
        'bias': np.mean(errors),
    }


def compute_panel_distances_strict(f2: pd.DataFrame, f4: pd.DataFrame) -> pd.DataFrame:
    """
    Compute smile distances using STRICT common grid:
    Grid = intersection of pre-scaling overlap and post-scaling overlap.

    This ensures we compare exactly the same smile region before and after.
    """

    results = []

    # Track filtering
    stats = {
        'total_pairs': 0,
        'dropped_f2_small': 0,
        'dropped_ref_small': 0,
        'dropped_letf_small': 0,
        'dropped_no_intersection': 0,
        'dropped_intersection_narrow': 0,
        'kept': 0,
    }

    f2_keys = set(zip(f2['trade_date'], f2['expiration']))
    f4_keys = set(zip(f4['trade_date'], f4['expiration']))
    common_keys = f2_keys & f4_keys
    stats['total_pairs'] = len(common_keys)

    for trade_date, expiry in sorted(common_keys):
        # Get data
        f2_cell = f2[(f2['trade_date'] == trade_date) & (f2['expiration'] == expiry)]

        if len(f2_cell) < 5:
            stats['dropped_f2_small'] += 1
            continue

        f4_cell = f4[(f4['trade_date'] == trade_date) & (f4['expiration'] == expiry)]
        ref_product = f4_cell['reference_product'].iloc[0]

        ref_data = f4_cell[f4_cell['product'] == ref_product]
        letf_data = f4_cell[f4_cell['product'] != ref_product]

        if len(ref_data) < 5:
            stats['dropped_ref_small'] += 1
            continue

        if len(letf_data) < 5:
            stats['dropped_letf_small'] += 1
            continue

        # Extract x-ranges
        x_ref = ref_data['LM_ref_axis'].values
        x_letf_native = f2_cell['LM_plot'].values
        x_letf_scaled = letf_data['LM_ref_axis'].values

        # Compute overlaps
        overlap_pre_min = max(x_letf_native.min(), x_ref.min())
        overlap_pre_max = min(x_letf_native.max(), x_ref.max())

        overlap_post_min = max(x_letf_scaled.min(), x_ref.min())
        overlap_post_max = min(x_letf_scaled.max(), x_ref.max())

        # Compute INTERSECTION of overlaps
        intersection_min = max(overlap_pre_min, overlap_post_min)
        intersection_max = min(overlap_pre_max, overlap_post_max)
        intersection_width = intersection_max - intersection_min

        # Check if intersection exists
        if intersection_max <= intersection_min:
            stats['dropped_no_intersection'] += 1
            continue

        # Check if intersection is wide enough
        if intersection_width < MIN_GRID_WIDTH:
            stats['dropped_intersection_narrow'] += 1
            continue

        # Create THE SINGLE COMMON GRID
        common_grid = np.linspace(intersection_min, intersection_max, N_GRID_POINTS)

        # Extract y-values
        y_ref = ref_data['iv_norm'].values
        y_letf_native = f2_cell['iv_norm'].values
        y_letf_scaled = letf_data['iv_norm'].values

        # Compute PRE-scaling RMSE on common grid
        pre_metrics = compute_rmse_on_grid(
            x_letf_native, y_letf_native,
            x_ref, y_ref,
            common_grid
        )

        # Compute POST-scaling RMSE on SAME common grid
        post_metrics = compute_rmse_on_grid(
            x_letf_scaled, y_letf_scaled,
            x_ref, y_ref,
            common_grid
        )

        # Record
        tau = letf_data['tau'].iloc[0]
        tte = letf_data['time_to_expiration'].iloc[0]
        beta = letf_data['beta'].iloc[0]

        results.append({
            'trade_date': trade_date,
            'expiration': expiry,
            'tau': tau,
            'time_to_expiration': tte,
            'beta': beta,

            # Grid info
            'grid_min': intersection_min,
            'grid_max': intersection_max,
            'grid_width': intersection_width,
            'n_grid_points': N_GRID_POINTS,

            # Pre-scaling (on common grid)
            'rmse_pre': pre_metrics['rmse'],
            'mae_pre': pre_metrics['mae'],
            'bias_pre': pre_metrics['bias'],

            # Post-scaling (on SAME common grid)
            'rmse_post': post_metrics['rmse'],
            'mae_post': post_metrics['mae'],
            'bias_post': post_metrics['bias'],
        })

        stats['kept'] += 1

    df = pd.DataFrame(results)

    if len(df) > 0:
        # Compute improvement
        df['delta_rmse'] = df['rmse_post'] - df['rmse_pre']
        df['delta_mae'] = df['mae_post'] - df['mae_pre']
        df['rmse_improvement_pct'] = 100 * (df['rmse_pre'] - df['rmse_post']) / df['rmse_pre']
        df['mae_improvement_pct'] = 100 * (df['mae_pre'] - df['mae_post']) / df['mae_pre']
        df['scaling_helps'] = df['delta_rmse'] < 0

        # Maturity buckets
        bins = [0, 30, 60, 90, 180, 365, float('inf')]
        labels = ['<30d', '30-60d', '60-90d', '90-180d', '180-365d', '>365d']
        df['tau_bucket'] = pd.cut(df['time_to_expiration'], bins=bins, labels=labels)

    return df, stats


# ============================================================
# STATISTICAL TESTS
# ============================================================

def perform_statistical_tests(panel: pd.DataFrame) -> dict:
    delta_rmse = panel['delta_rmse'].dropna().values
    n = len(delta_rmse)

    if n < 10:
        return {
            'error': 'Insufficient observations',
            'n_observations': n,
            'n_improved': int(np.sum(delta_rmse < 0)),
            'n_worsened': int(np.sum(delta_rmse > 0)),
            'proportion_improved': float(np.mean(delta_rmse < 0)) if n else float('nan'),
            'mean_delta_rmse': float(np.mean(delta_rmse)) if n else float('nan'),
            'median_delta_rmse': float(np.median(delta_rmse)) if n else float('nan'),
            'std_delta_rmse': float(np.std(delta_rmse)) if n else float('nan'),
            'sign_test_pvalue': float('nan'),
            'wilcoxon_stat': float('nan'),
            'wilcoxon_pvalue': float('nan'),
        }

    n_negative = np.sum(delta_rmse < 0)
    n_positive = np.sum(delta_rmse > 0)
    n_nonzero = n_negative + n_positive
    if n_nonzero == 0:
        sign_test_pvalue = float('nan')
    else:
        sign_test_pvalue = scipy_stats.binomtest(
            n_negative, n_nonzero, p=0.5, alternative='greater'
        ).pvalue

    try:
        wilcoxon_stat, wilcoxon_pvalue = scipy_stats.wilcoxon(
            delta_rmse,
            alternative='less',
            zero_method='wilcox'
        )
    except ValueError:
        wilcoxon_stat, wilcoxon_pvalue = float('nan'), float('nan')

    return {
        'n_observations': n,
        'n_improved': n_negative,
        'n_worsened': n_positive,
        'proportion_improved': n_negative / n,
        'mean_delta_rmse': np.mean(delta_rmse),
        'median_delta_rmse': np.median(delta_rmse),
        'std_delta_rmse': np.std(delta_rmse),
        'sign_test_pvalue': sign_test_pvalue,
        'wilcoxon_stat': wilcoxon_stat,
        'wilcoxon_pvalue': wilcoxon_pvalue,
    }


def tests_by_maturity(panel: pd.DataFrame) -> pd.DataFrame:
    results = []
    for bucket in panel['tau_bucket'].dropna().unique():
        bucket_data = panel[panel['tau_bucket'] == bucket]
        if len(bucket_data) < 10:
            continue
        test_results = perform_statistical_tests(bucket_data)
        test_results['tau_bucket'] = bucket
        test_results['mean_rmse_pre'] = bucket_data['rmse_pre'].mean()
        test_results['mean_rmse_post'] = bucket_data['rmse_post'].mean()
        results.append(test_results)
    return pd.DataFrame(results)


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(panel: pd.DataFrame, output_folder: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Distribution of ΔRMSE
    ax = axes[0, 0]
    ax.hist(panel['delta_rmse'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.axvline(panel['delta_rmse'].median(), color='green', linestyle='-', linewidth=2,
               label=f'Median: {panel["delta_rmse"].median():.4f}')
    ax.set_xlabel('ΔRMSE (post − pre)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of RMSE Change\n(Negative = Scaling Helped)')
    ax.legend()

    # 2. Pre vs Post scatter
    ax = axes[0, 1]
    ax.scatter(panel['rmse_pre'], panel['rmse_post'], alpha=0.3, s=15)
    max_val = max(panel['rmse_pre'].max(), panel['rmse_post'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    ax.set_xlabel('RMSE Pre-Scaling')
    ax.set_ylabel('RMSE Post-Scaling')
    ax.set_title('RMSE Before vs After (Same Grid)')
    ax.set_aspect('equal', adjustable='box')

    # 3. By maturity - median improvement
    ax = axes[1, 0]
    tau_order = ['<30d', '30-60d', '60-90d', '90-180d', '180-365d', '>365d']
    tau_stats = panel.groupby('tau_bucket')['rmse_improvement_pct'].median()
    tau_stats = tau_stats.reindex([b for b in tau_order if b in tau_stats.index])

    bars = ax.bar(range(len(tau_stats)), tau_stats.values, edgecolor='black')
    ax.set_xticks(range(len(tau_stats)))
    ax.set_xticklabels(tau_stats.index, rotation=45)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_ylabel('Median RMSE Improvement (%)')
    ax.set_title('Scaling Effectiveness by Maturity')
    for bar, val in zip(bars, tau_stats.values):
        bar.set_color('green' if val > 0 else 'red')

    # 4. Success rate by maturity
    ax = axes[1, 1]
    success = panel.groupby('tau_bucket')['scaling_helps'].mean() * 100
    success = success.reindex([b for b in tau_order if b in success.index])
    ax.bar(range(len(success)), success.values, color='steelblue', edgecolor='black')
    ax.axhline(50, color='red', linestyle='--')
    ax.set_xticks(range(len(success)))
    ax.set_xticklabels(success.index, rotation=45)
    ax.set_ylabel('% Cells Where Scaling Helps')
    ax.set_title('Success Rate by Maturity')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_folder / 'scaling_test_strict_grid.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN
# ============================================================

def run_strict_scaling_tests():
    print("=" * 70)
    print("SECTION 3.5: MONEYNESS SCALING TEST")
    print("STRICT VERSION: Same grid for pre and post")
    print("=" * 70)

    ensure_dirs()
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    print("\n[1] Loading data...")
    f2, f4 = load_data()

    # Compute
    print("\n[2] Computing distances on STRICT common grid...")
    print(f"    Grid = intersection of pre-overlap and post-overlap")
    print(f"    Minimum grid width: {MIN_GRID_WIDTH}")

    panel, stats = compute_panel_distances_strict(f2, f4)
    if panel.empty:
        raise RuntimeError(
            "Stage 09 found no overlapping smile cells after filtering. "
            "Check the stage 08 outputs and overlap assumptions."
        )

    print(f"\n    FILTERING SUMMARY:")
    print(f"    {'─' * 50}")
    print(f"    Total (date, expiry) pairs:      {stats['total_pairs']:,}")
    print(f"    Dropped (f2 < 5 points):         {stats['dropped_f2_small']}")
    print(f"    Dropped (ref < 5 points):        {stats['dropped_ref_small']}")
    print(f"    Dropped (letf < 5 points):       {stats['dropped_letf_small']}")
    print(f"    Dropped (no intersection):       {stats['dropped_no_intersection']}")
    print(f"    Dropped (intersection < {MIN_GRID_WIDTH}):    {stats['dropped_intersection_narrow']}")
    print(f"    Kept for analysis:               {stats['kept']:,}")

    # Summary stats
    print(f"\n[3] RESULTS")
    print(f"    {'─' * 50}")
    print(f"    Mean RMSE (pre):            {panel['rmse_pre'].mean():.6f}")
    print(f"    Mean RMSE (post):           {panel['rmse_post'].mean():.6f}")
    print(f"    Median RMSE (pre):          {panel['rmse_pre'].median():.6f}")
    print(f"    Median RMSE (post):         {panel['rmse_post'].median():.6f}")
    print(f"    ")
    print(f"    Mean ΔRMSE:                 {panel['delta_rmse'].mean():+.6f}")
    print(f"    Median ΔRMSE:               {panel['delta_rmse'].median():+.6f}")
    print(f"    Mean improvement (%):       {panel['rmse_improvement_pct'].mean():+.1f}%")
    print(f"    Median improvement (%):     {panel['rmse_improvement_pct'].median():+.1f}%")
    print(f"    ")
    print(f"    Cells improved:             {panel['scaling_helps'].sum():,} ({100*panel['scaling_helps'].mean():.1f}%)")
    print(f"    Cells worsened:             {(~panel['scaling_helps']).sum():,}")

    # Statistical tests
    print(f"\n[4] STATISTICAL TESTS")
    tests = perform_statistical_tests(panel)
    print(f"    {'─' * 50}")
    print(f"    Sign test p-value:          {tests['sign_test_pvalue']:.2e}")
    print(f"    Wilcoxon p-value:           {tests['wilcoxon_pvalue']:.2e}")

    # By maturity
    print(f"\n[5] BY MATURITY")
    mat_tests = tests_by_maturity(panel)
    print(f"    {'─' * 70}")
    print(f"    {'Bucket':<12} {'N':>6} {'RMSE Pre':>10} {'RMSE Post':>10} {'Improved':>10} {'p-value':>12}")
    print(f"    {'─' * 70}")
    for _, row in mat_tests.sort_values('tau_bucket').iterrows():
        print(f"    {row['tau_bucket']:<12} {row['n_observations']:>6} "
              f"{row['mean_rmse_pre']:>10.4f} {row['mean_rmse_post']:>10.4f} "
              f"{100*row['proportion_improved']:>9.1f}% {row['wilcoxon_pvalue']:>12.2e}")

    # Plots
    print(f"\n[6] Generating plots...")
    plot_results(panel, OUTPUT_FIGURE_DIR)

    # Save
    print(f"\n[7] Saving results...")
    panel.to_csv(OUTPUT_TABLE_DIR / 'scaling_test_panel_strict.csv', index=False)
    mat_tests.to_csv(OUTPUT_TABLE_DIR / 'maturity_tests_strict.csv', index=False)

    # Show grid width distribution
    print(f"\n[8] GRID WIDTH DISTRIBUTION")
    print(f"    Min:    {panel['grid_width'].min():.4f}")
    print(f"    Median: {panel['grid_width'].median():.4f}")
    print(f"    Max:    {panel['grid_width'].max():.4f}")

    print(f"\n    Results saved to: {OUTPUT_TABLE_DIR} and {OUTPUT_FIGURE_DIR}")
    print("=" * 70)

    return panel, stats, tests


if __name__ == "__main__":
    panel, stats, tests = run_strict_scaling_tests()
