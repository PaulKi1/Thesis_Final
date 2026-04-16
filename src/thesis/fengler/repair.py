from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.optimize import LinearConstraint, minimize, brentq
from scipy.stats import norm

# ---------------------------
# Configuration and metadata
# ---------------------------


@dataclass
class FenglerConfig:
    sheet_name: str = "deamericanized"
    n_kappa_nodes: int = 30
    coverage_target: float = 0.80
    narrow_slice_ratio: float = 0.35
    min_quotes_inside_band: int = 4
    pre_smoother_smoothing: float = 0.5
    pre_smoother_neighbors: Optional[int] = None
    fixed_lambda: float = 2
    max_calendar_refinement_rounds: int = 2
    dense_check_points: int = 201
    price_bound_tolerance: float = 1e-8
    black_inversion_tolerance: float = 1e-8
    total_variance_floor: float = 1e-12
    total_variance_upper: float = 25.0
    qp_gtol: float = 1e-8
    qp_xtol: float = 1e-8
    qp_barrier_tol: float = 1e-8
    qp_maxiter: int = 1000
    calendar_tolerance: float = 1e-8
    monotonicity_tolerance: float = 1e-8
    convexity_tolerance: float = 1e-8
    verbose: bool = False


@dataclass
class SliceMetadata:
    slice_id: str
    trade_date: pd.Timestamp
    expiration: pd.Timestamp
    tau: float
    forward_price: float
    discount_factor: float
    pv_div_spot: float
    observed_kappa_min: float
    observed_kappa_max: float
    observed_width: float
    n_quotes: int


@dataclass
class SplineSlice:
    strikes: np.ndarray
    node_values: np.ndarray
    gammas: np.ndarray  # interior second derivatives only, length n-2
    metadata: SliceMetadata

    @property
    def full_gammas(self) -> np.ndarray:
        return np.concatenate(([0.0], self.gammas, [0.0]))

    def evaluate(self, strikes: np.ndarray | float) -> np.ndarray | float:
        x = np.asarray(strikes, dtype=float)
        y = _evaluate_natural_cubic_spline(self.strikes, self.node_values, self.gammas, x)
        if np.isscalar(strikes):
            return float(y)
        return y


# ---------------------------
# Public API
# ---------------------------


def repair_fengler_surface_single_date(
    excel_path: str,
    trade_date: str | pd.Timestamp,
    config: Optional[FenglerConfig] = None,
) -> Dict[str, Any]:
    """
    Run the Fengler-style repair for exactly one trade date from the workbook.

    Returns a dictionary with:
        - quote_level_output
        - grid_level_output
        - effective_call_audit_table
        - slice_metadata_table
        - diagnostics
        - repaired_surface
        - config
    """
    cfg = config or FenglerConfig()
    trade_date = pd.Timestamp(trade_date)

    raw = pd.read_excel(excel_path, sheet_name=cfg.sheet_name)
    _assert_required_columns(raw)

    day = raw.loc[pd.to_datetime(raw["trade_date"]) == trade_date].copy()
    if day.empty:
        raise ValueError(f"No rows found for trade_date={trade_date.date()} in {excel_path!r}.")

    day = _validate_day_input(day)

    effective_calls, effective_call_audit = _build_effective_call_set(day)
    raw_tv_quotes, effective_call_audit = _compute_raw_total_variance(
        effective_calls,
        effective_call_audit,
        cfg,
    )

    slice_metadata = _build_slice_metadata(raw_tv_quotes)
    core_band = _choose_core_kappa_band(raw_tv_quotes, slice_metadata, cfg)
    grid = _build_common_grid(raw_tv_quotes, core_band, cfg)

    presmoother = _fit_thin_plate_tv_presmoother(raw_tv_quotes, cfg)
    repaired_surface = _repair_surface_backward(
        presmoother=presmoother,
        grid=grid,
        slice_metadata=slice_metadata,
        cfg=cfg,
    )
    # WAVINESS ISSUE
    waviness_issues = validate_repaired_smiles(repaired_surface, cfg, max_extrema=1)
    print(waviness_issues)

    grid_level_output = _build_grid_level_output(repaired_surface, grid, slice_metadata, cfg)
    quote_level_output = _evaluate_at_original_quotes(
        repaired_surface=repaired_surface,
        original_day=day,
        effective_call_audit=effective_call_audit,
        core_band=core_band,
        cfg=cfg,
    )
    diagnostics = _run_final_diagnostics(
        repaired_surface=repaired_surface,
        grid=grid,
        quote_level_output=quote_level_output,
        effective_call_audit=effective_call_audit,
        cfg=cfg,
    )

    return {
        "quote_level_output": quote_level_output,
        "grid_level_output": grid_level_output,
        "effective_call_audit_table": effective_call_audit,
        "slice_metadata_table": pd.DataFrame([asdict(x) for x in slice_metadata.values()]),
        "diagnostics": diagnostics,
        "repaired_surface": repaired_surface,
        "waviness_issues": waviness_issues,
        "config": cfg,
    }


# ---------------------------
# Step 1: validation / input
# ---------------------------


_REQUIRED_COLUMNS = {
    "trade_date",
    "expiration",
    "instrument_class",
    "strike_price",
    "mid_px_eu",
    "discount_factor",
    "forward_price",
    "Moneyness",
    "tau",
}


def _assert_required_columns(df: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Workbook is missing required columns: {missing}")


def _validate_day_input(day: pd.DataFrame) -> pd.DataFrame:
    day = day.copy()
    day["trade_date"] = pd.to_datetime(day["trade_date"])
    day["expiration"] = pd.to_datetime(day["expiration"])

    if day["trade_date"].nunique() != 1:
        raise ValueError("Input contains more than one trade_date. This function handles one trade_date only.")

    for col in ["strike_price", "mid_px_eu", "discount_factor", "forward_price", "Moneyness", "tau"]:
        day[col] = pd.to_numeric(day[col], errors="coerce")

    if (day["tau"] <= 0).any():
        raise ValueError("All tau values must be strictly positive.")
    if (day["discount_factor"] <= 0).any():
        raise ValueError("All discount factors must be strictly positive.")
    if (day["forward_price"] <= 0).any():
        raise ValueError("All forward prices must be strictly positive.")
    if (day["Moneyness"] <= 0).any():
        raise ValueError("All moneyness values must be strictly positive.")

    day["slice_id"] = day["trade_date"].dt.strftime("%Y-%m-%d") + "__" + day["expiration"].dt.strftime("%Y-%m-%d")
    day = day.drop_duplicates()
    day = day.sort_values(["expiration", "strike_price", "instrument_class"]).reset_index(drop=True)
    return day


# ---------------------------
# Step 2: effective calls and audit trail
# ---------------------------


def _build_effective_call_set(day: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    grouped = day.groupby(["slice_id", "strike_price"], sort=True, dropna=False)
    for (_, strike), grp in grouped:
        grp = grp.sort_values("instrument_class")

        first = grp.iloc[0]
        slice_id = first["slice_id"]
        trade_date = first["trade_date"]
        expiration = first["expiration"]
        tau = float(first["tau"])
        df = float(first["discount_factor"])
        forward = float(first["forward_price"])
        kappa = float(first["Moneyness"])
        pv_div_spot = df * forward

        call_row = grp.loc[grp["instrument_class"].astype(str).str.upper() == "C"]
        put_row = grp.loc[grp["instrument_class"].astype(str).str.upper() == "P"]

        raw_call_price = float(call_row.iloc[0]["mid_px_eu"]) if not call_row.empty else np.nan
        raw_put_price = float(put_row.iloc[0]["mid_px_eu"]) if not put_row.empty else np.nan
        put_converted_call = raw_put_price + df * (forward - strike) if not np.isnan(raw_put_price) else np.nan
        parity_gap = raw_call_price - put_converted_call if (not np.isnan(raw_call_price) and not np.isnan(put_converted_call)) else np.nan

        preferred_side = "P" if kappa < 1.0 else "C"

        if preferred_side == "C" and not np.isnan(raw_call_price):
            chosen_side = "C"
            discarded_side = "P" if not np.isnan(raw_put_price) else ""
            effective_call_preclean = raw_call_price
        elif preferred_side == "P" and not np.isnan(raw_put_price):
            chosen_side = "P"
            discarded_side = "C" if not np.isnan(raw_call_price) else ""
            effective_call_preclean = put_converted_call
        elif not np.isnan(raw_call_price):
            chosen_side = "C_fallback"
            discarded_side = "P" if not np.isnan(raw_put_price) else ""
            effective_call_preclean = raw_call_price
        elif not np.isnan(raw_put_price):
            chosen_side = "P_fallback"
            discarded_side = "C"
            effective_call_preclean = put_converted_call
        else:
            continue

        rows.append(
            {
                "slice_id": slice_id,
                "trade_date": trade_date,
                "expiration": expiration,
                "tau": tau,
                "strike_price": float(strike),
                "kappa": kappa,
                "discount_factor": df,
                "forward_price": forward,
                "pv_div_spot": pv_div_spot,
                "source_side_chosen": chosen_side,
                "discarded_side": discarded_side,
                "raw_call_price_eu": raw_call_price,
                "raw_put_price_eu": raw_put_price,
                "put_converted_call_eu": put_converted_call,
                "parity_gap": parity_gap,
                "effective_call_preclean": effective_call_preclean,
            }
        )

    audit = pd.DataFrame(rows).sort_values(["expiration", "strike_price"]).reset_index(drop=True)
    effective = audit.copy()
    return effective, audit.copy()


# ---------------------------
# Step 3: raw total variance
# ---------------------------


def _compute_raw_total_variance(
    effective_calls: pd.DataFrame,
    audit: pd.DataFrame,
    cfg: FenglerConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    effective_calls = effective_calls.copy()
    audit = audit.copy()

    cleaned_values = []
    iv_values = []
    tv_values = []
    clamp_flags = []
    inv_status = []
    lower_bounds = []
    upper_bounds = []
    f_obs_list = []

    for row in effective_calls.itertuples(index=False):
        strike = float(row.strike_price)
        df = float(row.discount_factor)
        forward = float(row.forward_price)
        tau = float(row.tau)
        price = float(row.effective_call_preclean)

        lower = max(df * (forward - strike), 0.0)
        upper = df * forward
        lower_bounds.append(lower)
        upper_bounds.append(upper)

        cleaned = price
        clamp_flag = ""
        if cleaned < lower - cfg.price_bound_tolerance:
            cleaned = lower
            clamp_flag = "clamped_to_lower"
        elif cleaned > upper + cfg.price_bound_tolerance:
            cleaned = upper
            clamp_flag = "clamped_to_upper"
        elif cleaned < lower:
            cleaned = lower
            clamp_flag = "nudged_to_lower"
        elif cleaned > upper:
            cleaned = upper
            clamp_flag = "nudged_to_upper"

        try:
            total_variance = _implied_total_variance_from_call_price(
                price=cleaned,
                strike=strike,
                forward=forward,
                discount_factor=df,
                tau=tau,
                tol=cfg.black_inversion_tolerance,
                upper=cfg.total_variance_upper,
            )
            implied_vol = math.sqrt(total_variance / tau)
            status = "ok"
        except Exception as exc:
            total_variance = np.nan
            implied_vol = np.nan
            status = f"failed:{type(exc).__name__}"

        f_obs = cleaned / max(df * strike, np.finfo(float).tiny)

        cleaned_values.append(cleaned)
        iv_values.append(implied_vol)
        tv_values.append(total_variance)
        clamp_flags.append(clamp_flag)
        inv_status.append(status)
        f_obs_list.append(f_obs)

    effective_calls["effective_call_clean"] = cleaned_values
    effective_calls["lower_bound"] = lower_bounds
    effective_calls["upper_bound"] = upper_bounds
    effective_calls["clamp_flag"] = clamp_flags
    effective_calls["black_inversion_status"] = inv_status
    effective_calls["raw_implied_vol"] = iv_values
    effective_calls["raw_total_variance"] = tv_values
    effective_calls["f_observed"] = f_obs_list

    audit = audit.merge(
        effective_calls[
            [
                "slice_id",
                "strike_price",
                "effective_call_clean",
                "lower_bound",
                "upper_bound",
                "clamp_flag",
                "black_inversion_status",
                "raw_implied_vol",
                "raw_total_variance",
                "f_observed",
            ]
        ],
        on=["slice_id", "strike_price"],
        how="left",
    )

    usable = effective_calls.loc[np.isfinite(effective_calls["raw_total_variance"])].copy()
    usable = usable.sort_values(["expiration", "strike_price"]).reset_index(drop=True)
    return usable, audit


# ---------------------------
# Step 4: slice metadata and core band
# ---------------------------


def _build_slice_metadata(raw_tv_quotes: pd.DataFrame) -> Dict[str, SliceMetadata]:
    meta: Dict[str, SliceMetadata] = {}
    for slice_id, grp in raw_tv_quotes.groupby("slice_id", sort=True):
        first = grp.iloc[0]
        kappa_min = float(grp["kappa"].min())
        kappa_max = float(grp["kappa"].max())
        meta[slice_id] = SliceMetadata(
            slice_id=slice_id,
            trade_date=pd.Timestamp(first["trade_date"]),
            expiration=pd.Timestamp(first["expiration"]),
            tau=float(first["tau"]),
            forward_price=float(first["forward_price"]),
            discount_factor=float(first["discount_factor"]),
            pv_div_spot=float(first["pv_div_spot"]),
            observed_kappa_min=kappa_min,
            observed_kappa_max=kappa_max,
            observed_width=kappa_max - kappa_min,
            n_quotes=int(len(grp)),
        )
    return dict(sorted(meta.items(), key=lambda kv: kv[1].expiration))


def _choose_core_kappa_band(
    raw_tv_quotes: pd.DataFrame,
    slice_metadata: Dict[str, SliceMetadata],
    cfg: FenglerConfig,
) -> Tuple[float, float]:
    widths = np.array([m.observed_width for m in slice_metadata.values()], dtype=float)
    median_width = float(np.median(widths))

    eligible_ids = [
        sid
        for sid, m in slice_metadata.items()
        if m.observed_width >= cfg.narrow_slice_ratio * median_width
    ]
    if not eligible_ids:
        eligible_ids = list(slice_metadata.keys())

    boundaries: List[float] = []
    for sid in eligible_ids:
        boundaries.extend(
            [
                slice_metadata[sid].observed_kappa_min,
                slice_metadata[sid].observed_kappa_max,
            ]
        )
    boundaries = sorted(set(boundaries))

    best_band: Optional[Tuple[float, float]] = None
    best_width = -np.inf

    for lower in boundaries:
        for upper in boundaries:
            if upper <= lower:
                continue

            covered = []
            enough_points = True
            for sid in eligible_ids:
                meta = slice_metadata[sid]
                if meta.observed_kappa_min <= lower and meta.observed_kappa_max >= upper:
                    grp = raw_tv_quotes.loc[raw_tv_quotes["slice_id"] == sid]
                    n_inside = int(((grp["kappa"] >= lower) & (grp["kappa"] <= upper)).sum())
                    if n_inside < cfg.min_quotes_inside_band:
                        enough_points = False
                        break
                    covered.append(sid)
            if not enough_points:
                continue
            coverage = len(covered) / max(len(eligible_ids), 1)
            width = upper - lower
            if coverage >= cfg.coverage_target and width > best_width:
                best_band = (lower, upper)
                best_width = width

    if best_band is None:
        best_band = (
            max(slice_metadata[sid].observed_kappa_min for sid in eligible_ids),
            min(slice_metadata[sid].observed_kappa_max for sid in eligible_ids),
        )

    if best_band[1] <= best_band[0]:
        raise ValueError("Unable to determine a positive common kappa band.")

    return best_band


# ---------------------------
# Step 5: common grid and presmoother
# ---------------------------


@dataclass
class CommonGrid:
    kappa_nodes: np.ndarray
    core_band: Tuple[float, float]
    ordered_slice_ids: List[str]


def _build_common_grid(
    raw_tv_quotes: pd.DataFrame,
    core_band: Tuple[float, float],
    cfg: FenglerConfig,
) -> CommonGrid:
    ordered_slices = (
        raw_tv_quotes[["slice_id", "expiration"]]
        .drop_duplicates()
        .sort_values("expiration")
    )
    kappa_nodes = np.linspace(core_band[0], core_band[1], cfg.n_kappa_nodes)
    return CommonGrid(
        kappa_nodes=kappa_nodes,
        core_band=core_band,
        ordered_slice_ids=ordered_slices["slice_id"].tolist(),
    )


def _fit_thin_plate_tv_presmoother(
    raw_tv_quotes: pd.DataFrame,
    cfg: FenglerConfig,
) -> RBFInterpolator:
    points = raw_tv_quotes[["kappa", "tau"]].to_numpy(dtype=float)
    values = raw_tv_quotes["raw_total_variance"].to_numpy(dtype=float)
    model = RBFInterpolator(
        y=points,
        d=values,
        kernel="thin_plate_spline",
        smoothing=cfg.pre_smoother_smoothing,
        neighbors=cfg.pre_smoother_neighbors,
    )
    return model


def _evaluate_presmoother_on_slice_grid(
    presmoother: RBFInterpolator,
    kappa_nodes: np.ndarray,
    tau: float,
    cfg: FenglerConfig,
) -> np.ndarray:
    pts = np.column_stack([kappa_nodes, np.full_like(kappa_nodes, float(tau))])
    tv = presmoother(pts).reshape(-1)
    tv = np.maximum(tv, cfg.total_variance_floor)
    return tv


# ---------------------------
# Step 6: backward repair
# ---------------------------


@dataclass
class RepairedSurface:
    slices: Dict[str, SplineSlice]
    ordered_slice_ids: List[str]
    kappa_nodes: np.ndarray
    core_band: Tuple[float, float]
    slice_metadata: Dict[str, SliceMetadata]


def _repair_surface_backward(
    presmoother: RBFInterpolator,
    grid: CommonGrid,
    slice_metadata: Dict[str, SliceMetadata],
    cfg: FenglerConfig,
) -> RepairedSurface:
    current_kappa_nodes = np.array(grid.kappa_nodes, dtype=float)
    ordered_slice_ids = list(grid.ordered_slice_ids)

    repaired_surface: Optional[RepairedSurface] = None

    for _ in range(cfg.max_calendar_refinement_rounds):
        repaired: Dict[str, SplineSlice] = {}
        next_longer: Optional[SplineSlice] = None

        for slice_id in reversed(ordered_slice_ids):
            meta = slice_metadata[slice_id]
            u = current_kappa_nodes * meta.forward_price
            tv_target = _evaluate_presmoother_on_slice_grid(presmoother, current_kappa_nodes, meta.tau, cfg)
            y_target = np.array(
                [
                    _black_forward_call_from_total_variance(
                        strike=float(k),
                        forward=meta.forward_price,
                        discount_factor=meta.discount_factor,
                        total_variance=float(w),
                    )
                    for k, w in zip(u, tv_target)
                ],
                dtype=float,
            )

            repaired_slice = _solve_one_slice_qp(
                strikes=u,
                y_target=y_target,
                metadata=meta,
                next_longer=next_longer,
                lam=cfg.fixed_lambda,
                cfg=cfg,
            )
            repaired[slice_id] = repaired_slice
            next_longer = repaired_slice

        candidate_surface = RepairedSurface(
            slices={sid: repaired[sid] for sid in ordered_slice_ids},
            ordered_slice_ids=ordered_slice_ids,
            kappa_nodes=current_kappa_nodes,
            core_band=grid.core_band,
            slice_metadata=slice_metadata,
        )

        viol_kappa = _find_dense_calendar_violations(candidate_surface, cfg)
        repaired_surface = candidate_surface
        if viol_kappa.size == 0:
            break
        current_kappa_nodes = np.unique(np.concatenate([current_kappa_nodes, viol_kappa]))
        current_kappa_nodes.sort()

    assert repaired_surface is not None
    return repaired_surface


# ---------------------------
# Step 7: single slice QP
# ---------------------------


def _solve_one_slice_qp(
    strikes: np.ndarray,
    y_target: np.ndarray,
    metadata: SliceMetadata,
    next_longer: Optional[SplineSlice],
    lam: float,
    cfg: FenglerConfig,
) -> SplineSlice:
    n = len(strikes)
    if n < 4:
        raise ValueError("Need at least 4 grid nodes for a stable natural cubic spline repair.")

    Q = _build_Q_matrix(strikes)
    R = _build_R_matrix(strikes)
    A_eq = np.concatenate([Q.T, -R], axis=1)
    b_eq = np.zeros(n - 2)

    P = np.zeros((2 * n - 2, 2 * n - 2), dtype=float)
    P[:n, :n] = np.eye(n)
    P[n:, n:] = lam * R
    q = np.concatenate([-y_target, np.zeros(n - 2)])

    G_list: List[np.ndarray] = []
    h_list: List[float] = []

    # Convexity: gamma >= 0  ->  -gamma <= 0
    for j in range(n - 2):
        row = np.zeros(2 * n - 2)
        row[n + j] = -1.0
        G_list.append(row)
        h_list.append(0.0)

    # Nodewise monotonicity: g_i - g_{i+1} >= 0  ->  -g_i + g_{i+1} <= 0
    for i in range(n - 1):
        row = np.zeros(2 * n - 2)
        row[i] = -1.0
        row[i + 1] = 1.0
        G_list.append(row)
        h_list.append(0.0)

    # Nodewise no-arbitrage price bounds
    for i, strike in enumerate(strikes):
        lower = max(metadata.discount_factor * (metadata.forward_price - strike), 0.0)
        upper = metadata.pv_div_spot

        row_low = np.zeros(2 * n - 2)
        row_low[i] = -1.0  # g_i >= lower -> -g_i <= -lower
        G_list.append(row_low)
        h_list.append(-lower)

        row_up = np.zeros(2 * n - 2)
        row_up[i] = 1.0  # g_i <= upper
        G_list.append(row_up)
        h_list.append(upper)

    # Exact natural-spline edge slopes
    h_left = strikes[1] - strikes[0]
    row_left = np.zeros(2 * n - 2)
    row_left[0] = 1.0 / h_left
    row_left[1] = -1.0 / h_left
    row_left[n + 0] = h_left / 6.0
    # slope_left >= -D  ->  -slope_left <= D
    G_list.append(row_left)
    h_list.append(metadata.discount_factor)

    h_right = strikes[-1] - strikes[-2]
    row_right = np.zeros(2 * n - 2)
    row_right[n - 2] = -1.0 / h_right
    row_right[n - 1] = 1.0 / h_right
    row_right[n + (n - 3)] = h_right / 6.0
    # slope_right <= 0
    G_list.append(row_right)
    h_list.append(0.0)

    # Calendar: shorter slice <= dividend-growth-scaled longer slice at same kappa grid
    if next_longer is not None:
        scale = metadata.pv_div_spot / next_longer.metadata.pv_div_spot
        upper_vals = scale * next_longer.node_values
        for i, ub in enumerate(upper_vals):
            row = np.zeros(2 * n - 2)
            row[i] = 1.0
            G_list.append(row)
            h_list.append(float(ub))

    G = np.vstack(G_list)
    h = np.array(h_list, dtype=float)

    x0 = _equality_qp_start(P, q, A_eq, b_eq)
    x = _solve_qp_trust_constr(P=P, q=q, A_eq=A_eq, b_eq=b_eq, G=G, h=h, x0=x0, cfg=cfg)

    g = x[:n]
    gamma = x[n:]
    return SplineSlice(strikes=np.array(strikes, dtype=float), node_values=g, gammas=gamma, metadata=metadata)


# ---------------------------
# Step 8: outputs
# ---------------------------


def _build_grid_level_output(
    repaired_surface: RepairedSurface,
    grid: CommonGrid,
    slice_metadata: Dict[str, SliceMetadata],
    cfg: FenglerConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for slice_id in repaired_surface.ordered_slice_ids:
        spline = repaired_surface.slices[slice_id]
        meta = slice_metadata[slice_id]
        kappa_nodes = repaired_surface.kappa_nodes
        strikes = kappa_nodes * meta.forward_price
        call_prices = spline.evaluate(strikes)
        call_prices = np.maximum(call_prices, 0.0)

        tv = np.array(
            [
                _implied_total_variance_from_call_price(
                    price=float(c),
                    strike=float(k),
                    forward=meta.forward_price,
                    discount_factor=meta.discount_factor,
                    tau=meta.tau,
                    tol=cfg.black_inversion_tolerance,
                    upper=cfg.total_variance_upper,
                )
                for c, k in zip(call_prices, strikes)
            ]
        )
        iv = np.sqrt(tv / meta.tau)
        put_prices = call_prices - meta.discount_factor * (meta.forward_price - strikes)

        for kappa, strike, c, p, w, vol in zip(kappa_nodes, strikes, call_prices, put_prices, tv, iv):
            inside_slice = (meta.observed_kappa_min <= kappa <= meta.observed_kappa_max)
            support_class = "direct_slice_support" if inside_slice else "core_band_only"
            rows.append(
                {
                    "slice_id": slice_id,
                    "trade_date": meta.trade_date,
                    "expiration": meta.expiration,
                    "tau": meta.tau,
                    "kappa": kappa,
                    "strike_price": strike,
                    "inside_global_core_band": True,
                    "inside_slice_observed_span": inside_slice,
                    "support_class": support_class,
                    "repaired_call_eu": c,
                    "repaired_put_eu": p,
                    "repaired_total_variance": w,
                    "repaired_implied_vol": vol,
                    "f_repaired": c / max(meta.discount_factor * strike, np.finfo(float).tiny),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_at_original_quotes(
    repaired_surface: RepairedSurface,
    original_day: pd.DataFrame,
    effective_call_audit: pd.DataFrame,
    core_band: Tuple[float, float],
    cfg: FenglerConfig,
) -> pd.DataFrame:
    audit_key = effective_call_audit[["slice_id", "strike_price", "source_side_chosen"]].drop_duplicates()
    rows: List[Dict[str, Any]] = []

    for row in original_day.itertuples(index=False):
        slice_id = row.slice_id
        meta = repaired_surface.slice_metadata[slice_id]
        inside_core = core_band[0] <= row.Moneyness <= core_band[1]
        inside_slice = meta.observed_kappa_min <= row.Moneyness <= meta.observed_kappa_max
        support_class = (
            "outside_core_band"
            if not inside_core
            else ("direct_slice_support" if inside_slice else "core_band_only")
        )

        repaired_call = np.nan
        repaired_put = np.nan
        repaired_total_variance = np.nan
        repaired_implied_vol = np.nan
        repaired_price_for_original_class = np.nan
        f_repaired = np.nan

        if inside_core:
            spline = repaired_surface.slices[slice_id]
            repaired_call = float(spline.evaluate(float(row.strike_price)))
            repaired_call = max(repaired_call, 0.0)
            repaired_put = repaired_call - row.discount_factor * (row.forward_price - row.strike_price)
            repaired_total_variance = _implied_total_variance_from_call_price(
                price=repaired_call,
                strike=float(row.strike_price),
                forward=float(row.forward_price),
                discount_factor=float(row.discount_factor),
                tau=float(row.tau),
                tol=cfg.black_inversion_tolerance,
                upper=cfg.total_variance_upper,
            )
            repaired_implied_vol = math.sqrt(repaired_total_variance / float(row.tau))
            repaired_price_for_original_class = repaired_call if str(row.instrument_class).upper() == "C" else repaired_put
            f_repaired = repaired_call / max(row.discount_factor * row.strike_price, np.finfo(float).tiny)

        rows.append(
            {
                **{col: getattr(row, col) for col in original_day.columns},
                "inside_global_core_band": inside_core,
                "inside_slice_observed_span": inside_slice,
                "support_class": support_class,
                "repaired_total_variance": repaired_total_variance,
                "repaired_implied_vol": repaired_implied_vol,
                "repaired_call_eu": repaired_call,
                "repaired_put_eu": repaired_put,
                "repaired_price_for_original_class": repaired_price_for_original_class,
                "f_repaired": f_repaired,
            }
        )

    out = pd.DataFrame(rows)
    out = out.merge(audit_key, on=["slice_id", "strike_price"], how="left")
    out["source_side_chosen"] = out["source_side_chosen"].fillna("")
    out["was_selected_as_effective_quote"] = (
        ((out["source_side_chosen"].isin(["C", "C_fallback"])) & (out["instrument_class"].astype(str).str.upper() == "C"))
        | ((out["source_side_chosen"].isin(["P", "P_fallback"])) & (out["instrument_class"].astype(str).str.upper() == "P"))
    )

    price_error_abs = out["repaired_price_for_original_class"] - out["mid_px_eu"]
    price_error_rel = price_error_abs / np.maximum(out["mid_px_eu"].abs(), 1e-12)
    out["price_error_abs"] = price_error_abs
    out["price_error_rel"] = price_error_rel
    return out


# ---------------------------
# Step 9: diagnostics
# ---------------------------


def _find_dense_calendar_violations(surface: RepairedSurface, cfg: FenglerConfig) -> np.ndarray:
    dense_kappa = np.linspace(surface.core_band[0], surface.core_band[1], cfg.dense_check_points)
    bad_locations: List[float] = []
    ordered = surface.ordered_slice_ids

    for sid_short, sid_long in zip(ordered[:-1], ordered[1:]):
        short_slice = surface.slices[sid_short]
        long_slice = surface.slices[sid_long]
        meta_short = short_slice.metadata
        meta_long = long_slice.metadata

        strikes_short = dense_kappa * meta_short.forward_price
        strikes_long = dense_kappa * meta_long.forward_price

        c_short = short_slice.evaluate(strikes_short)
        c_long = long_slice.evaluate(strikes_long)
        scale = meta_short.pv_div_spot / meta_long.pv_div_spot

        violation_mask = c_short > scale * c_long + cfg.calendar_tolerance
        if violation_mask.any():
            bad_locations.extend(dense_kappa[violation_mask].tolist())

    if not bad_locations:
        return np.array([], dtype=float)
    return np.unique(np.array(bad_locations, dtype=float))


def _run_final_diagnostics(
    repaired_surface: RepairedSurface,
    grid: CommonGrid,
    quote_level_output: pd.DataFrame,
    effective_call_audit: pd.DataFrame,
    cfg: FenglerConfig,
) -> Dict[str, Any]:
    dense_kappa = np.linspace(grid.core_band[0], grid.core_band[1], cfg.dense_check_points)
    strike_checks: List[Dict[str, Any]] = []
    calendar_checks: List[Dict[str, Any]] = []

    # Strike-side diagnostics
    for slice_id in repaired_surface.ordered_slice_ids:
        spline = repaired_surface.slices[slice_id]
        meta = spline.metadata
        strikes = dense_kappa * meta.forward_price
        calls = spline.evaluate(strikes)
        first_diff = np.diff(calls)
        second_diff = np.diff(calls, n=2)
        lower = np.maximum(meta.discount_factor * (meta.forward_price - strikes), 0.0)
        upper = np.full_like(strikes, meta.pv_div_spot)

        strike_checks.append(
            {
                "slice_id": slice_id,
                "expiration": meta.expiration,
                "monotone_pass": bool(np.all(first_diff <= cfg.monotonicity_tolerance)),
                "convex_pass": bool(np.all(second_diff >= -cfg.convexity_tolerance)),
                "bounds_pass": bool(np.all(calls >= lower - cfg.price_bound_tolerance) and np.all(calls <= upper + cfg.price_bound_tolerance)),
                "max_monotonicity_violation": float(max(0.0, np.max(first_diff))),
                "max_convexity_violation": float(max(0.0, -np.min(second_diff))),
            }
        )

    # Calendar diagnostics
    ordered = repaired_surface.ordered_slice_ids
    for sid_short, sid_long in zip(ordered[:-1], ordered[1:]):
        short_slice = repaired_surface.slices[sid_short]
        long_slice = repaired_surface.slices[sid_long]
        meta_short = short_slice.metadata
        meta_long = long_slice.metadata
        strikes_short = dense_kappa * meta_short.forward_price
        strikes_long = dense_kappa * meta_long.forward_price
        c_short = short_slice.evaluate(strikes_short)
        c_long = long_slice.evaluate(strikes_long)
        scale = meta_short.pv_div_spot / meta_long.pv_div_spot
        call_gap = c_short - scale * c_long

        w_short = np.array(
            [
                _implied_total_variance_from_call_price(
                    price=float(c),
                    strike=float(k),
                    forward=meta_short.forward_price,
                    discount_factor=meta_short.discount_factor,
                    tau=meta_short.tau,
                    tol=cfg.black_inversion_tolerance,
                    upper=cfg.total_variance_upper,
                )
                for c, k in zip(c_short, strikes_short)
            ]
        )
        w_long = np.array(
            [
                _implied_total_variance_from_call_price(
                    price=float(c),
                    strike=float(k),
                    forward=meta_long.forward_price,
                    discount_factor=meta_long.discount_factor,
                    tau=meta_long.tau,
                    tol=cfg.black_inversion_tolerance,
                    upper=cfg.total_variance_upper,
                )
                for c, k in zip(c_long, strikes_long)
            ]
        )
        w_gap = w_short - w_long

        calendar_checks.append(
            {
                "slice_id_short": sid_short,
                "slice_id_long": sid_long,
                "expiration_short": meta_short.expiration,
                "expiration_long": meta_long.expiration,
                "call_calendar_pass": bool(np.all(call_gap <= cfg.calendar_tolerance)),
                "total_variance_calendar_pass": bool(np.all(w_gap <= cfg.calendar_tolerance)),
                "max_call_calendar_violation": float(max(0.0, np.max(call_gap))),
                "max_total_variance_calendar_violation": float(max(0.0, np.max(w_gap))),
            }
        )

    def _nanmean_abs(series: pd.Series) -> float:
        vals = np.asarray(series, dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(np.abs(vals))) if vals.size else float('nan')

    support_summary = (
        quote_level_output.groupby("support_class", dropna=False)
        .agg(
            n_rows=("support_class", "size"),
            mean_abs_price_error=("price_error_abs", _nanmean_abs),
        )
        .reset_index()
    )

    effective_subset = effective_call_audit.merge(
        quote_level_output[
            [
                "slice_id",
                "strike_price",
                "repaired_call_eu",
                "repaired_total_variance",
                "repaired_implied_vol",
            ]
        ].drop_duplicates(),
        on=["slice_id", "strike_price"],
        how="left",
    )
    effective_subset["effective_call_error"] = effective_subset["repaired_call_eu"] - effective_subset["effective_call_clean"]
    effective_subset["effective_total_variance_error"] = effective_subset["repaired_total_variance"] - effective_subset["raw_total_variance"]
    effective_subset["effective_iv_error"] = effective_subset["repaired_implied_vol"] - effective_subset["raw_implied_vol"]

    return {
        "strike_checks": pd.DataFrame(strike_checks),
        "calendar_checks": pd.DataFrame(calendar_checks),
        "support_summary": support_summary,
        "effective_fit_residuals": effective_subset,
        "no_strike_arb_pass": bool(pd.DataFrame(strike_checks)[["monotone_pass", "convex_pass", "bounds_pass"]].all().all()),
        "no_calendar_arb_pass": bool(pd.DataFrame(calendar_checks)[["call_calendar_pass", "total_variance_calendar_pass"]].all().all()) if calendar_checks else True,
    }

# SMILE ISSUE
def compute_smile_extrema(iv: np.ndarray) -> int:
    """Count local extrema (sign changes in first derivative)."""
    d1 = np.diff(iv)
    return int(np.sum(np.diff(np.sign(d1)) != 0))


def validate_repaired_smiles(
        repaired_surface: RepairedSurface,
        cfg: FenglerConfig,
        max_extrema: int = 1,
) -> List[Dict[str, Any]]:
    """
    Check each slice for waviness.
    Returns list of problematic slices.

    Args:
        max_extrema: Maximum allowed local extrema (1 = smile, 0 = pure skew)
    """
    issues = []

    for slice_id in repaired_surface.ordered_slice_ids:
        spline = repaired_surface.slices[slice_id]
        meta = repaired_surface.slice_metadata[slice_id]

        # Evaluate on dense grid
        kappa_dense = np.linspace(
            repaired_surface.core_band[0],
            repaired_surface.core_band[1],
            201
        )
        strikes_dense = kappa_dense * meta.forward_price
        call_prices = spline.evaluate(strikes_dense)
        call_prices = np.maximum(call_prices, 0.0)

        # Convert to IV
        iv = np.array([
            math.sqrt(
                _implied_total_variance_from_call_price(
                    price=float(c),
                    strike=float(k),
                    forward=meta.forward_price,
                    discount_factor=meta.discount_factor,
                    tau=meta.tau,
                    tol=cfg.black_inversion_tolerance,
                    upper=cfg.total_variance_upper,
                ) / meta.tau
            )
            for c, k in zip(call_prices, strikes_dense)
        ])

        n_extrema = compute_smile_extrema(iv)
        days_to_expiry = int(round(meta.tau * 365))

        if n_extrema > max_extrema:
            issues.append({
                "slice_id": slice_id,
                "days_to_expiry": days_to_expiry,
                "n_extrema": n_extrema,
                "expiration": meta.expiration,
            })
            print(f"  ⚠️  {days_to_expiry:>3}d ({meta.expiration.date()}): {n_extrema} extrema detected")

    if not issues:
        print("  ✓ All slices pass waviness check")

    return issues


# ---------------------------
# Low-level helpers
# ---------------------------


def _build_Q_matrix(strikes: np.ndarray) -> np.ndarray:
    u = np.asarray(strikes, dtype=float)
    h = np.diff(u)
    n = len(u)
    if np.any(h <= 0):
        raise ValueError("Strikes must be strictly increasing.")
    Q = np.zeros((n, n - 2), dtype=float)
    for p in range(n - 2):
        Q[p, p] = 1.0 / h[p]
        Q[p + 1, p] = -1.0 / h[p] - 1.0 / h[p + 1]
        Q[p + 2, p] = 1.0 / h[p + 1]
    return Q


def _build_R_matrix(strikes: np.ndarray) -> np.ndarray:
    u = np.asarray(strikes, dtype=float)
    h = np.diff(u)
    n = len(u)
    R = np.zeros((n - 2, n - 2), dtype=float)
    for p in range(n - 2):
        R[p, p] = (h[p] + h[p + 1]) / 3.0
        if p < n - 3:
            R[p, p + 1] = h[p + 1] / 6.0
            R[p + 1, p] = h[p + 1] / 6.0
    return R


def _equality_qp_start(P: np.ndarray, q: np.ndarray, A_eq: np.ndarray, b_eq: np.ndarray) -> np.ndarray:
    n = P.shape[0]
    m = A_eq.shape[0]
    KKT = np.block(
        [
            [P, A_eq.T],
            [A_eq, np.zeros((m, m), dtype=float)],
        ]
    )
    rhs = np.concatenate([-q, b_eq])
    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    return sol[:n]


def _solve_qp_trust_constr(
    P: np.ndarray,
    q: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    x0: np.ndarray,
    cfg: FenglerConfig,
) -> np.ndarray:
    def fun(x: np.ndarray) -> float:
        return 0.5 * float(x @ P @ x) + float(q @ x)

    def jac(x: np.ndarray) -> np.ndarray:
        return P @ x + q

    def hess(_: np.ndarray) -> np.ndarray:
        return P

    constraints = [
        LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
        LinearConstraint(G, lb=-np.inf * np.ones_like(h), ub=h),
    ]

    res = minimize(
        fun=fun,
        x0=x0,
        jac=jac,
        hess=hess,
        method="trust-constr",
        constraints=constraints,
        options={
            "gtol": cfg.qp_gtol,
            "xtol": cfg.qp_xtol,
            "barrier_tol": cfg.qp_barrier_tol,
            "maxiter": cfg.qp_maxiter,
            "verbose": 3 if cfg.verbose else 0,
        },
    )
    if not res.success:
        raise RuntimeError(f"QP solver failed: {res.message}")
    return np.asarray(res.x, dtype=float)


def _evaluate_natural_cubic_spline(
    strikes: np.ndarray,
    g: np.ndarray,
    gammas_interior: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    u = np.asarray(strikes, dtype=float)
    y = np.asarray(x, dtype=float)
    full_gamma = np.concatenate(([0.0], np.asarray(gammas_interior, dtype=float), [0.0]))
    out = np.empty_like(y, dtype=float)

    h_left = u[1] - u[0]
    slope_left = (g[1] - g[0]) / h_left - (h_left / 6.0) * full_gamma[1]

    h_right = u[-1] - u[-2]
    slope_right = (g[-1] - g[-2]) / h_right + (h_right / 6.0) * full_gamma[-2]

    left_mask = y <= u[0]
    right_mask = y >= u[-1]
    mid_mask = ~(left_mask | right_mask)

    out[left_mask] = g[0] + (y[left_mask] - u[0]) * slope_left
    out[right_mask] = g[-1] + (y[right_mask] - u[-1]) * slope_right

    if np.any(mid_mask):
        z = y[mid_mask]
        idx = np.searchsorted(u, z, side="right") - 1
        idx = np.clip(idx, 0, len(u) - 2)

        ui = u[idx]
        uj = u[idx + 1]
        hi = uj - ui
        gi = g[idx]
        gj = g[idx + 1]
        gamma_i = full_gamma[idx]
        gamma_j = full_gamma[idx + 1]

        term1 = ((z - ui) * gj + (uj - z) * gi) / hi
        term2 = (1.0 / 6.0) * (z - ui) * (uj - z) * (
            (1.0 + (z - ui) / hi) * gamma_j + (1.0 + (uj - z) / hi) * gamma_i
        )
        out[mid_mask] = term1 - term2

    return out


def _black_forward_call_from_total_variance(
    strike: float,
    forward: float,
    discount_factor: float,
    total_variance: float,
) -> float:
    intrinsic = discount_factor * max(forward - strike, 0.0)
    if total_variance <= 0.0:
        return intrinsic
    sqrt_w = math.sqrt(total_variance)
    log_fk = math.log(forward / strike)
    d1 = log_fk / sqrt_w + 0.5 * sqrt_w
    d2 = d1 - sqrt_w
    return discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def _implied_total_variance_from_call_price(
    price: float,
    strike: float,
    forward: float,
    discount_factor: float,
    tau: float,
    tol: float,
    upper: float,
) -> float:
    lower = max(discount_factor * (forward - strike), 0.0)
    upper_price = discount_factor * forward

    if price <= lower + tol:
        return 0.0
    if price >= upper_price - tol:
        return upper

    def f(w: float) -> float:
        return _black_forward_call_from_total_variance(strike, forward, discount_factor, w) - price

    lo = 1e-16
    hi = max(1.0, upper)
    flo = f(lo)
    fhi = f(hi)
    expand = 0
    while flo * fhi > 0 and expand < 10:
        hi *= 2.0
        fhi = f(hi)
        expand += 1
    if flo * fhi > 0:
        raise RuntimeError("Unable to bracket implied total variance.")
    return float(brentq(f, lo, hi, xtol=tol, rtol=tol, maxiter=200))


# ---------------------------
# Optional convenience helper
# ---------------------------


def save_outputs_to_excel(results: Dict[str, Any], output_path: str) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results["quote_level_output"].to_excel(writer, sheet_name="quote_level_output", index=False)
        results["grid_level_output"].to_excel(writer, sheet_name="grid_level_output", index=False)
        results["effective_call_audit_table"].to_excel(writer, sheet_name="effective_call_audit", index=False)
        results["slice_metadata_table"].to_excel(writer, sheet_name="slice_metadata", index=False)
        diagnostics = results["diagnostics"]
        diagnostics["strike_checks"].to_excel(writer, sheet_name="diag_strike_checks", index=False)
        diagnostics["calendar_checks"].to_excel(writer, sheet_name="diag_calendar_checks", index=False)
        diagnostics["support_summary"].to_excel(writer, sheet_name="diag_support_summary", index=False)
        diagnostics["effective_fit_residuals"].to_excel(writer, sheet_name="diag_fit_residuals", index=False)
