import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal
import numpy as np
import pandas as pd
try:
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False
    njit = None

Branch = Literal["underlying", "letf"]
MonotoneMode = Literal["auto", "increasing", "decreasing", "none"]

REQUIRED_COLUMNS = [
    "underlying",
    "trade_date",
    "symbol",
    "expiration",
    "instrument_class",
    "strike_price",
    "mid_px",
    "underlying_price",
    "time_to_expiration",
    "discount_factor",
]

OUTPUT_ADDITIONAL_COLUMNS = [
    "carry",
    "forward_factor",
    "mid_px_eu",
    "Log_Moneyness",
    "Moneyness",
    "forward_price",
    "tau",
]

SLICE_KEYS = ["underlying", "trade_date", "expiration"]
GROUP_KEYS = ["underlying", "trade_date"]


@dataclass(frozen=True)
class PipelineConfig:
    branch: Branch
    ticker: str | None = None
    input_sheet: str | None = None
    output_sheet: str = "deamericanized"
    day_count: int = 365
    time_col_is_years: bool = False
    tree_dt_target: float = 8e-4
    tree_min_steps: int = 500
    iv_tol: float = 1e-8
    iv_max_iter: int = 80
    forward_n_atm_pairs: int = 5
    forward_mad_z: float = 5.0
    curve_smooth: float = 1e-6
    monotone_penalty: float = 1e-4
    smooth_iterations: int = 8
    letf_carry_floor: float | None = None
    letf_exclude_long_end_from_fit: bool = False
    letf_long_end_cutoff_days: float | None = None
    letf_monotone_mode: MonotoneMode = "none"


@dataclass
class DeAmericanizedQuote:
    price_eu: float
    sigma_am: float


class DeAmericanizationError(Exception):
    pass


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower()


def coerce_float(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip().replace("\u00a0", "")
    if s == "":
        return float("nan")
    s = s.replace(" ", "")

    if "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    return float(s)


def resolve_time_to_years(raw_t: float, *, day_count: int, time_col_is_years: bool) -> float:
    raw_t = coerce_float(raw_t)
    if time_col_is_years:
        return raw_t
    return raw_t / float(day_count)


def compute_steps(T: float, *, dt_target: float, min_steps: int) -> int:
    return max(int(min_steps), int(math.ceil(T / dt_target)))


def _normalize_option_type(values: pd.Series) -> pd.Series:
    out = (
        values.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[^A-Z]", "", regex=True)
        .str[0]
    )
    bad = ~out.isin(["C", "P"])
    if bad.any():
        bad_values = sorted(pd.unique(values[bad].astype(str)))
        raise ValueError(
            "instrument_class must map to C or P. "
            f"Could not interpret: {bad_values[:10]}"
        )
    return out


def _validate_and_canonicalize(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    normalized_lookup = {normalize_column_name(c): c for c in df.columns}
    missing = [c for c in REQUIRED_COLUMNS if c not in normalized_lookup]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rename_map = {normalized_lookup[c]: c for c in REQUIRED_COLUMNS}
    canonical = df.rename(columns=rename_map).copy()
    original_columns = list(df.columns)
    return canonical, original_columns


def _prepare_base_frame(df: pd.DataFrame, *, config: PipelineConfig) -> pd.DataFrame:
    work, _ = _validate_and_canonicalize(df)
    work = work.copy()
    work["_row_id"] = np.arange(len(work))

    if config.ticker is not None:
        ticker = str(config.ticker).strip().upper()
        mask = work["underlying"].astype(str).str.strip().str.upper().eq(ticker)
        work = work.loc[mask].copy()

    work["option_type"] = _normalize_option_type(work["instrument_class"])
    work["strike_price"] = work["strike_price"].map(coerce_float)
    work["mid_px"] = work["mid_px"].map(coerce_float)
    work["underlying_price"] = work["underlying_price"].map(coerce_float)
    work["time_to_expiration"] = work["time_to_expiration"].map(coerce_float)
    work["discount_factor"] = work["discount_factor"].map(coerce_float)
    work["tau"] = work["time_to_expiration"].map(
        lambda x: resolve_time_to_years(
            x,
            day_count=config.day_count,
            time_col_is_years=config.time_col_is_years,
        )
        if np.isfinite(x)
        else np.nan
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        work["r"] = -np.log(work["discount_factor"]) / work["tau"]

    if work.empty:
        raise ValueError("No rows remain after applying the ticker filter.")

    return work


# ---------------------------------------------------------------------------
# CRR pricers
# ---------------------------------------------------------------------------


def _crr_params(r: float, b: float, T: float, sigma: float, steps: int) -> tuple[float, float, float, float, float]:
    dt = T / steps
    sqrt_dt = math.sqrt(dt)
    u = math.exp(sigma * sqrt_dt)
    d = 1.0 / u
    p = (math.exp(b * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise DeAmericanizationError(
            "CRR risk-neutral probability fell outside (0, 1)."
        )
    disc = math.exp(-r * dt)
    return dt, u, d, p, disc


if HAS_NUMBA:

    @njit(cache=True)
    def _american_option_crr_numba(
        S: float,
        K: float,
        r: float,
        b: float,
        T: float,
        sigma: float,
        steps: int,
        is_call_flag: int,
    ) -> float:
        dt = T / steps
        sqrt_dt = math.sqrt(dt)
        u = math.exp(sigma * sqrt_dt)
        d = 1.0 / u
        p = (math.exp(b * dt) - d) / (u - d)
        if not (0.0 < p < 1.0):
            return math.nan
        disc = math.exp(-r * dt)

        stock = np.empty(steps + 1, dtype=np.float64)
        values = np.empty(steps + 1, dtype=np.float64)

        for j in range(steps + 1):
            s_val = S * (u ** j) * (d ** (steps - j))
            stock[j] = s_val
            payoff = s_val - K if is_call_flag == 1 else K - s_val
            values[j] = payoff if payoff > 0.0 else 0.0

        for n in range(steps - 1, -1, -1):
            for j in range(n + 1):
                continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j])
                stock[j] = stock[j] / d
                exercise = stock[j] - K if is_call_flag == 1 else K - stock[j]
                if exercise < 0.0:
                    exercise = 0.0
                values[j] = exercise if exercise > continuation else continuation

        return values[0]


    @njit(cache=True)
    def _american_and_european_option_crr_numba(
        S: float,
        K: float,
        r: float,
        b: float,
        T: float,
        sigma: float,
        steps: int,
        is_call_flag: int,
    ) -> tuple[float, float]:
        dt = T / steps
        sqrt_dt = math.sqrt(dt)
        u = math.exp(sigma * sqrt_dt)
        d = 1.0 / u
        p = (math.exp(b * dt) - d) / (u - d)
        if not (0.0 < p < 1.0):
            return math.nan, math.nan
        disc = math.exp(-r * dt)

        stock = np.empty(steps + 1, dtype=np.float64)
        am_values = np.empty(steps + 1, dtype=np.float64)
        eu_values = np.empty(steps + 1, dtype=np.float64)

        for j in range(steps + 1):
            s_val = S * (u ** j) * (d ** (steps - j))
            stock[j] = s_val
            payoff = s_val - K if is_call_flag == 1 else K - s_val
            payoff = payoff if payoff > 0.0 else 0.0
            am_values[j] = payoff
            eu_values[j] = payoff

        for n in range(steps - 1, -1, -1):
            for j in range(n + 1):
                am_cont = disc * (p * am_values[j + 1] + (1.0 - p) * am_values[j])
                eu_values[j] = disc * (p * eu_values[j + 1] + (1.0 - p) * eu_values[j])
                stock[j] = stock[j] / d
                exercise = stock[j] - K if is_call_flag == 1 else K - stock[j]
                if exercise < 0.0:
                    exercise = 0.0
                am_values[j] = exercise if exercise > am_cont else am_cont

        return am_values[0], eu_values[0]


def american_option_crr(
    S: float,
    K: float,
    r: float,
    b: float,
    T: float,
    sigma: float,
    steps: int,
    *,
    is_call: bool,
) -> float:
    if HAS_NUMBA:
        out = _american_option_crr_numba(S, K, r, b, T, sigma, steps, 1 if is_call else 0)
        if math.isnan(out):
            raise DeAmericanizationError(
                "CRR risk-neutral probability fell outside (0, 1)."
            )
        return float(out)

    _, u, d, p, disc = _crr_params(r, b, T, sigma, steps)
    j = np.arange(steps + 1, dtype=float)
    stock = S * (u ** j) * (d ** (steps - j))
    values = np.maximum(stock - K, 0.0) if is_call else np.maximum(K - stock, 0.0)

    for n in range(steps - 1, -1, -1):
        continuation = disc * (p * values[1 : n + 2] + (1.0 - p) * values[0 : n + 1])
        stock = stock[0 : n + 1] / d
        exercise = np.maximum(stock - K, 0.0) if is_call else np.maximum(K - stock, 0.0)
        values = np.maximum(exercise, continuation)

    return float(values[0])


def american_and_european_option_crr(
    S: float,
    K: float,
    r: float,
    b: float,
    T: float,
    sigma: float,
    steps: int,
    *,
    is_call: bool,
) -> tuple[float, float]:
    if HAS_NUMBA:
        am, eu = _american_and_european_option_crr_numba(
            S, K, r, b, T, sigma, steps, 1 if is_call else 0
        )
        if math.isnan(am) or math.isnan(eu):
            raise DeAmericanizationError(
                "CRR risk-neutral probability fell outside (0, 1)."
            )
        return float(am), float(eu)

    _, u, d, p, disc = _crr_params(r, b, T, sigma, steps)
    j = np.arange(steps + 1, dtype=float)
    stock = S * (u ** j) * (d ** (steps - j))
    terminal = np.maximum(stock - K, 0.0) if is_call else np.maximum(K - stock, 0.0)
    am_values = terminal.copy()
    eu_values = terminal.copy()

    for n in range(steps - 1, -1, -1):
        am_cont = disc * (p * am_values[1 : n + 2] + (1.0 - p) * am_values[0 : n + 1])
        eu_values = disc * (p * eu_values[1 : n + 2] + (1.0 - p) * eu_values[0 : n + 1])
        stock = stock[0 : n + 1] / d
        exercise = np.maximum(stock - K, 0.0) if is_call else np.maximum(K - stock, 0.0)
        am_values = np.maximum(exercise, am_cont)

    return float(am_values[0]), float(eu_values[0])


# ---------------------------------------------------------------------------
# American -> European inversion
# ---------------------------------------------------------------------------


def _solve_sigma_bisection(
    *,
    S: float,
    K: float,
    r: float,
    b: float,
    T: float,
    american_price: float,
    steps: int,
    is_call: bool,
    sigma_seed: float | None,
    tol: float,
    max_iter: int,
) -> float:
    dt = T / steps
    min_valid_sigma = max(1e-6, abs(b) * math.sqrt(dt) + 1e-10)

    def objective(sig: float) -> float:
        return american_option_crr(S, K, r, b, T, sig, steps, is_call=is_call) - american_price

    def try_bracket(lo: float, hi: float) -> tuple[float, float, float, float]:
        lo = max(lo, min_valid_sigma)
        hi = max(hi, lo * 1.2)
        f_lo = objective(lo)
        f_hi = objective(hi)
        while f_hi < 0.0 and hi < 10.0:
            hi *= 2.0
            f_hi = objective(hi)
        return lo, hi, f_lo, f_hi

    if sigma_seed is not None and sigma_seed > 0.0:
        lo, hi, f_lo, f_hi = try_bracket(0.70 * sigma_seed, 1.35 * sigma_seed)
        if f_lo > 0.0 or f_hi < 0.0:
            lo, hi, f_lo, f_hi = try_bracket(min_valid_sigma, max(1.0, 2.0 * sigma_seed))
    else:
        lo, hi, f_lo, f_hi = try_bracket(min_valid_sigma, 1.0)

    if f_lo > 0.0:
        raise DeAmericanizationError(
            "Could not bracket the implied-volatility root at the lower bound."
        )
    if f_hi < 0.0:
        raise DeAmericanizationError(
            "Could not bracket the implied-volatility root at the upper bound."
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = objective(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_mid > 0.0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


PUT_INTRINSIC_BUFFER = 0.01
CALL_INTRINSIC_ABS_TOL = 1e-6


def deamericanize_option(
    *,
    S: float,
    K: float,
    tau: float,
    discount_factor: float,
    american_price: float,
    option_type: str,
    carry: float,
    config: PipelineConfig,
    sigma_seed: float | None,
) -> DeAmericanizedQuote:
    opt = str(option_type).strip().upper()
    if opt not in {"C", "P"}:
        raise DeAmericanizationError("option_type must be C or P.")
    is_call = opt == "C"

    if not np.isfinite(S) or S <= 0.0:
        raise DeAmericanizationError("underlying_price must be positive.")
    if not np.isfinite(K) or K <= 0.0:
        raise DeAmericanizationError("strike_price must be positive.")
    if not np.isfinite(tau) or tau <= 0.0:
        raise DeAmericanizationError("tau must be positive.")
    if not np.isfinite(discount_factor) or not (0.0 < discount_factor <= 1.0):
        raise DeAmericanizationError("discount_factor must lie in (0, 1].")
    if not np.isfinite(american_price) or american_price <= 0.0:
        raise DeAmericanizationError("mid_px must be positive.")
    if not np.isfinite(carry):
        raise DeAmericanizationError("carry must be finite.")

    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if is_call:
        intrinsic_tol = max(1e-8, CALL_INTRINSIC_ABS_TOL * max(S, K, 1.0))
        if american_price <= intrinsic + intrinsic_tol:
            raise DeAmericanizationError(
                "call mid_px is essentially equal to intrinsic value."
            )
    else:
        if american_price <= intrinsic * (1.0 + PUT_INTRINSIC_BUFFER):
            raise DeAmericanizationError(
                "put mid_px does not exceed intrinsic value by more than 1%."
            )

    r = -math.log(discount_factor) / tau
    b = r - carry
    steps = compute_steps(tau, dt_target=config.tree_dt_target, min_steps=config.tree_min_steps)

    sigma_am = _solve_sigma_bisection(
        S=S,
        K=K,
        r=r,
        b=b,
        T=tau,
        american_price=american_price,
        steps=steps,
        is_call=is_call,
        sigma_seed=sigma_seed,
        tol=config.iv_tol,
        max_iter=config.iv_max_iter,
    )
    _, european_price = american_and_european_option_crr(
        S,
        K,
        r,
        b,
        tau,
        sigma_am,
        steps,
        is_call=is_call,
    )
    return DeAmericanizedQuote(price_eu=european_price, sigma_am=sigma_am)


# ---------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------


def _build_matched_pairs(g: pd.DataFrame, *, price_col: str) -> pd.DataFrame:
    use = g[["strike_price", "option_type", price_col]].copy()
    use = use[np.isfinite(use["strike_price"]) & np.isfinite(use[price_col])]
    if use.empty:
        return pd.DataFrame()

    grouped = use.groupby(["strike_price", "option_type"], as_index=False)[price_col].median()
    pivot = grouped.pivot(index="strike_price", columns="option_type", values=price_col)
    if not {"C", "P"}.issubset(pivot.columns):
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["C", "P"]).sort_index().copy()
    if pivot.empty:
        return pd.DataFrame()
    pivot.index = pivot.index.astype(float)
    return pivot


def _estimate_raw_forward_for_expiry(
    g: pd.DataFrame,
    *,
    price_col: str,
    n_atm_pairs: int,
    mad_z: float,
) -> dict[str, float]:
    D = float(np.nanmedian(g["discount_factor"]))
    S = float(np.nanmedian(g["underlying_price"]))
    tau = float(np.nanmedian(g["tau"]))
    r = float(np.nanmedian(g["r"]))

    out = {
        "raw_forward": np.nan,
        "raw_log_forward": np.nan,
        "pair_count_used": 0.0,
        "tau": tau,
        "r": r,
        "spot": S,
    }

    if not (np.isfinite(D) and D > 0.0 and np.isfinite(S) and S > 0.0 and np.isfinite(tau) and tau > 0.0):
        return out

    pairs = _build_matched_pairs(g, price_col=price_col)
    if pairs.empty:
        return out

    pairs = pairs.copy()
    pairs["cp_diff"] = pairs["C"] - pairs["P"]
    pairs["forward"] = pairs.index.to_numpy(dtype=float) + pairs["cp_diff"].to_numpy(dtype=float) / D
    pairs["atm_score"] = np.abs(pairs["cp_diff"].to_numpy(dtype=float))
    pairs = pairs[np.isfinite(pairs["forward"]) & (pairs["forward"] > 0.0)]
    if pairs.empty:
        return out

    pairs["__strike"] = pairs.index.to_numpy(dtype=float)
    selected = pairs.sort_values(["atm_score", "__strike"]).head(max(1, min(int(n_atm_pairs), len(pairs))))
    f_vals = selected["forward"].to_numpy(dtype=float)
    med = float(np.nanmedian(f_vals))
    mad = float(np.nanmedian(np.abs(f_vals - med)))

    if np.isfinite(mad) and mad > 0.0:
        scale = 1.4826 * mad
        keep = np.abs(f_vals - med) <= mad_z * scale
        if not np.any(keep):
            keep = np.ones_like(f_vals, dtype=bool)
        used = selected.loc[keep].copy()
    else:
        used = selected.copy()

    if used.empty:
        return out

    F = float(np.nanmean(used["forward"].to_numpy(dtype=float)))
    if not (np.isfinite(F) and F > 0.0):
        return out

    out["raw_forward"] = F
    out["raw_log_forward"] = math.log(F / S)
    out["pair_count_used"] = float(len(used))
    return out


def _estimate_letf_carry_for_expiry(g: pd.DataFrame) -> tuple[float, int]:
    D = float(np.nanmedian(g["discount_factor"]))
    S = float(np.nanmedian(g["underlying_price"]))
    tau = float(np.nanmedian(g["tau"]))
    if not (np.isfinite(D) and D > 0.0 and np.isfinite(S) and S > 0.0 and np.isfinite(tau) and tau > 0.0):
        return np.nan, 0

    pairs = _build_matched_pairs(g, price_col="mid_px")
    if pairs.empty:
        return np.nan, 0

    strikes = pairs.index.to_numpy(dtype=float)
    atm_idx = int(np.argmin(np.abs(strikes / S - 1.0)))
    selected_idx = [atm_idx]
    if atm_idx - 1 >= 0:
        selected_idx.append(atm_idx - 1)
    if atm_idx + 1 < len(strikes):
        selected_idx.append(atm_idx + 1)

    q_values: list[float] = []
    for idx in sorted(set(selected_idx)):
        K = float(strikes[idx])
        C = float(pairs.iloc[idx]["C"])
        P = float(pairs.iloc[idx]["P"])
        numerator = C - P + K * D
        if numerator <= 0.0:
            continue
        q_mid = -math.log(numerator / S) / tau
        if np.isfinite(q_mid):
            q_values.append(q_mid)

    if not q_values:
        return np.nan, 0

    return float(np.mean(q_values)), int(len(q_values))


def _clip_carry_to_floor(carry: float, carry_floor: float | None) -> float:
    if carry_floor is None or not np.isfinite(carry_floor):
        return carry
    if np.isfinite(carry):
        return max(float(carry), float(carry_floor))
    return float(carry_floor)


def _infer_monotone_mode_from_carry(
    r: np.ndarray,
    carry: np.ndarray,
    *,
    requested_mode: MonotoneMode,
) -> MonotoneMode:
    if requested_mode != "auto":
        return requested_mode

    valid = np.isfinite(r) & np.isfinite(carry)
    if not np.any(valid):
        return "none"

    median_net_carry = float(np.nanmedian(r[valid] - carry[valid]))
    if median_net_carry > 1e-12:
        return "increasing"
    if median_net_carry < -1e-12:
        return "decreasing"
    return "none"


def _normalize_positive_weights(weights: np.ndarray) -> np.ndarray:
    out = np.asarray(weights, dtype=float).copy()
    out = np.where(np.isfinite(out), out, 0.0)
    out = np.where(out > 0.0, out, 0.0)
    positive = out[out > 0.0]
    if positive.size:
        out = out / max(float(np.nanmedian(positive)), 1.0)
    return out


def _long_end_cutoff_tau(config: PipelineConfig) -> float | None:
    cutoff_days = config.letf_long_end_cutoff_days
    if cutoff_days is None or not np.isfinite(cutoff_days) or cutoff_days <= 0.0:
        return None
    return float(cutoff_days) / float(config.day_count)


def _build_letf_fit_weights(
    tau: np.ndarray,
    pair_count_used: np.ndarray,
    raw_point_count: np.ndarray,
    *,
    config: PipelineConfig,
) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    pair_count_used = np.asarray(pair_count_used, dtype=float)
    raw_point_count = np.asarray(raw_point_count, dtype=float)

    pair_count_used = np.where(np.isfinite(pair_count_used), pair_count_used, 0.0)
    pair_count_used = np.where(pair_count_used > 0.0, pair_count_used, 0.0)
    raw_point_count = np.where(np.isfinite(raw_point_count), raw_point_count, 0.0)
    raw_point_count = np.where(raw_point_count > 0.0, raw_point_count, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        maturity_scale = np.where(np.isfinite(tau) & (tau > 0.0), 1.0 / np.sqrt(tau), 0.0)

    # Primary weighting: parity-forward slice quality times raw-carry point count,
    # downweighted by maturity. This mirrors the underlying branch more closely
    # by letting slice reliability matter, not only tau.
    weights = pair_count_used * raw_point_count * maturity_scale

    # Sparse days can have no usable pass-1 parity forwards anywhere. In that case,
    # fall back to the raw carry point count so the seed-based curve can still be fit.
    if not np.any(weights > 0.0):
        weights = raw_point_count * maturity_scale

    cutoff_tau = _long_end_cutoff_tau(config)
    if config.letf_exclude_long_end_from_fit and cutoff_tau is not None:
        weights = np.where(tau > cutoff_tau, 0.0, weights)

    return _normalize_positive_weights(weights)


# ---------------------------------------------------------------------------
# Underlying forward smoothing
# ---------------------------------------------------------------------------


def _first_diff_matrix(t: np.ndarray) -> np.ndarray:
    n = len(t)
    if n < 2:
        return np.zeros((0, n), dtype=float)
    dt = np.diff(t)
    B = np.zeros((n - 1, n), dtype=float)
    for i, h in enumerate(dt):
        h = max(float(h), 1e-12)
        B[i, i] = -1.0 / h
        B[i, i + 1] = 1.0 / h
    return B


def _slope_change_matrix(t: np.ndarray) -> np.ndarray:
    n = len(t)
    if n < 3:
        return np.zeros((0, n), dtype=float)
    B = _first_diff_matrix(t)
    M = np.zeros((n - 2, n - 1), dtype=float)
    for i in range(n - 2):
        M[i, i] = -1.0
        M[i, i + 1] = 1.0
    return M @ B


def _linear_fill(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    obs = np.isfinite(y)
    if not np.any(obs):
        return np.full_like(y, np.nan, dtype=float)
    if np.sum(obs) == 1:
        return np.full_like(y, float(y[obs][0]), dtype=float)
    return np.interp(t, t[obs], y[obs], left=float(y[obs][0]), right=float(y[obs][-1]))


def _fit_smoothed_log_forward(
    tau: np.ndarray,
    raw_log_forward: np.ndarray,
    weights: np.ndarray,
    *,
    curve_smooth: float,
    monotone_penalty: float,
    iterations: int,
    monotone_mode: MonotoneMode = "increasing",
) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    y = np.asarray(raw_log_forward, dtype=float)
    w = np.asarray(weights, dtype=float)

    valid_tau = np.isfinite(tau) & (tau > 0.0)
    if not np.all(valid_tau):
        raise ValueError("All tau nodes passed to the smoother must be positive and finite.")

    obs = np.isfinite(y) & (w > 0.0)
    if not np.any(obs):
        return np.full_like(y, np.nan, dtype=float)

    if np.sum(obs) == 1:
        return np.full_like(y, float(y[obs][0]), dtype=float)

    y_obs = np.where(obs, y, np.nan)
    x = _linear_fill(tau, y_obs)
    B = _first_diff_matrix(tau)
    C = _slope_change_matrix(tau)
    rhs = np.nan_to_num(y_obs, nan=0.0) * w
    W = np.diag(w)
    ridge = 1e-10 * np.eye(len(tau))

    for _ in range(max(1, int(iterations))):
        if B.shape[0] == 0 or monotone_mode == "none":
            monotone_violation_penalty = np.zeros((len(tau), len(tau)), dtype=float)
        else:
            slopes = B @ x
            if monotone_mode == "increasing":
                mask = (slopes < 0.0).astype(float)
            elif monotone_mode == "decreasing":
                mask = (slopes > 0.0).astype(float)
            else:
                raise ValueError(f"Unsupported monotone_mode: {monotone_mode}")
            monotone_violation_penalty = B.T @ np.diag(mask) @ B

        system = W + curve_smooth * (C.T @ C) + monotone_penalty * monotone_violation_penalty + ridge
        x_new = np.linalg.solve(system, rhs)
        if np.max(np.abs(x_new - x)) < 1e-12:
            x = x_new
            break
        x = x_new

    return x


# ---------------------------------------------------------------------------
# Branch processors
# ---------------------------------------------------------------------------


def _deamericanize_side(
    side: pd.DataFrame,
    *,
    carry: float,
    config: PipelineConfig,
) -> pd.Series:
    result = pd.Series(np.nan, index=side.index, dtype=float)

    for option_type, group in side.groupby("option_type", sort=False, dropna=False):
        sigma_seed: float | None = None
        ordered = group.sort_values("strike_price", kind="mergesort")

        for idx, row in ordered.iterrows():
            try:
                quote = deamericanize_option(
                    S=float(row["underlying_price"]),
                    K=float(row["strike_price"]),
                    tau=float(row["tau"]),
                    discount_factor=float(row["discount_factor"]),
                    american_price=float(row["mid_px"]),
                    option_type=str(option_type),
                    carry=float(carry),
                    config=config,
                    sigma_seed=sigma_seed,
                )
                result.at[idx] = quote.price_eu
                sigma_seed = quote.sigma_am
            except Exception:
                sigma_seed = None
                result.at[idx] = np.nan

    return result


def _apply_underlying_pass(
    df: pd.DataFrame,
    *,
    carry_by_expiry: dict[object, float],
    config: PipelineConfig,
) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)

    for expiration, g in df.groupby("expiration", sort=False, dropna=False):
        carry = carry_by_expiry.get(expiration, np.nan)
        is_call = g["option_type"].eq("C")
        out.loc[g.index[is_call]] = g.loc[is_call, "mid_px"].to_numpy(dtype=float)

        put_side = g.loc[~is_call]
        if put_side.empty:
            continue
        carry_to_use = 0.0 if not np.isfinite(carry) else float(carry)
        out.loc[put_side.index] = _deamericanize_side(
            put_side,
            carry=carry_to_use,
            config=config,
        )

    return out


def _apply_letf_pass(
    df: pd.DataFrame,
    *,
    carry_by_expiry: dict[object, float],
    config: PipelineConfig,
) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)

    for expiration, g in df.groupby("expiration", sort=False, dropna=False):
        carry = carry_by_expiry.get(expiration, np.nan)
        if not np.isfinite(carry):
            continue
        out.loc[g.index] = _deamericanize_side(g, carry=float(carry), config=config)

    return out


def _process_underlying_trade_date(df: pd.DataFrame, *, config: PipelineConfig) -> pd.DataFrame:
    out = df.copy()

    # pass 1: calls unchanged, puts de-Americanized with zero extra carry
    pass1 = _apply_underlying_pass(out, carry_by_expiry={}, config=config)
    out["_mid_px_eu_pass1"] = pass1

    # raw expiry forwards from pass-1 pseudo-European chain
    expiry_rows: list[dict[str, float | object]] = []
    for expiration, g in out.groupby("expiration", sort=True, dropna=False):
        info = _estimate_raw_forward_for_expiry(
            g,
            price_col="_mid_px_eu_pass1",
            n_atm_pairs=config.forward_n_atm_pairs,
            mad_z=config.forward_mad_z,
        )
        info["expiration"] = expiration
        expiry_rows.append(info)

    expiry_table = pd.DataFrame(expiry_rows)
    if expiry_table.empty:
        out["carry"] = np.nan
        out["mid_px_eu"] = out["_mid_px_eu_pass1"]
        out["forward_factor"] = np.nan
        out["forward_price"] = np.nan
        out["Log_Moneyness"] = np.nan
        out["Moneyness"] = np.nan
        return out.drop(columns=["_mid_px_eu_pass1"])

    expiry_table = expiry_table.sort_values("tau", kind="mergesort").reset_index(drop=True)
    tau_nodes = expiry_table["tau"].to_numpy(dtype=float)
    y = expiry_table["raw_log_forward"].to_numpy(dtype=float)
    weights = expiry_table["pair_count_used"].to_numpy(dtype=float)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.where(weights > 0.0, weights, 0.0)
    positive_weights = weights[weights > 0.0]
    if positive_weights.size:
        weights = weights / max(float(np.nanmedian(positive_weights)), 1.0)

    x_fit = _fit_smoothed_log_forward(
        tau_nodes,
        y,
        weights,
        curve_smooth=config.curve_smooth,
        monotone_penalty=config.monotone_penalty,
        iterations=config.smooth_iterations,
        monotone_mode="increasing",
    )

    expiry_table["fitted_log_forward"] = x_fit
    expiry_table["carry"] = expiry_table["r"] - expiry_table["fitted_log_forward"] / expiry_table["tau"]
    expiry_table["forward_factor"] = np.exp(expiry_table["fitted_log_forward"])

    carry_by_expiry = dict(zip(expiry_table["expiration"], expiry_table["carry"]))
    ff_by_expiry = dict(zip(expiry_table["expiration"], expiry_table["forward_factor"]))

    # pass 2: calls unchanged, puts de-Americanized with fitted expiry carry
    out["mid_px_eu"] = _apply_underlying_pass(out, carry_by_expiry=carry_by_expiry, config=config)
    out["carry"] = out["expiration"].map(carry_by_expiry)
    out["forward_factor"] = out["expiration"].map(ff_by_expiry)
    out["forward_price"] = out["underlying_price"] * out["forward_factor"]
    out["Log_Moneyness"] = np.log(out["strike_price"] / out["forward_price"])
    out["Moneyness"] = out["strike_price"] / out["forward_price"]

    return out.drop(columns=["_mid_px_eu_pass1"])


def _process_letf_trade_date(df: pd.DataFrame, *, config: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    carry_floor = config.letf_carry_floor

    seed_rows: list[dict[str, float | object]] = []
    for expiration, g in out.groupby("expiration", sort=True, dropna=False):
        tau = float(np.nanmedian(g["tau"]))
        r = float(np.nanmedian(g["r"]))
        spot = float(np.nanmedian(g["underlying_price"]))
        raw_carry, raw_point_count = _estimate_letf_carry_for_expiry(g)
        seed_carry = _clip_carry_to_floor(raw_carry, carry_floor)
        seed_rows.append(
            {
                "expiration": expiration,
                "tau": tau,
                "r": r,
                "spot": spot,
                "raw_carry": raw_carry,
                "seed_carry": seed_carry,
                "raw_point_count": float(raw_point_count),
            }
        )

    seed_table = pd.DataFrame(seed_rows)
    if seed_table.empty:
        out["carry"] = np.nan
        out["mid_px_eu"] = np.nan
        out["forward_factor"] = np.nan
        out["forward_price"] = np.nan
        out["Log_Moneyness"] = np.nan
        out["Moneyness"] = np.nan
        return out

    seed_table = seed_table.sort_values("tau", kind="mergesort").reset_index(drop=True)
    seed_carry_by_expiry = dict(zip(seed_table["expiration"], seed_table["seed_carry"]))

    # pass 1: de-Americanize both calls and puts with raw expiry carry seeds
    out["_mid_px_eu_pass1"] = _apply_letf_pass(out, carry_by_expiry=seed_carry_by_expiry, config=config)

    # raw expiry forwards from the pass-1 pseudo-European chain
    expiry_rows: list[dict[str, float | object]] = []
    for expiration, g in out.groupby("expiration", sort=True, dropna=False):
        info = _estimate_raw_forward_for_expiry(
            g,
            price_col="_mid_px_eu_pass1",
            n_atm_pairs=config.forward_n_atm_pairs,
            mad_z=config.forward_mad_z,
        )

        seed_row = seed_table.loc[seed_table["expiration"].eq(expiration)].iloc[0]
        seed_carry = float(seed_row["seed_carry"])
        tau = float(seed_row["tau"])
        r = float(seed_row["r"])
        spot = float(seed_row["spot"])

        if (not np.isfinite(info.get("raw_log_forward", np.nan))) and np.isfinite(seed_carry):
            raw_log_forward = (r - seed_carry) * tau
            info["raw_log_forward"] = raw_log_forward
            info["raw_forward"] = spot * math.exp(raw_log_forward) if np.isfinite(spot) and spot > 0.0 else np.nan

        if carry_floor is not None and np.isfinite(carry_floor) and np.isfinite(info.get("raw_log_forward", np.nan)):
            forward_cap = (r - float(carry_floor)) * tau
            info["raw_log_forward"] = min(float(info["raw_log_forward"]), forward_cap)
            if np.isfinite(spot) and spot > 0.0:
                info["raw_forward"] = spot * math.exp(float(info["raw_log_forward"]))

        info["expiration"] = expiration
        info["seed_carry"] = seed_carry
        info["raw_point_count"] = float(seed_row["raw_point_count"])
        expiry_rows.append(info)

    expiry_table = pd.DataFrame(expiry_rows)
    if expiry_table.empty:
        out["carry"] = np.nan
        out["mid_px_eu"] = out["_mid_px_eu_pass1"]
        out["forward_factor"] = np.nan
        out["forward_price"] = np.nan
        out["Log_Moneyness"] = np.nan
        out["Moneyness"] = np.nan
        return out.drop(columns=["_mid_px_eu_pass1"])

    expiry_table = expiry_table.sort_values("tau", kind="mergesort").reset_index(drop=True)
    tau_nodes = expiry_table["tau"].to_numpy(dtype=float)
    raw_log_forward = expiry_table["raw_log_forward"].to_numpy(dtype=float)
    r_nodes = expiry_table["r"].to_numpy(dtype=float)
    seed_carry_nodes = expiry_table["seed_carry"].to_numpy(dtype=float)
    pair_count_nodes = expiry_table["pair_count_used"].to_numpy(dtype=float)
    raw_point_count_nodes = expiry_table["raw_point_count"].to_numpy(dtype=float)

    weights = _build_letf_fit_weights(
        tau_nodes,
        pair_count_nodes,
        raw_point_count_nodes,
        config=config,
    )
    monotone_mode = _infer_monotone_mode_from_carry(
        r_nodes,
        seed_carry_nodes,
        requested_mode=config.letf_monotone_mode,
    )

    x_fit = _fit_smoothed_log_forward(
        tau_nodes,
        raw_log_forward,
        weights,
        curve_smooth=config.curve_smooth,
        monotone_penalty=config.monotone_penalty,
        iterations=config.smooth_iterations,
        monotone_mode=monotone_mode,
    )

    fitted_carry = expiry_table["r"].to_numpy(dtype=float) - x_fit / tau_nodes
    if carry_floor is not None and np.isfinite(carry_floor):
        fitted_carry = np.maximum(fitted_carry, float(carry_floor))
        x_fit = (expiry_table["r"].to_numpy(dtype=float) - fitted_carry) * tau_nodes

    expiry_table["fitted_log_forward"] = x_fit
    expiry_table["carry"] = fitted_carry
    expiry_table["forward_factor"] = np.exp(expiry_table["fitted_log_forward"])

    carry_by_expiry = dict(zip(expiry_table["expiration"], expiry_table["carry"]))
    ff_by_expiry = dict(zip(expiry_table["expiration"], expiry_table["forward_factor"]))

    # pass 2: de-Americanize both calls and puts with the fitted expiry carry
    out["mid_px_eu"] = _apply_letf_pass(out, carry_by_expiry=carry_by_expiry, config=config)
    out["carry"] = out["expiration"].map(carry_by_expiry)
    out["forward_factor"] = out["expiration"].map(ff_by_expiry)
    out["forward_price"] = out["underlying_price"] * out["forward_factor"]
    out["Log_Moneyness"] = np.log(out["strike_price"] / out["forward_price"])
    out["Moneyness"] = out["strike_price"] / out["forward_price"]
    return out.drop(columns=["_mid_px_eu_pass1"])


def _process_trade_date_group(df: pd.DataFrame, *, config: PipelineConfig) -> pd.DataFrame:
    if config.branch == "underlying":
        return _process_underlying_trade_date(df, config=config)
    if config.branch == "letf":
        return _process_letf_trade_date(df, config=config)
    raise ValueError(f"Unsupported branch: {config.branch}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    data: pd.DataFrame | str | Path,
    *,
    branch: Branch,
    ticker: str | None = None,
    input_sheet: str | None = None,
    output_path: str | Path | None = None,
    output_sheet: str = "deamericanized",
    day_count: int = 365,
    time_col_is_years: bool = False,
    tree_dt_target: float = 8e-4,
    tree_min_steps: int = 1000,
    forward_n_atm_pairs: int = 3,
    forward_mad_z: float = 5.0,
    curve_smooth: float = 1e-6,
    monotone_penalty: float = 1e-4,
    smooth_iterations: int = 8,
    letf_carry_floor: float | None = None,
    letf_exclude_long_end_from_fit: bool = False,
    letf_long_end_cutoff_days: float | None = None,
    letf_monotone_mode: MonotoneMode = "none",
) -> pd.DataFrame:
    """Run the de-Americanization pipeline.

    Parameters
    ----------
    data:
        Input DataFrame or path to an .xlsx workbook.
    branch:
        "underlying" for the plain no-dividend branch or "letf" for the LETF branch.
    ticker:
        Optional ticker filter applied to the canonical `underlying` column.
    output_path:
        Optional Output reverse Loop workbook path. When omitted, the function only returns a DataFrame.

    Returns
    -------
    pandas.DataFrame
        Original input columns plus:
        carry, forward_factor, mid_px_eu, Log_Moneyness, Moneyness, forward_price, tau
    """
    config = PipelineConfig(
        branch=branch,
        ticker=ticker,
        input_sheet=input_sheet,
        output_sheet=output_sheet,
        day_count=day_count,
        time_col_is_years=time_col_is_years,
        tree_dt_target=tree_dt_target,
        tree_min_steps=tree_min_steps,
        forward_n_atm_pairs=forward_n_atm_pairs,
        forward_mad_z=forward_mad_z,
        curve_smooth=curve_smooth,
        monotone_penalty=monotone_penalty,
        smooth_iterations=smooth_iterations,
        letf_carry_floor=letf_carry_floor,
        letf_exclude_long_end_from_fit=letf_exclude_long_end_from_fit,
        letf_long_end_cutoff_days=letf_long_end_cutoff_days,
        letf_monotone_mode=letf_monotone_mode,
    )

    if isinstance(data, (str, Path)):
        input_path = Path(data).expanduser().resolve()
        sheet_name = 0 if input_sheet is None else input_sheet
        raw = pd.read_excel(input_path, sheet_name=sheet_name)
    else:
        raw = data.copy()

    if isinstance(raw, dict):
        raw = next(iter(raw.values())).copy()

    canonical, original_columns = _validate_and_canonicalize(raw)
    work = _prepare_base_frame(canonical, config=config)

    processed_parts: list[pd.DataFrame] = []
    for _, group in work.groupby(GROUP_KEYS, sort=False, dropna=False):
        processed_parts.append(_process_trade_date_group(group.copy(), config=config))

    if processed_parts:
        processed = pd.concat(processed_parts, axis=0, ignore_index=False)
    else:
        processed = work.copy()
        for col in OUTPUT_ADDITIONAL_COLUMNS:
            processed[col] = np.nan

    processed = processed.sort_values("_row_id", kind="mergesort")

    final = raw.copy().iloc[processed["_row_id"].to_numpy()].copy()
    for col in OUTPUT_ADDITIONAL_COLUMNS:
        final[col] = processed[col].to_numpy()
    final = final.loc[:, original_columns + OUTPUT_ADDITIONAL_COLUMNS]
    final = final.reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path).expanduser().resolve()
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            final.to_excel(writer, sheet_name=output_sheet, index=False)

    return final


def run_underlying_pipeline(
    data: pd.DataFrame | str | Path,
    **kwargs,
) -> pd.DataFrame:
    return run_pipeline(data, branch="underlying", **kwargs)


def run_letf_pipeline(
    data: pd.DataFrame | str | Path,
    **kwargs,
) -> pd.DataFrame:
    return run_pipeline(data, branch="letf", **kwargs)
