"""
optimization_part2.py — Part II: Carbon-constrained portfolio optimization.

Implements:
  - Carbon-constrained minimum variance (Section 3.2)
  - Tracking-error minimization with carbon constraint (Section 3.3 and 4.1)
  - Year-by-year rolling versions of both
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Single-period optimizations
# ---------------------------------------------------------------------------

def compute_carbon_constrained_mv_weights(cov_matrix, cf_per_unit, cf_target):
    """
    Solve the carbon-constrained minimum variance problem (Section 3.2):

        min  alpha' Sigma alpha
        s.t. sum_i(alpha_i * cf_per_unit_i) <= cf_target
             sum(alpha) = 1
             alpha_i >= 0  for all i

    Parameters
    ----------
    cov_matrix : np.ndarray  (N x N)
    cf_per_unit : np.ndarray  (N,)  — E_i / Cap_i for each asset
    cf_target : float  — maximum allowed carbon footprint

    Returns
    -------
    np.ndarray (N,) — optimal weights, or None if infeasible
    """
    n = cov_matrix.shape[0]

    def objective(w):
        return w @ cov_matrix @ w

    constraints = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: cf_target - np.dot(w, cf_per_unit)},
    ]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not result.success:
        return None

    return result.x


def compute_tracking_error_min_weights(cov_matrix, vw_weights, cf_per_unit, cf_target):
    """
    Solve the tracking-error minimization with a carbon constraint (Section 3.3):

        min  (alpha - alpha_vw)' Sigma (alpha - alpha_vw)
        s.t. sum_i(alpha_i * cf_per_unit_i) <= cf_target
             sum(alpha) = 1
             alpha_i >= 0  for all i

    Parameters
    ----------
    cov_matrix : np.ndarray  (N x N)
    vw_weights : np.ndarray  (N,)  — value-weighted benchmark weights
    cf_per_unit : np.ndarray  (N,)
    cf_target : float

    Returns
    -------
    np.ndarray (N,) — optimal weights, or None if infeasible
    """
    n = cov_matrix.shape[0]

    def objective(w):
        diff = w - vw_weights
        return diff @ cov_matrix @ diff

    constraints = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: cf_target - np.dot(w, cf_per_unit)},
    ]
    bounds = [(0, 1)] * n
    w0 = vw_weights.copy()

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not result.success:
        return None

    return result.x


# ---------------------------------------------------------------------------
# Year-by-year rolling optimizations
# ---------------------------------------------------------------------------

def compute_carbon_mv_weights_by_year(
    returns_matrix,
    universe_by_year,
    mv_cf_by_year,
    carbon_data,
    annual_caps_data,
    rebalance_years,
    rolling_window_months,
    carbon_reduction_target=0.50,
    covariance_by_year=None,
    cf_per_unit_by_year=None,
):
    """
    Compute carbon-constrained MV weights year by year (Section 3.2).

    The CF target each year is:  0.5 * CF_oos_mv_Y

    Parameters
    ----------
    mv_cf_by_year : dict {year -> float}
        Carbon footprint of the unconstrained MV portfolio for each year.

    Returns
    -------
    dict {year -> pd.Series}  — weights indexed by ISIN, or fallback to
    unconstrained MV when the constraint is infeasible.
    """
    from src.carbon import compute_cf_per_unit

    weights_by_year = {}

    for year in rebalance_years:
        universe = universe_by_year[year]
        end_prev = pd.Timestamp(f"{year - 1}-12-31")

        print(f"[carbon-MV] {year}: {len(universe)} assets ...", flush=True)

        if covariance_by_year is None:
            window = returns_matrix.loc[universe, :end_prev].iloc[:, -rolling_window_months:]
            cov = window.T.cov().values
        else:
            cov = covariance_by_year[year].values

        if cf_per_unit_by_year is None:
            cf_pu = compute_cf_per_unit(carbon_data, annual_caps_data, universe, end_prev)
        else:
            cf_pu = cf_per_unit_by_year[year]
        cf_pu_values = cf_pu.reindex(universe).fillna(0).values

        # Target = 50% of unconstrained MV portfolio CF
        mv_cf = mv_cf_by_year.get(year, np.nan)
        if np.isnan(mv_cf) or mv_cf <= 0:
            print(f"  [carbon-MV] {year}: no valid MV CF — skipping constraint.", flush=True)
            from scipy.optimize import minimize as _min
            n = cov.shape[0]
            res = _min(
                lambda w: w @ cov @ w,
                np.ones(n) / n,
                method="SLSQP",
                bounds=[(0, 1)] * n,
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
            )
            weights_by_year[year] = pd.Series(res.x, index=universe)
            continue

        cf_target = carbon_reduction_target * mv_cf

        weights = compute_carbon_constrained_mv_weights(cov, cf_pu_values, cf_target)

        if weights is None:
            print(f"  [carbon-MV] {year}: optimization infeasible — using unconstrained MV.", flush=True)
            from scipy.optimize import minimize as _min
            n = cov.shape[0]
            res = _min(
                lambda w: w @ cov @ w,
                np.ones(n) / n,
                method="SLSQP",
                bounds=[(0, 1)] * n,
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
            )
            weights_by_year[year] = pd.Series(res.x, index=universe)
        else:
            weights_by_year[year] = pd.Series(weights, index=universe)

        print(f"  [carbon-MV] {year}: done.", flush=True)

    return weights_by_year


def compute_te_min_weights_by_year(
    returns_matrix,
    market_caps_data,
    universe_by_year,
    cf_target_by_year,
    carbon_data,
    annual_caps_data,
    rebalance_years,
    rolling_window_months,
    covariance_by_year=None,
    cf_per_unit_by_year=None,
    vw_weights_by_year=None,
):
    """
    Compute tracking-error minimizing weights year by year (Sections 3.3 and 4.1).

    Parameters
    ----------
    cf_target_by_year : dict {year -> float}
        The carbon footprint upper bound for each year.

    Returns
    -------
    dict {year -> pd.Series}
    """
    from src.carbon import compute_cf_per_unit

    weights_by_year = {}

    for year in rebalance_years:
        universe = universe_by_year[year]
        end_prev = pd.Timestamp(f"{year - 1}-12-31")

        print(f"[TE-min] {year}: {len(universe)} assets ...", flush=True)

        if covariance_by_year is None:
            window = returns_matrix.loc[universe, :end_prev].iloc[:, -rolling_window_months:]
            cov = window.T.cov().values
        else:
            cov = covariance_by_year[year].values

        if vw_weights_by_year is None:
            vw_caps = _get_vw_weights(market_caps_data, universe, end_prev)
        else:
            vw_caps = vw_weights_by_year[year]
        vw_weights_arr = vw_caps.reindex(universe).fillna(0).values

        if cf_per_unit_by_year is None:
            cf_pu = compute_cf_per_unit(carbon_data, annual_caps_data, universe, end_prev)
        else:
            cf_pu = cf_per_unit_by_year[year]
        cf_pu_values = cf_pu.reindex(universe).fillna(0).values

        cf_target = cf_target_by_year.get(year, np.nan)
        if np.isnan(cf_target):
            print(f"  [TE-min] {year}: no CF target — using pure TE minimization.", flush=True)
            cf_target = np.inf

        weights = compute_tracking_error_min_weights(cov, vw_weights_arr, cf_pu_values, cf_target)

        if weights is None:
            print(f"  [TE-min] {year}: optimization failed — falling back to VW.", flush=True)
            weights_by_year[year] = vw_caps
        else:
            weights_by_year[year] = pd.Series(weights, index=universe)

        print(f"  [TE-min] {year}: done.", flush=True)

    return weights_by_year


def _get_vw_weights(market_caps_data, universe, end_date):
    """
    Helper: compute value-weighted benchmark weights at a given date.
    Uses the last available market cap observation up to end_date.
    """
    available_dates = market_caps_data.columns[market_caps_data.columns <= end_date]
    if len(available_dates) == 0:
        return pd.Series(1.0 / len(universe), index=universe)

    last_date = available_dates.max()
    caps = market_caps_data.loc[
        market_caps_data.index.isin(universe), last_date
    ].apply(pd.to_numeric, errors="coerce")
    caps = caps.where(caps > 0, other=np.nan)
    total = caps.sum()

    if total <= 0 or np.isnan(total):
        return pd.Series(1.0 / len(universe), index=universe)

    return caps / total
