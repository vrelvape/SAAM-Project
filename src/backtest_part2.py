"""
backtest_part2.py — Part II: Backtest utilities for carbon-constrained portfolios.

Reuses the same backtest logic as Part I (dynamic weight update within the year)
applied to the new carbon-constrained portfolios.
"""

import pandas as pd


def backtest_dynamic_portfolio(returns_matrix, weights, year):
    """
    Compute monthly portfolio returns over a given calendar year.

    Identical to Part I's implementation; reproduced here for self-containment.
    """
    returns_year = returns_matrix.loc[weights.index, f"{year}-01-01":f"{year}-12-31"]

    current_weights = weights.copy()
    portfolio_returns = []

    for date in returns_year.columns:
        month_returns = returns_year[date]

        valid_mask = month_returns.notna()
        month_returns = month_returns[valid_mask]
        current_weights = current_weights[valid_mask]

        current_weights = current_weights / current_weights.sum()

        portfolio_return = (current_weights * month_returns).sum()
        portfolio_returns.append(portfolio_return)

        current_weights = current_weights * (1 + month_returns)
        current_weights = current_weights / current_weights.sum()

    return pd.Series(
        portfolio_returns,
        index=returns_year.columns,
    )


def run_portfolio_backtest(returns_matrix, weights_by_year, rebalance_years, label="portfolio"):
    """
    Run a dynamic backtest for any portfolio defined by annual weights.

    Parameters
    ----------
    returns_matrix : pd.DataFrame  (ISIN x monthly dates)
    weights_by_year : dict {year -> pd.Series of weights}
    rebalance_years : list of ints
    label : str — used for progress messages

    Returns
    -------
    returns_by_year : dict {year -> pd.Series}
    returns_oos : pd.Series — full out-of-sample monthly return series
    """
    returns_by_year = {}

    for year in rebalance_years:
        weights = weights_by_year[year]
        realized = backtest_dynamic_portfolio(returns_matrix, weights, year)
        returns_by_year[year] = realized

    returns_oos = pd.concat(returns_by_year.values()).sort_index()

    return returns_by_year, returns_oos


# ---------------------------------------------------------------------------
# Carbon metric time series
# ---------------------------------------------------------------------------

def compute_carbon_metrics_timeseries(
    weights_by_year,
    universe_by_year,
    carbon_data,
    annual_caps_data,
    carbon_intensity,
    rebalance_years,
):
    """
    Compute WACI and CF for each rebalance year, given portfolio weights.

    Returns
    -------
    waci_series : pd.Series  indexed by year
    cf_series   : pd.Series  indexed by year
    """
    from src.carbon import (
        compute_portfolio_cf,
        compute_portfolio_waci,
        compute_cf_per_unit,
    )

    waci_vals = {}
    cf_vals = {}

    for year in rebalance_years:
        weights = weights_by_year[year]
        universe = universe_by_year[year]
        end_prev = pd.Timestamp(f"{year - 1}-12-31")

        cf_pu = compute_cf_per_unit(carbon_data, annual_caps_data, universe, end_prev)
        cf = compute_portfolio_cf(weights, cf_pu)
        waci = compute_portfolio_waci(weights, carbon_intensity, universe, end_prev)

        cf_vals[year] = cf
        waci_vals[year] = waci

    return pd.Series(waci_vals), pd.Series(cf_vals)


def compute_vw_carbon_metrics_timeseries(
    market_caps_data,
    universe_by_year,
    carbon_data,
    annual_caps_data,
    carbon_intensity,
    rebalance_years,
):
    """
    Compute WACI and CF for the value-weighted benchmark over all years.
    """
    from src.carbon import compute_vw_cf, compute_portfolio_waci
    from src.optimization_part2 import _get_vw_weights

    waci_vals = {}
    cf_vals = {}

    for year in rebalance_years:
        universe = universe_by_year[year]
        end_prev = pd.Timestamp(f"{year - 1}-12-31")

        vw_weights = _get_vw_weights(market_caps_data, universe, end_prev)

        cf = compute_vw_cf(carbon_data, annual_caps_data, universe, end_prev)
        waci = compute_portfolio_waci(vw_weights, carbon_intensity, universe, end_prev)

        cf_vals[year] = cf
        waci_vals[year] = waci

    return pd.Series(waci_vals), pd.Series(cf_vals)
