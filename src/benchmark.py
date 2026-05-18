import numpy as np
import pandas as pd


def backtest_value_weighted_portfolio(returns_matrix, market_caps_data, universe, year):
    """
    Compute monthly returns of the value-weighted benchmark portfolio
    using lagged market capitalization weights.
    """

    returns_year = returns_matrix.loc[universe, f"{year}-01-01":f"{year}-12-31"]

    portfolio_returns = []

    for date in returns_year.columns:
        previous_dates = market_caps_data.columns[market_caps_data.columns < date]

        if len(previous_dates) == 0:
            portfolio_returns.append(np.nan)
            continue

        prev_date = previous_dates.max()

        caps = market_caps_data.loc[universe, prev_date]
        caps = pd.to_numeric(caps, errors="coerce")

        month_returns = returns_matrix.loc[universe, date]

        # Keep only assets with both market cap and return available
        valid_mask = caps.notna() & month_returns.notna()
        caps = caps[valid_mask]
        month_returns = month_returns[valid_mask]

        if caps.empty or caps.sum() <= 0:
            portfolio_returns.append(np.nan)
            continue

        weights = caps / caps.sum()
        portfolio_return = (weights * month_returns).sum()

        portfolio_returns.append(portfolio_return)

    portfolio_returns = pd.Series(
        portfolio_returns,
        index=returns_year.columns,
        name=f"vw_returns_{year}"
    )

    return portfolio_returns


def run_vw_backtest(returns_matrix, market_caps_data, universe_by_year, rebalance_years):
    """
    Run the value-weighted benchmark backtest year by year and return the full
    out-of-sample monthly return series.
    """

    vw_returns_by_year = {}

    for year in rebalance_years:
        universe = universe_by_year[year]
        realized_returns = backtest_value_weighted_portfolio(
            returns_matrix=returns_matrix,
            market_caps_data=market_caps_data,
            universe=universe,
            year=year
        )
        vw_returns_by_year[year] = realized_returns

    vw_returns_oos = pd.concat(vw_returns_by_year.values()).sort_index()

    return vw_returns_by_year, vw_returns_oos