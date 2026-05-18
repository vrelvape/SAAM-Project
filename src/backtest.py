import pandas as pd


def backtest_dynamic_portfolio(returns_matrix, weights, year):
    """
    Compute monthly portfolio returns over a given calendar year.

    The portfolio starts the year with weights estimated at the end of the
    previous year. Between rebalancing dates, weights evolve dynamically
    according to realized asset returns.
    """

    returns_year = returns_matrix.loc[weights.index, f"{year}-01-01":f"{year}-12-31"]

    current_weights = weights.copy()
    portfolio_returns = []

    for date in returns_year.columns:
        month_returns = returns_year[date]

        # Keep only assets with available returns for the month
        valid_mask = month_returns.notna()
        month_returns = month_returns[valid_mask]
        current_weights = current_weights[valid_mask]

        # Renormalize weights before computing portfolio return
        current_weights = current_weights / current_weights.sum()

        portfolio_return = (current_weights * month_returns).sum()
        portfolio_returns.append(portfolio_return)

        # Update weights dynamically within the year
        current_weights = current_weights * (1 + month_returns)
        current_weights = current_weights / current_weights.sum()

    portfolio_returns = pd.Series(
        portfolio_returns,
        index=returns_year.columns,
        name=f"mv_returns_{year}"
    )

    return portfolio_returns

def run_mv_backtest(returns_matrix, mv_weights_by_year, rebalance_years):
    """
    Run the minimum variance backtest year by year and return the full
    out-of-sample monthly return series.
    """

    mv_returns_by_year = {}

    for year in rebalance_years:
        weights = mv_weights_by_year[year]
        realized_returns = backtest_dynamic_portfolio(
            returns_matrix=returns_matrix,
            weights=weights,
            year=year
        )
        mv_returns_by_year[year] = realized_returns

    mv_returns_oos = pd.concat(mv_returns_by_year.values()).sort_index()

    return mv_returns_by_year, mv_returns_oos