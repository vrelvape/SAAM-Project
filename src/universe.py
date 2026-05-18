import pandas as pd


def is_eligible(series, min_history_months, stale_return_threshold, price_available, carbon_available):
    """
    Determine whether an asset is eligible for portfolio construction.
    """

    if series.count() < min_history_months:
        return False

    zero_ratio = (series == 0).mean()
    if zero_ratio > stale_return_threshold:
        return False

    if not price_available:
        return False

    if not carbon_available:
        return False

    return True


def build_universe_by_year(
    returns_matrix,
    price_data,
    carbon_data,
    rebalance_years,
    rolling_window_months,
    min_history_months,
    stale_return_threshold,
):
    """
    Build the dynamic investment universe year by year.
    """

    universe_by_year = {}

    for year in rebalance_years:
        end = pd.Timestamp(f"{year-1}-12-31")
        window = returns_matrix.loc[:, :end].iloc[:, -rolling_window_months:]

        eligible_assets = []

        for isin in window.index:
            series = window.loc[isin]

            cond_price = False
            if isin in price_data.index:
                available_price_dates = price_data.columns[price_data.columns <= end]
                if len(available_price_dates) > 0:
                    last_price_date = available_price_dates.max()
                    cond_price = pd.notna(price_data.loc[isin, last_price_date])

            cond_carbon = False
            if isin in carbon_data.index and end in carbon_data.columns:
                cond_carbon = pd.notna(carbon_data.loc[isin, end])

            if is_eligible(
                series=series,
                min_history_months=min_history_months,
                stale_return_threshold=stale_return_threshold,
                price_available=cond_price,
                carbon_available=cond_carbon,
            ):
                eligible_assets.append(isin)

        universe_by_year[year] = eligible_assets

    return universe_by_year
