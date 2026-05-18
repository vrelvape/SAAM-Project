import pandas as pd
import numpy as np

def prepare_price_data(prices_raw, region_isins):
    """
    Prepare the monthly price matrix for the selected regional universe.
    """

    prices = prices_raw.copy()

    # Use ISIN as identifier
    prices = prices.set_index("ISIN")

    # Remove invalid rows from Datastream exports
    prices = prices[prices.index.notna()]

    # Remove non-time-series column (NAME)
    if "NAME" in prices.columns:
        prices = prices.drop(columns="NAME")

    # Convert date columns to datetime and sort chronologically
    prices.columns = pd.to_datetime(prices.columns)
    prices = prices.sort_index(axis=1)

    # Restrict to the selected regional universe
    price_data = prices.loc[prices.index.isin(region_isins)].copy()

    # Ensure numeric values
    price_data = price_data.apply(pd.to_numeric, errors="coerce")

    return price_data


def prepare_market_caps_data(market_caps_raw, region_isins):
    """
    Prepare the monthly market capitalization matrix for the selected regional universe.
    """

    market_caps = market_caps_raw.copy()

    # Use ISIN as identifier
    market_caps = market_caps.set_index("ISIN")

    # Remove invalid rows from Datastream exports
    market_caps = market_caps[market_caps.index.notna()]

    # Remove non-time-series column (NAME)
    if "NAME" in market_caps.columns:
        market_caps = market_caps.drop(columns="NAME")

    # Convert date columns to datetime and sort chronologically
    market_caps.columns = pd.to_datetime(market_caps.columns)
    market_caps = market_caps.sort_index(axis=1)

    # Restrict to the selected regional universe
    market_caps_data = market_caps.loc[market_caps.index.isin(region_isins)].copy()

    # Ensure numeric values
    market_caps_data = market_caps_data.apply(pd.to_numeric, errors="coerce")

    return market_caps_data

def compute_returns(price_data):
    """
    Compute simple monthly returns.
    """
    returns = price_data.pct_change(axis=1, fill_method=None)
    return returns


def apply_low_price_filter(price_data, low_price_threshold):
    """
    Replace prices below threshold with NaN.
    """
    filtered_price_data = price_data.copy()
    filtered_price_data[filtered_price_data < low_price_threshold] = float("nan")
    return filtered_price_data

def apply_delisting_returns(price_data, returns):
    """
    Assign a -100% return to the first missing month after the last valid price
    when a firm disappears from the sample permanently.
    """
    adjusted_returns = returns.copy()

    for isin in price_data.index:
        prices = price_data.loc[isin]
        valid_mask = prices.notna()

        if valid_mask.any():
            last_valid_pos = np.where(valid_mask.values)[0][-1]

            if last_valid_pos < len(prices.index) - 1:
                trailing_block = prices.iloc[last_valid_pos + 1:]

                if trailing_block.isna().all():
                    first_missing_date = prices.index[last_valid_pos + 1]
                    adjusted_returns.loc[isin, first_missing_date] = -1.0

    return adjusted_returns

def prepare_carbon_data(carbon_raw, region_isins):
    """
    Prepare the annual carbon emissions matrix for the selected regional universe.
    """

    carbon = carbon_raw.copy()

    # Use ISIN as identifier if still present as a column
    if "ISIN" in carbon.columns:
        carbon = carbon.set_index("ISIN")

    # Remove invalid rows
    carbon = carbon[carbon.index.notna()]

    # Remove non-time-series column (NAME)
    if "NAME" in carbon.columns:
        carbon = carbon.drop(columns="NAME")

    # Restrict to the selected regional universe
    carbon_data = carbon.loc[carbon.index.isin(region_isins)].copy()

    # Convert values to numeric
    carbon_data = carbon_data.apply(pd.to_numeric, errors="coerce")

    # Convert annual columns to end-of-year datetime
    carbon_data.columns = pd.to_datetime(carbon_data.columns.astype(str), format="%Y")
    carbon_data.columns = carbon_data.columns + pd.offsets.YearEnd(0)

    # Sort chronologically
    carbon_data = carbon_data.sort_index(axis=1)

    # Forward-fill annual values across time
    carbon_data = carbon_data.ffill(axis=1)

    return carbon_data