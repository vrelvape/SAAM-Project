"""
carbon.py — Part II: Carbon data loading and metrics computation.

Implements:
  - Revenue and annual market cap data loading
  - Carbon intensity (CI)
  - Weighted Average Carbon Intensity (WACI)
  - Carbon Footprint (CF) for any portfolio
  - Carbon Footprint for the value-weighted benchmark
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def prepare_revenue_data(revenue_raw, region_isins):
    """
    Prepare the annual revenue matrix (in thousands USD) for the regional universe.
    Applies forward-fill for missing interior/trailing values.
    """
    rev = revenue_raw.copy()

    if "ISIN" in rev.columns:
        rev = rev.set_index("ISIN")

    rev = rev[rev.index.notna()]

    if "NAME" in rev.columns:
        rev = rev.drop(columns="NAME")

    rev_data = rev.loc[rev.index.isin(region_isins)].copy()
    rev_data = rev_data.apply(pd.to_numeric, errors="coerce")

    # Convert year columns to end-of-year timestamps
    rev_data.columns = pd.to_datetime(rev_data.columns.astype(str), format="%Y")
    rev_data.columns = rev_data.columns + pd.offsets.YearEnd(0)
    rev_data = rev_data.sort_index(axis=1)

    # Forward-fill: missing interior/trailing values replaced by previous year
    rev_data = rev_data.ffill(axis=1)

    return rev_data


def prepare_annual_market_caps(annual_caps_raw, region_isins):
    """
    Prepare the annual market capitalization matrix (in million USD) for the regional universe.
    Applies forward-fill for missing interior/trailing values.
    """
    caps = annual_caps_raw.copy()

    if "ISIN" in caps.columns:
        caps = caps.set_index("ISIN")

    caps = caps[caps.index.notna()]

    if "NAME" in caps.columns:
        caps = caps.drop(columns="NAME")

    caps_data = caps.loc[caps.index.isin(region_isins)].copy()
    caps_data = caps_data.apply(pd.to_numeric, errors="coerce")

    # Convert year columns to end-of-year timestamps
    caps_data.columns = pd.to_datetime(caps_data.columns.astype(str), format="%Y")
    caps_data.columns = caps_data.columns + pd.offsets.YearEnd(0)
    caps_data = caps_data.sort_index(axis=1)

    caps_data = caps_data.ffill(axis=1)

    return caps_data


# ---------------------------------------------------------------------------
# Carbon intensity
# ---------------------------------------------------------------------------

def compute_carbon_intensity(carbon_data, revenue_data):
    """
    Compute carbon intensity for each firm and year.

    CI_{i,Y} = E_{i,Y} / (Rev_{i,Y} / 1000)

    where:
      E_{i,Y}   — CO2 emissions in tonnes
      Rev_{i,Y} — revenues in thousands USD  =>  Rev/1000 in millions USD

    Result: CI in tonnes CO2 / million USD of revenue.

    NaN when either emission or revenue is missing or revenue is zero.
    """
    # Align on common ISINs and dates
    common_isins = carbon_data.index.intersection(revenue_data.index)
    common_dates = carbon_data.columns.intersection(revenue_data.columns)

    E = carbon_data.loc[common_isins, common_dates]
    Rev_millions = revenue_data.loc[common_isins, common_dates] / 1000.0

    # Avoid division by zero or negative revenues
    Rev_millions = Rev_millions.where(Rev_millions > 0, other=np.nan)

    ci = E / Rev_millions
    return ci


# ---------------------------------------------------------------------------
# Portfolio carbon metrics
# ---------------------------------------------------------------------------

def compute_cf_per_unit(carbon_data, annual_caps_data, universe, end_year_date):
    """
    Compute the carbon footprint contribution per unit of weight for each firm.

    cf_per_unit_{i} = E_{i,Y} / Cap_{i,Y}

    where Cap is in million USD, E in tonnes CO2.
    Result: tonnes CO2 / million USD invested.

    Returns a pd.Series indexed by ISIN, restricted to `universe`.
    Missing values (no emissions or no market cap) are returned as NaN.
    """
    isins = [i for i in universe if i in carbon_data.index and i in annual_caps_data.index]

    if end_year_date not in carbon_data.columns or end_year_date not in annual_caps_data.columns:
        return pd.Series(np.nan, index=isins)

    E = carbon_data.loc[isins, end_year_date]
    Cap = annual_caps_data.loc[isins, end_year_date]

    # Cap must be strictly positive
    Cap = Cap.where(Cap > 0, other=np.nan)

    cf_pu = E / Cap
    return cf_pu


def compute_portfolio_cf(weights, cf_per_unit):
    """
    Compute the carbon footprint of a portfolio.

    CF^(p)_Y = sum_i( alpha_{i,Y} * E_{i,Y} / Cap_{i,Y} )

    Parameters
    ----------
    weights : pd.Series  (indexed by ISIN, must sum to 1)
    cf_per_unit : pd.Series  (indexed by ISIN)

    Returns
    -------
    float  — tonnes CO2 per million USD invested
    """
    common = weights.index.intersection(cf_per_unit.index)
    cf = (weights.loc[common] * cf_per_unit.loc[common]).sum()
    return cf


def compute_vw_cf(carbon_data, annual_caps_data, universe, end_year_date):
    """
    Compute the carbon footprint of the value-weighted benchmark.

    CF^(vw)_Y = (1 / Cap_Y) * sum_i( E_{i,Y} )

    where Cap_Y = sum_i( Cap_{i,Y} ).

    Returns
    -------
    float  — tonnes CO2 per million USD invested, or NaN if data is unavailable
    """
    isins = [i for i in universe if i in carbon_data.index and i in annual_caps_data.index]

    if end_year_date not in carbon_data.columns or end_year_date not in annual_caps_data.columns:
        return np.nan

    E = carbon_data.loc[isins, end_year_date].fillna(0)
    Cap = annual_caps_data.loc[isins, end_year_date]
    Cap = Cap.where(Cap > 0, other=np.nan)

    total_cap = Cap.sum(skipna=True)
    if total_cap <= 0:
        return np.nan

    return E.sum() / total_cap


def compute_portfolio_waci(weights, carbon_intensity, universe, end_year_date):
    """
    Compute the Weighted Average Carbon Intensity of a portfolio.

    WACI^(p)_Y = sum_i( alpha_{i,Y} * CI_{i,Y} )

    Parameters
    ----------
    weights : pd.Series  (indexed by ISIN)
    carbon_intensity : pd.DataFrame  (ISIN x date)
    universe : list of ISINs
    end_year_date : pd.Timestamp

    Returns
    -------
    float
    """
    if end_year_date not in carbon_intensity.columns:
        return np.nan

    ci_year = carbon_intensity.loc[
        carbon_intensity.index.isin(universe), end_year_date
    ]

    common = weights.index.intersection(ci_year.index)
    waci = (weights.loc[common] * ci_year.loc[common]).sum()
    return waci


# ---------------------------------------------------------------------------
# Top emitters
# ---------------------------------------------------------------------------

def get_top_carbon_emitters(weights, cf_per_unit, static_df, n=10):
    """
    Return the top-n firms ranked by weighted carbon contribution.

    Parameters
    ----------
    weights : pd.Series (ISIN-indexed)
    cf_per_unit : pd.Series (ISIN-indexed)
    static_df : pd.DataFrame with columns ['ISIN', 'NAME'] (or similar)
    n : int

    Returns
    -------
    pd.DataFrame
    """
    common = weights.index.intersection(cf_per_unit.index)
    contribution = weights.loc[common] * cf_per_unit.loc[common]
    top = contribution.nlargest(n).reset_index()
    top.columns = ["ISIN", "Weighted_CF_Contribution"]

    if "ISIN" in static_df.columns and "NAME" in static_df.columns:
        name_map = static_df.set_index("ISIN")["NAME"].to_dict()
        top["Name"] = top["ISIN"].map(name_map)

    top["Weight"] = weights.loc[top["ISIN"].values].values
    top["CF_per_unit"] = cf_per_unit.loc[top["ISIN"].values].values

    return top
