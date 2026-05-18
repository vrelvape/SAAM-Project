import pandas as pd


def load_raw_datasets(data_raw_path):
    """
    Load the raw datasets required for Part I of the project.
    """

    static = pd.read_excel(data_raw_path / "Static_2025.xlsx")
    prices_raw = pd.read_excel(data_raw_path / "DS_RI_T_USD_M_2025.xlsx")
    market_caps_raw = pd.read_excel(data_raw_path / "DS_MV_T_USD_M_2025.xlsx")
    carbon_raw = pd.read_excel(data_raw_path / "DS_CO2_SCOPE_1_Y_2025.xlsx")

    return static, prices_raw, market_caps_raw, carbon_raw