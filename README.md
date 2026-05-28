# SAAM Project - Group N

This repository contains our complete project for Group N, completed as
part of the Sustainability Aware Asset Management course at HEC
Lausanne (2026).

The project investigates whether carbon-aware portfolio construction can reduce
a portfolio's carbon footprint without significantly sacrificing financial
performance. It does so by comparing standard and carbon-constrained portfolio
strategies over a 2014–2025 out-of-sample period, using a universe of Emerging
Markets equities.

Group N strategy:

```text
Region: Emerging Markets
Carbon scope: Scope 1 emissions
```

---

## Structure

```text
data_raw/             Original Excel files from Datastream and course material
data_processed/       Space kept for possible cleaned datasets
outputs/              Tables, figures, and Excel files produced by the code
outputs/tables/       CSV tables used for reporting and checks
outputs/figures/      Saved plots
outputs/excel/        Filled Excel templates
outputs/intermediate/ Temporary files generated while the code runs
resources/templates/  Official course templates
src/                  Reusable Python modules used by the notebook and main.py
```

The main idea is simple: reusable code lives in `src/`. Both the notebook and
`main.py` import from there, so the two entry points always stay aligned.

---

## Main Notebook

The final notebook is located at:

```text
notebooks/SAAM_Project_GroupN.ipynb
```

It is self-contained and reproduces all results shown in the report when run
from top to bottom, with no manual intervention. It combines:

- **Part I**: standard portfolio allocation — minimum-variance and
  value-weighted portfolios, out-of-sample backtest, and performance analysis.
- **Part II**: carbon-aware portfolio allocation — carbon intensity and
  footprint metrics, carbon-constrained optimization, tracking-error
  minimization under a carbon budget, and a net-zero glide path.

---

## Data Inputs

All raw files should be placed in:

```text
data_raw/
```

The project uses the following Datastream extracts and course files:

```text
Static_2025.xlsx           Firm-level metadata (ISIN, name, country, region)
DS_RI_T_USD_M_2025.xlsx    Monthly total return index
DS_MV_T_USD_M_2025.xlsx    Monthly market capitalization
DS_RI_T_USD_Y_2025.xlsx    Annual total return index (used for Part II)
DS_MV_T_USD_Y_2025.xlsx    Annual market capitalization (used for Part II)
DS_CO2_SCOPE_1_Y_2025.xlsx Annual Scope 1 CO2 emissions
DS_REV_Y_2025.xlsx         Annual revenues
Risk_Free_Rate_2025.xlsx   Monthly risk-free rate (used for Sharpe ratios)
```

For Group N, the static universe is filtered to Emerging Markets firms and
Scope 1 carbon emissions are used throughout. Scope 2 data is present in
`data_raw/` but is not used for our assigned strategy.

---

## Portfolios

The notebook constructs and compares five portfolios:

| Portfolio | Description |
|-----------|-------------|
| **MV** | Long-only minimum-variance portfolio |
| **VW** | Value-weighted benchmark |
| **MV(0.5)** | Minimum-variance portfolio with a 50% carbon-footprint reduction constraint |
| **VW(0.5)** | Tracking-error-minimizing portfolio with a 50% carbon-footprint reduction constraint |
| **VW(NZ)** | Net-zero portfolio following a 10% annual carbon-footprint reduction path |

MV and VW form the Part I baseline. The three carbon-aware portfolios in Part
II are designed to progressively reduce the portfolio's carbon footprint while
controlling for financial performance relative to the benchmark.

---

## Methodology Step By Step

The workflow follows the project statement, in this order:

1. Load the raw Datastream files from `data_raw/`.
2. Select the Group N universe: Emerging Markets firms.
3. Clean monthly total return index data and treat prices below 0.5 as missing.
4. Convert monthly price indexes into simple monthly returns.
5. Apply a delisting adjustment when a firm disappears from the sample.
6. Build a dynamic investment universe for each rebalance year.
7. Estimate covariance matrices using the previous 10 years of monthly returns.
8. Construct the long-only minimum-variance portfolio.
9. Construct the value-weighted benchmark.
10. Compute Part I performance statistics and fill the Excel template.
11. Load annual revenue and annual market capitalization data for Part II.
12. Compute carbon intensity, WACI, and carbon footprint.
13. Build the carbon-constrained minimum-variance portfolio.
14. Build the 50% carbon-reduction tracking-error portfolio.
15. Build the net-zero portfolio with a 10% annual reduction path.
16. Export all final tables and figures.

Sharpe ratios are computed using the standard excess-return formula:

```text
Sharpe ratio = annualized(Rp - Rf) / annualized volatility
```

where `Rf` comes from `Risk_Free_Rate_2025.xlsx`.

---

## Requirements

Python 3.12 or above is recommended. Install all dependencies with:

```zsh
pip install -r requirements.txt
```

Key libraries used: `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`.

---

## Run

The recommended entry point is the final notebook:

```text
notebooks/SAAM_Project_GroupN.ipynb
```

Open it and run all cells from top to bottom.

Alternatively, the complete pipeline can also be executed from the project root with:

```zsh
python main.py
```

The script prints progress messages along the way, which helps identify where
it is if a run takes time.

---

## Outputs

Generated outputs are saved under:

```text
outputs/tables/
outputs/figures/
outputs/excel/
```

The main files to look at after a run are:

```text
outputs/tables/portfolio_performance_summary.csv
outputs/tables/performance_summary_part2.csv
outputs/tables/carbon_metrics_summary_part2.csv
outputs/excel/Part_I_template_filled.xlsx
outputs/excel/SAAM_Project_GroupN_Final_Results.xlsx
```

---

## Code Map

```text
main.py                    Full pipeline: Part I followed by Part II
src/config.py              Main parameters: region, scope, dates, thresholds
src/paths.py               Centralized project paths
src/loaders.py             Raw Excel file loading
src/cleaning.py            Price, market cap, carbon data cleaning
src/universe.py            Dynamic investment universe construction
src/optimization.py        Standard minimum-variance optimization
src/backtest.py            Dynamic backtest for minimum-variance portfolios
src/benchmark.py           Value-weighted benchmark backtest
src/carbon.py              Carbon intensity, WACI, and carbon footprint metrics
src/optimization_part2.py  Carbon-constrained and tracking-error optimizations
src/backtest_part2.py      Backtests for Part II portfolios
src/reporting.py           Part I statistics, plots, and Excel template
src/reporting_part2.py     Part II statistics, carbon tables, plots, and exports
```

---

All results presented in the report were generated directly from this pipeline.
