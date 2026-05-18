# SAAM Project - Group N

This repository contains our complete SAAM project for Group N.

Group N strategy:

```text
Region: Emerging Markets
Carbon scope: Scope 1 emissions
```

The project compares standard and carbon-aware portfolio strategies over the
2014-2025 out-of-sample period.

## Structure

```text
data_raw/             Original Excel files from Datastream and course material
data_processed/       Space kept for possible cleaned datasets
notebooks/            Jupyter notebooks, including the final notebook
outputs/              Tables, figures, and Excel files produced by the code
outputs/tables/       CSV tables used for reporting and checks
outputs/figures/      Saved plots
outputs/excel/        Filled Excel template for Part I
outputs/intermediate/ Temporary files generated while the code runs
resources/templates/  Official course templates
src/                  Reusable Python modules used by the notebook and main.py
```

The main idea is simple: reusable code lives in one place.

```text
src/
```

`main.py` imports from `src/` and runs the full project. The notebook uses the
same files and the same logic, so both versions stay aligned.

## Main Notebook

The final notebook is:

```text
notebooks/SAAM_Project_GroupN.ipynb
```

It combines:

- Part I: standard portfolio allocation
- Part II: portfolio allocation with carbon emission reduction

Two Part I notebooks are kept for traceability:

```text
notebooks/SAAM_project_part1_clean.ipynb
notebooks/SAAM_project_part1_analysis.ipynb
```

## Data Inputs

All raw files should be in:

```text
data_raw/
```

The project uses:

```text
Static_2025.xlsx
DS_RI_T_USD_M_2025.xlsx
DS_MV_T_USD_M_2025.xlsx
DS_CO2_SCOPE_1_Y_2025.xlsx
DS_REV_Y_2025.xlsx
DS_MV_T_USD_Y_2025.xlsx
Risk_Free_Rate_2025.xlsx
```

For Group N, we filter the static universe to Emerging Markets and use Scope 1
carbon emissions. Scope 2 data is still present in `data_raw/`, but it is not
used for our assigned strategy.

## Portfolios

The notebook compares:

- MV: long-only minimum-variance portfolio
- VW: value-weighted benchmark
- MV(0.5): minimum-variance portfolio with a 50% carbon-footprint constraint
- VW(0.5): tracking-error-minimizing portfolio with a 50% carbon-footprint constraint
- VW(NZ): net-zero portfolio with a 10% annual carbon-footprint reduction path

## Methodology Step By Step

The workflow follows the project statement, in this order:

1. Load the raw Datastream files from `data_raw/`.
2. Select the Group N universe: Emerging Markets firms.
3. Clean monthly total return index data and treat prices below 0.5 as missing.
4. Convert monthly price indexes into simple monthly returns.
5. Apply a delisting adjustment when a firm disappears from the sample.
6. Build a dynamic investment universe for each rebalance year.
7. Estimate expected returns and covariance matrices using the previous 10 years
   of monthly returns.
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

## Run

Install dependencies:

```zsh
pip install -r requirements.txt
```

Then either open the final notebook and run all cells from top to bottom:

```text
notebooks/SAAM_Project_GroupN.ipynb
```

or run the full Python pipeline from the project root:

```zsh
python main.py
```

The script prints progress messages along the way, which helps identify where
it is if a run takes time.

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
```

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
