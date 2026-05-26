import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from src.config import (
    REGION,
    LOW_PRICE_THRESHOLD,
    ROLLING_WINDOW_MONTHS,
    MIN_HISTORY_MONTHS,
    STALE_RETURN_THRESHOLD,
    CARBON_REDUCTION_TARGET,
    NET_ZERO_REDUCTION,
    NET_ZERO_BASE_YEAR,
    BACKTEST_START_YEAR,
)
from src.paths import get_project_paths, ensure_project_directories
from src.loaders import load_raw_datasets
from src.cleaning import (
    prepare_price_data,
    prepare_market_caps_data,
    prepare_carbon_data,
    apply_low_price_filter,
    compute_returns,
    apply_delisting_returns,
)
from src.universe import build_universe_by_year
from src.optimization import compute_mv_weights_by_year
from src.backtest import run_mv_backtest
from src.benchmark import run_vw_backtest


def build_covariance_by_year(
    returns_matrix,
    universe_by_year,
    rebalance_years,
    rolling_window_months,
):
    """
    Pre-compute the rolling covariance matrix used for each annual rebalance.

    The same covariance matrix is needed by the standard minimum-variance
    portfolio and the Part II carbon-constrained optimizations, so computing it
    once avoids repeated work and keeps the pipeline easier to audit.
    """
    covariance_by_year = {}

    for year in rebalance_years:
        universe = universe_by_year[year]
        end = pd.Timestamp(f"{year - 1}-12-31")
        window = returns_matrix.loc[universe, :end].iloc[:, -rolling_window_months:]
        covariance_by_year[year] = window.T.cov()

    return covariance_by_year


def build_part2_static_inputs(
    market_caps_data,
    carbon_data,
    annual_caps_data,
    universe_by_year,
    rebalance_years,
):
    """
    Pre-compute annual carbon-footprint inputs and value-weighted weights.

    These quantities depend only on the investment universe and data available
    at each rebalance date, so they can be shared across Part II strategies.
    """
    from src.carbon import compute_cf_per_unit
    from src.optimization_part2 import _get_vw_weights

    cf_per_unit_by_year = {}
    vw_weights_by_year = {}

    for year in rebalance_years:
        universe = universe_by_year[year]
        end_prev = pd.Timestamp(f"{year - 1}-12-31")

        cf_per_unit_by_year[year] = compute_cf_per_unit(
            carbon_data,
            annual_caps_data,
            universe,
            end_prev,
        )

        vw_weights_by_year[year] = _get_vw_weights(
            market_caps_data,
            universe,
            end_prev,
        )

    return cf_per_unit_by_year, vw_weights_by_year


def run_part2_pipeline(
    paths,
    returns_matrix,
    market_caps_data,
    carbon_data,
    universe_by_year,
    mv_weights_by_year,
    mv_returns_oos,
    vw_returns_oos,
    rebalance_years,
    em_isins,
    covariance_by_year,
):
    """
    Run the Part II climate-aware allocation strategies and export results.

    This function reuses the cleaned Part I data, the dynamic investment
    universe, and the unconstrained minimum-variance/value-weighted returns.
    """
    print("\nStarting Part II pipeline...", flush=True)

    from src.carbon import (
        prepare_revenue_data,
        prepare_annual_market_caps,
        compute_carbon_intensity,
        compute_vw_cf,
    )
    from src.backtest_part2 import (
        run_portfolio_backtest,
        compute_carbon_metrics_timeseries,
        compute_vw_carbon_metrics_timeseries,
    )
    from src.optimization_part2 import (
        compute_carbon_mv_weights_by_year,
        compute_te_min_weights_by_year,
    )
    from src.reporting_part2 import (
        build_performance_table as build_part2_performance_table,
        build_carbon_metrics_table,
        compute_cumulative_returns,
        export_part2_excel_workbook,
        export_part2_outputs,
        export_part2_table_pngs,
        plot_carbon_metrics,
        plot_cumulative_performance as plot_part2_cumulative_performance,
        prepare_risk_free_rate,
    )


    # Load and prepare annual carbon inputs

    revenue_raw = pd.read_excel(paths["DATA_RAW"] / "DS_REV_Y_2025.xlsx")
    annual_caps_raw = pd.read_excel(paths["DATA_RAW"] / "DS_MV_T_USD_Y_2025.xlsx")
    risk_free_raw = pd.read_excel(paths["DATA_RAW"] / "Risk_Free_Rate_2025.xlsx")
    risk_free_rate = prepare_risk_free_rate(risk_free_raw)

    revenue_data = prepare_revenue_data(revenue_raw, em_isins)
    annual_caps_data = prepare_annual_market_caps(annual_caps_raw, em_isins)
    carbon_intensity = compute_carbon_intensity(carbon_data, revenue_data)

    cf_per_unit_by_year, vw_weights_by_year = build_part2_static_inputs(
        market_caps_data=market_caps_data,
        carbon_data=carbon_data,
        annual_caps_data=annual_caps_data,
        universe_by_year=universe_by_year,
        rebalance_years=rebalance_years,
    )

    print("Part II carbon inputs prepared.", flush=True)

  
    # Baseline carbon metrics: MV and VW

    mv_waci_series, mv_cf_series = compute_carbon_metrics_timeseries(
        weights_by_year=mv_weights_by_year,
        universe_by_year=universe_by_year,
        carbon_data=carbon_data,
        annual_caps_data=annual_caps_data,
        carbon_intensity=carbon_intensity,
        rebalance_years=rebalance_years,
    )

    vw_waci_series, vw_cf_series = compute_vw_carbon_metrics_timeseries(
        market_caps_data=market_caps_data,
        universe_by_year=universe_by_year,
        carbon_data=carbon_data,
        annual_caps_data=annual_caps_data,
        carbon_intensity=carbon_intensity,
        rebalance_years=rebalance_years,
    )

    # Section 3.2 : Minimum Variance with 50% CF reduction

    carbon_mv_weights = compute_carbon_mv_weights_by_year(
        returns_matrix=returns_matrix,
        universe_by_year=universe_by_year,
        mv_cf_by_year=mv_cf_series.to_dict(),
        carbon_data=carbon_data,
        annual_caps_data=annual_caps_data,
        rebalance_years=rebalance_years,
        rolling_window_months=ROLLING_WINDOW_MONTHS,
        carbon_reduction_target=CARBON_REDUCTION_TARGET,
        covariance_by_year=covariance_by_year,
        cf_per_unit_by_year=cf_per_unit_by_year,
    )

    _, carbon_mv_returns = run_portfolio_backtest(
        returns_matrix,
        carbon_mv_weights,
        rebalance_years,
        label="MV (0.5)",
    )

    cmv_waci, cmv_cf = compute_carbon_metrics_timeseries(
        carbon_mv_weights,
        universe_by_year,
        carbon_data,
        annual_caps_data,
        carbon_intensity,
        rebalance_years,
    )

    print("MV carbon-constrained portfolio completed.", flush=True)


    # Section 3.3 : Tracking-Error Minimization with 50% CF reduction

    cf_target_50_vw = {
        year: CARBON_REDUCTION_TARGET * vw_cf_series.loc[year]
        for year in rebalance_years
    }

    te_50_weights = compute_te_min_weights_by_year(
        returns_matrix=returns_matrix,
        market_caps_data=market_caps_data,
        universe_by_year=universe_by_year,
        cf_target_by_year=cf_target_50_vw,
        carbon_data=carbon_data,
        annual_caps_data=annual_caps_data,
        rebalance_years=rebalance_years,
        rolling_window_months=ROLLING_WINDOW_MONTHS,
        covariance_by_year=covariance_by_year,
        cf_per_unit_by_year=cf_per_unit_by_year,
        vw_weights_by_year=vw_weights_by_year,
    )

    _, te_50_returns = run_portfolio_backtest(
        returns_matrix,
        te_50_weights,
        rebalance_years,
        label="VW (0.5)",
    )

    te50_waci, te50_cf = compute_carbon_metrics_timeseries(
        te_50_weights,
        universe_by_year,
        carbon_data,
        annual_caps_data,
        carbon_intensity,
        rebalance_years,
    )

    print("Tracking-error constrained portfolio completed.", flush=True)

  
    # Section 4 : Net-Zero Trajectory Portfolio

    base_year_date = pd.Timestamp(f"{NET_ZERO_BASE_YEAR}-12-31")
    base_universe = universe_by_year[BACKTEST_START_YEAR]

    vw_cf_base = compute_vw_cf(
        carbon_data,
        annual_caps_data,
        base_universe,
        base_year_date,
    )

    cf_target_nz = {
        year: (1 - NET_ZERO_REDUCTION) ** (year - NET_ZERO_BASE_YEAR) * vw_cf_base
        for year in rebalance_years
    }

    nz_weights = compute_te_min_weights_by_year(
        returns_matrix=returns_matrix,
        market_caps_data=market_caps_data,
        universe_by_year=universe_by_year,
        cf_target_by_year=cf_target_nz,
        carbon_data=carbon_data,
        annual_caps_data=annual_caps_data,
        rebalance_years=rebalance_years,
        rolling_window_months=ROLLING_WINDOW_MONTHS,
        covariance_by_year=covariance_by_year,
        cf_per_unit_by_year=cf_per_unit_by_year,
        vw_weights_by_year=vw_weights_by_year,
    )

    _, nz_returns = run_portfolio_backtest(
        returns_matrix,
        nz_weights,
        rebalance_years,
        label="VW (NZ)",
    )

    nz_waci, nz_cf = compute_carbon_metrics_timeseries(
        nz_weights,
        universe_by_year,
        carbon_data,
        annual_caps_data,
        carbon_intensity,
        rebalance_years,
    )

    print("Net-zero portfolio construction completed.", flush=True)

    # Collect final return and carbon series

    all_returns = {
        "MV": mv_returns_oos,
        "MV (0.5)": carbon_mv_returns,
        "VW": vw_returns_oos,
        "VW (0.5)": te_50_returns,
        "VW (NZ)": nz_returns,
    }

    waci_all = {
        "MV": mv_waci_series,
        "MV (0.5)": cmv_waci,
        "VW": vw_waci_series,
        "VW (0.5)": te50_waci,
        "VW (NZ)": nz_waci,
    }

    cf_all = {
        "MV": mv_cf_series,
        "MV (0.5)": cmv_cf,
        "VW": vw_cf_series,
        "VW (0.5)": te50_cf,
        "VW (NZ)": nz_cf,
    }

    perf_summary = build_part2_performance_table(
        all_returns,
        risk_free_rate=risk_free_rate,
    )

    carbon_summary = build_carbon_metrics_table(waci_all, cf_all)


    # Figures used in the final report

    # Baseline carbon metrics: MV vs VW
    plot_carbon_metrics(
        {"MV": mv_waci_series, "VW": vw_waci_series},
        ylabel="WACI (tCO2 / M$ revenue)",
        title="Weighted Average Carbon Intensity -- MV vs VW",
        figures_dir=paths["FIGURES_DIR"],
        filename="waci_mv_vw.png",
        show_plot=False,
    )

    plot_carbon_metrics(
        {"MV": mv_cf_series, "VW": vw_cf_series},
        ylabel="CF (tCO2 / M$ invested)",
        title="Carbon Footprint: MV vs VW",
        figures_dir=paths["FIGURES_DIR"],
        filename="cf_mv_vw.png",
        show_plot=False,
    )

    # Section 3.2: MV vs MV(0.5)
    plot_part2_cumulative_performance(
        compute_cumulative_returns(
            {
                "MV": mv_returns_oos,
                "MV (0.5)": carbon_mv_returns,
            }
        ),
        figures_dir=paths["FIGURES_DIR"],
        filename="cumulative_section32.png",
        title="Cumulative Returns: MV vs MV(0.5)",
        show_plot=False,
    )

    plot_carbon_metrics(
        {"MV": mv_cf_series, "MV (0.5)": cmv_cf},
        ylabel="CF (tCO2 / M$ invested)",
        title="Carbon Footprint: MV vs MV(0.5)",
        figures_dir=paths["FIGURES_DIR"],
        filename="cf_section32.png",
        show_plot=False,
    )

    # Section 3.3: VW vs VW(0.5)
    plot_part2_cumulative_performance(
        compute_cumulative_returns(
            {
                "VW": vw_returns_oos,
                "VW (0.5)": te_50_returns,
            }
        ),
        figures_dir=paths["FIGURES_DIR"],
        filename="cumulative_section33.png",
        title="Cumulative Returns: VW vs VW(0.5)",
        show_plot=False,
    )

    plot_carbon_metrics(
        {"VW": vw_cf_series, "VW (0.5)": te50_cf},
        ylabel="CF (tCO2 / M$ invested)",
        title="Carbon Footprint: VW vs VW(0.5)",
        figures_dir=paths["FIGURES_DIR"],
        filename="cf_section33.png",
        show_plot=False,
    )

    # Section 3.4: comparison of 50% reduction portfolios
    plot_part2_cumulative_performance(
        compute_cumulative_returns(
            {
                "MV": mv_returns_oos,
                "MV (0.5)": carbon_mv_returns,
                "VW": vw_returns_oos,
                "VW (0.5)": te_50_returns,
            }
        ),
        figures_dir=paths["FIGURES_DIR"],
        filename="cumulative_section34.png",
        title="Cumulative Returns: MV, MV(0.5), VW, VW(0.5)",
        show_plot=False,
    )

    plot_carbon_metrics(
        {
            "MV": mv_cf_series,
            "MV (0.5)": cmv_cf,
            "VW": vw_cf_series,
            "VW (0.5)": te50_cf,
        },
        ylabel="CF (tCO2 / M$ invested)",
        title="Carbon Footprint: All Portfolios (Section 3.4)",
        figures_dir=paths["FIGURES_DIR"],
        filename="cf_section34.png",
        show_plot=False,
    )

    # Section 4: VW, VW(0.5), VW(NZ)
    # These match the final report Figures 10 and 11.
    plot_part2_cumulative_performance(
        compute_cumulative_returns(
            {
                "VW": vw_returns_oos,
                "VW (0.5)": te_50_returns,
                "VW (NZ)": nz_returns,
            }
        ),
        figures_dir=paths["FIGURES_DIR"],
        filename="cumulative_section4.png",
        title="Cumulative Returns: VW, VW(0.5), VW(NZ)",
        show_plot=False,
    )

    plot_carbon_metrics(
        {
            "VW": vw_cf_series,
            "VW (0.5)": te50_cf,
            "VW (NZ)": nz_cf,
        },
        ylabel="CF (tCO2 / M$ invested)",
        title="Carbon Footprint: VW, VW(0.5) and VW(NZ)",
        figures_dir=paths["FIGURES_DIR"],
        filename="cf_section4.png",
        show_plot=False,
    )

    # Tables and Excel exports

    exported_paths = export_part2_outputs(
        all_returns,
        waci_all,
        cf_all,
        paths["TABLES_DIR"],
    )

    perf_path = paths["TABLES_DIR"] / "performance_summary_part2.csv"
    carbon_path = paths["TABLES_DIR"] / "carbon_metrics_summary_part2.csv"

    perf_summary.to_csv(perf_path)
    carbon_summary.to_csv(carbon_path)

    table_png_paths = export_part2_table_pngs(
        performance_summary=perf_summary,
        carbon_summary=carbon_summary,
        waci_by_year=pd.DataFrame(waci_all),
        cf_by_year=pd.DataFrame(cf_all),
        figures_dir=paths["FIGURES_DIR"],
    )

    part2_excel_path = export_part2_excel_workbook(
        returns_dict=all_returns,
        performance_summary=perf_summary,
        carbon_summary=carbon_summary,
        waci_by_year=pd.DataFrame(waci_all),
        cf_by_year=pd.DataFrame(cf_all),
        excel_dir=paths["EXCEL_DIR"],
        figures_dir=paths["FIGURES_DIR"],
    )

    print("Part II pipeline completed successfully.")
    print("Part II exported files:")

    for name, path in exported_paths.items():
        print(f"  {name}: {path}")

    print("  performance_summary_part2:", perf_path)
    print("  carbon_metrics_summary_part2:", carbon_path)

    for name, path in table_png_paths.items():
        print(f"  {name}: {path}")

    print("  Part II Excel workbook:", part2_excel_path)

    return {
        "returns": all_returns,
        "waci": waci_all,
        "cf": cf_all,
        "performance": perf_summary,
        "carbon_summary": carbon_summary,
    }


def main():
    """
    Run the full project from the raw Excel files to the final outputs.

    The order follows the assignment: first the standard portfolios from Part I,
    then the carbon-aware portfolios from Part II.
    """
    print("Launching SAAM pipeline...", flush=True)
    print("Starting Part I pipeline...", flush=True)

    # Project paths


    paths = get_project_paths()
    ensure_project_directories(paths)

    matplotlib_cache = paths["INTERMEDIATE_DIR"] / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    print("Project paths ready.", flush=True)


    # Load raw data

    static, prices_raw, market_caps_raw, carbon_raw = load_raw_datasets(
        paths["DATA_RAW"]
    )

    print("Raw datasets loaded.", flush=True)

    # Group N works on Emerging Markets firms.
    em_firms = static[static["Region"] == REGION].copy()
    em_isins = em_firms["ISIN"].tolist()


    # Clean and align data

    price_data = prepare_price_data(prices_raw, em_isins)
    price_data = apply_low_price_filter(price_data, LOW_PRICE_THRESHOLD)

    market_caps_data = prepare_market_caps_data(market_caps_raw, em_isins)
    carbon_data = prepare_carbon_data(carbon_raw, em_isins)

    print("Datasets cleaned and aligned.", flush=True)

   
    # Returns and delisting treatment

    returns_matrix = compute_returns(price_data)
    returns_matrix = apply_delisting_returns(price_data, returns_matrix)

    print("Returns matrix computed.", flush=True)


    # Dynamic universe

    rebalance_years = list(range(2014, 2026))

    universe_by_year = build_universe_by_year(
        returns_matrix=returns_matrix,
        price_data=price_data,
        carbon_data=carbon_data,
        rebalance_years=rebalance_years,
        rolling_window_months=ROLLING_WINDOW_MONTHS,
        min_history_months=MIN_HISTORY_MONTHS,
        stale_return_threshold=STALE_RETURN_THRESHOLD,
    )

    print("Dynamic universe built.", flush=True)

    # Rolling covariance matrices

    covariance_by_year = build_covariance_by_year(
        returns_matrix=returns_matrix,
        universe_by_year=universe_by_year,
        rebalance_years=rebalance_years,
        rolling_window_months=ROLLING_WINDOW_MONTHS,
    )

    print("Rolling covariance matrices built.", flush=True)


    # Part I portfolios

    mv_weights_by_year = compute_mv_weights_by_year(
        returns_matrix=returns_matrix,
        universe_by_year=universe_by_year,
        rebalance_years=rebalance_years,
        rolling_window_months=ROLLING_WINDOW_MONTHS,
        covariance_by_year=covariance_by_year,
    )

    mv_returns_by_year, mv_returns_oos = run_mv_backtest(
        returns_matrix=returns_matrix,
        mv_weights_by_year=mv_weights_by_year,
        rebalance_years=rebalance_years,
    )

    vw_returns_by_year, vw_returns_oos = run_vw_backtest(
        returns_matrix=returns_matrix,
        market_caps_data=market_caps_data,
        universe_by_year=universe_by_year,
        rebalance_years=rebalance_years,
    )

    print("Backtests completed.", flush=True)


    # Part I reporting

    print("Loading reporting module...", flush=True)

    from src.reporting import (
        build_performance_table,
        compute_cumulative_series,
        plot_cumulative_performance,
        export_part1_outputs,
        fill_part1_excel_template,
        prepare_risk_free_rate as prepare_part1_risk_free_rate,
    )

    risk_free_raw = pd.read_excel(paths["DATA_RAW"] / "Risk_Free_Rate_2025.xlsx")
    part1_risk_free_rate = prepare_part1_risk_free_rate(risk_free_raw)

    performance = build_performance_table(
        mv_returns_oos=mv_returns_oos,
        vw_returns_oos=vw_returns_oos,
        risk_free_rate=part1_risk_free_rate,
    )

    mv_cumulative, vw_cumulative = compute_cumulative_series(
        mv_returns_oos=mv_returns_oos,
        vw_returns_oos=vw_returns_oos,
    )

    figure_path = plot_cumulative_performance(
        mv_cumulative=mv_cumulative,
        vw_cumulative=vw_cumulative,
        figures_dir=paths["FIGURES_DIR"],
        show_plot=False,
    )

    exported_paths = export_part1_outputs(
        mv_returns_oos=mv_returns_oos,
        vw_returns_oos=vw_returns_oos,
        performance=performance,
        tables_dir=paths["TABLES_DIR"],
    )

    print("Part I pipeline completed successfully.")
    print("Figure saved at:", figure_path)
    print("Exported files:")

    for name, path in exported_paths.items():
        print(f"  {name}: {path}")

    filled_template_path = fill_part1_excel_template(
        templates_dir=paths["TEMPLATES_DIR"],
        excel_dir=paths["EXCEL_DIR"],
        figures_dir=paths["FIGURES_DIR"],
        mv_returns_oos=mv_returns_oos,
        vw_returns_oos=vw_returns_oos,
        risk_free_rate=part1_risk_free_rate,
    )

    print("Filled Excel template saved at:", filled_template_path)


    # Part II pipeline

    part2_results = run_part2_pipeline(
        paths=paths,
        returns_matrix=returns_matrix,
        market_caps_data=market_caps_data,
        carbon_data=carbon_data,
        universe_by_year=universe_by_year,
        mv_weights_by_year=mv_weights_by_year,
        mv_returns_oos=mv_returns_oos,
        vw_returns_oos=vw_returns_oos,
        rebalance_years=rebalance_years,
        em_isins=em_isins,
        covariance_by_year=covariance_by_year,
    )

    # Final combined Excel workbook

    from src.reporting_part2 import export_final_results_workbook

    final_workbook_path = export_final_results_workbook(
        part1_performance=performance,
        part2_returns=part2_results["returns"],
        part2_performance=part2_results["performance"],
        carbon_summary=part2_results["carbon_summary"],
        waci_by_year=pd.DataFrame(part2_results["waci"]),
        cf_by_year=pd.DataFrame(part2_results["cf"]),
        excel_dir=paths["EXCEL_DIR"],
        figures_dir=paths["FIGURES_DIR"],
    )

    print("Final combined results workbook saved at:", final_workbook_path)
    print("SAAM pipeline completed successfully.", flush=True)


if __name__ == "__main__":
    main()