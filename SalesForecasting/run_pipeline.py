"""
run_pipeline.py
===============

Master pipeline execution script for the End-to-End Sales Forecasting & Demand
Intelligence System. Runs preprocessing, fits and selects forecasting models,
runs anomaly detection, performs demand segmentation, and outputs all charts and
the final Word report (summary.docx).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Force matplotlib to use non-interactive Agg backend before any other imports
import matplotlib
matplotlib.use('Agg')


# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import PATHS, get_logger, save_dataframe
from src.preprocessing import SuperstoreSalesPreprocessor
from src.forecasting import train_and_compare_models
from src.anomaly import run_and_plot_anomalies
from src.clustering import run_and_plot_segmentation
from src.reporting import generate_executive_report
from src.visualization import plot_forecast, plot_monthly_sales, plot_regional_sales, plot_category_sales, plot_sales_heatmap

logger = get_logger("pipeline_runner")


def main():
    logger.info("Starting End-to-End Sales Forecasting Pipeline execution...")
    PATHS.ensure_all()

    # ------------------------------------------------------------------
    # Step 1: Preprocessing
    # ------------------------------------------------------------------
    train_path = PATHS.data / "train.csv"
    if not train_path.exists():
        logger.error("Dataset 'train.csv' not found in data/ folder. Copying it from root...")
        root_train = PROJECT_ROOT / "train.csv"
        if root_train.exists():
            import shutil
            shutil.copy(str(root_train), str(train_path))
            logger.info("Successfully copied train.csv to data/")
        else:
            logger.error("Could not locate train.csv at root or data/ directory. Aborting.")
            return

    preprocessor = SuperstoreSalesPreprocessor(train_path)
    clean_df = preprocessor.run()
    
    # Save clean dataset
    save_dataframe(clean_df, PATHS.data / "train_clean.csv")

    # Aggregate datasets
    monthly_sales = preprocessor.aggregate_monthly(clean_df)
    daily_sales = preprocessor.aggregate_daily(clean_df)
    weekly_sales = preprocessor.aggregate_weekly(clean_df)
    
    save_dataframe(monthly_sales, PATHS.data / "monthly_sales.csv")
    save_dataframe(daily_sales, PATHS.data / "daily_sales.csv")
    save_dataframe(weekly_sales, PATHS.data / "weekly_sales.csv")

    # ------------------------------------------------------------------
    # Step 2: Exploratory Data Analysis & Viz Export (Static PNGs)
    # ------------------------------------------------------------------
    logger.info("Generating static EDA charts...")
    plot_monthly_sales(monthly_sales, save_path=PATHS.charts / "monthly_sales_trend.png", interactive=False)
    plot_monthly_sales(monthly_sales, save_path=PATHS.charts / "monthly_sales_trend.html", interactive=True)
    
    plot_regional_sales(clean_df, save_path=PATHS.charts / "regional_sales.png", interactive=False)
    plot_regional_sales(clean_df, save_path=PATHS.charts / "regional_sales.html", interactive=True)
    
    plot_category_sales(clean_df, save_path=PATHS.charts / "category_sales.png", interactive=False)
    plot_category_sales(clean_df, save_path=PATHS.charts / "category_sales.html", interactive=True)
    
    # Seasonality heatmap: Month vs Day of Week
    # We map DayOfWeek (0-6) to Names, Month to Names
    # Clean df already has MonthName and DayName
    plot_sales_heatmap(clean_df, index_col="MonthName", columns_col="DayName", save_path=PATHS.charts / "sales_heatmap.png", interactive=False)
    plot_sales_heatmap(clean_df, index_col="MonthName", columns_col="DayName", save_path=PATHS.charts / "sales_heatmap.html", interactive=True)

    # ------------------------------------------------------------------
    # Step 3: Time Series Forecasting
    # ------------------------------------------------------------------
    logger.info("Running forecasting models comparison...")
    forecast_results, comparison_df = train_and_compare_models(monthly_sales, val_periods=3)
    
    # Print metrics to console
    print("\n=== Forecasting Model Comparison ===")
    print(comparison_df.to_string(index=False))
    print("=====================================\n")

    # Predict future 3 months with the best model
    best_model_name = forecast_results["best_model_name"]
    best_model = forecast_results["best_model"]
    forecast_df = best_model.predict(horizon=3)
    save_dataframe(forecast_df, PATHS.data / "forecast_future_3m.csv")
    
    # Export forecast plots
    plot_forecast(
        historical_df=monthly_sales,
        forecast_df=forecast_df,
        title=f"Sales Forecast using {best_model_name}",
        save_path=PATHS.charts / "forecast_sales_forecast.png",
        interactive=False
    )
    plot_forecast(
        historical_df=monthly_sales,
        forecast_df=forecast_df,
        title=f"Sales Forecast using {best_model_name}",
        save_path=PATHS.charts / "forecast_sales_forecast.html",
        interactive=True
    )

    # ------------------------------------------------------------------
    # Step 4: Anomaly Detection
    # ------------------------------------------------------------------
    logger.info("Detecting sales anomalies...")
    # Run anomalies on weekly sales
    df_anom, table_anom = run_and_plot_anomalies(weekly_sales, interactive=False)
    # Also generate interactive version for dashboard
    run_and_plot_anomalies(weekly_sales, interactive=True)

    # Save to data folder
    save_dataframe(df_anom, PATHS.data / "weekly_sales_anomalies.csv")
    save_dataframe(table_anom, PATHS.data / "anomalies_only.csv")

    # ------------------------------------------------------------------
    # Step 5: Demand Segmentation
    # ------------------------------------------------------------------
    logger.info("Performing product demand segmentation (KMeans)...")
    segmented_features, pca_df = run_and_plot_segmentation(clean_df, n_clusters=4, interactive=False)
    # Also generate interactive PCA plot
    run_and_plot_segmentation(clean_df, n_clusters=4, interactive=True)

    # Save to data folder
    save_dataframe(segmented_features, PATHS.data / "segmented_products.csv", index=True)

    # ------------------------------------------------------------------
    # Step 6: Document Reporting
    # ------------------------------------------------------------------
    logger.info("Compiling summary.docx report...")
    generate_executive_report(
        df_sales=clean_df,
        forecast_results=forecast_results,
        anomaly_table=table_anom,
        segmented_features=segmented_features,
    )

    logger.info("Pipeline execution complete! All artifacts and summary.docx have been generated successfully.")


if __name__ == "__main__":
    main()
