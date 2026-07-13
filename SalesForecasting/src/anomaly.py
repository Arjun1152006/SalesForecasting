"""
anomaly.py
==========

Anomaly detection module for the End-to-End Sales Forecasting & Demand Intelligence System.
Implements Isolation Forest, Rolling Mean, and Z-score algorithms, combines them
into an ensemble flag, and exports anomaly tables and charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.utils import PATHS, get_logger, DataValidationError
from src.visualization import plot_anomalies

logger = get_logger(__name__)


def detect_anomalies(
    df: pd.DataFrame,
    date_col: str = "Date",
    sales_col: str = "Sales",
    iforest_contamination: float = 0.03,
    rolling_window: int = 8,
    rolling_k: float = 2.0,
    z_threshold: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run three anomaly detection methods on the sales series and flag outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Transactional or aggregated sales dataframe.
    date_col : str
        Name of date column.
    sales_col : str
        Name of numerical sales column.
    iforest_contamination : float
        Proportion of outliers in Isolation Forest.
    rolling_window : int
        Rolling window size for local thresholding.
    rolling_k : float
        Standard deviation multiplier for rolling threshold.
    z_threshold : float
        Standard deviation threshold for static Z-score.

    Returns
    -------
    df_output : pd.DataFrame
        DataFrame with anomaly flags and scores appended.
    anomaly_table : pd.DataFrame
        Filtered DataFrame containing only rows flagged as consensus anomalies.
    """
    if df.empty:
        raise DataValidationError("Cannot detect anomalies on empty DataFrame.")
    if sales_col not in df.columns:
        raise DataValidationError(f"Sales column '{sales_col}' not found in DataFrame.")

    df_out = df.copy().sort_values(by=date_col)

    # 1. Static Z-score Anomaly Detection
    sales_mean = df_out[sales_col].mean()
    sales_std = df_out[sales_col].std()
    if sales_std == 0:
        sales_std = 1e-5  # avoid division by zero

    df_out["Z_Score"] = (df_out[sales_col] - sales_mean) / sales_std
    df_out["Anomaly_Z"] = (df_out["Z_Score"].abs() > z_threshold).astype(int)

    # 2. Rolling Mean & Standard Deviation (Dynamic local bounds)
    # Ensure window is not larger than data size
    actual_window = max(3, min(rolling_window, len(df_out) // 3))
    df_out["Rolling_Mean"] = df_out[sales_col].rolling(window=actual_window, min_periods=1, center=True).mean()
    df_out["Rolling_Std"] = df_out[sales_col].rolling(window=actual_window, min_periods=1, center=True).std().fillna(0.0)

    # Upper/lower bounds
    upper_bound = df_out["Rolling_Mean"] + rolling_k * df_out["Rolling_Std"]
    lower_bound = df_out["Rolling_Mean"] - rolling_k * df_out["Rolling_Std"]

    df_out["Anomaly_Rolling"] = ((df_out[sales_col] > upper_bound) | (df_out[sales_col] < lower_bound)).astype(int)

    # 3. Isolation Forest (Multidimensional distribution boundary)
    X = df_out[[sales_col]].values
    try:
        # Standard fit
        iforest = IsolationForest(
            contamination=iforest_contamination,
            random_state=42,
            n_estimators=100,
        )
        iforest_preds = iforest.fit_predict(X)
        # sklearn outputs -1 for anomalies, 1 for normal
        df_out["Anomaly_IForest"] = (iforest_preds == -1).astype(int)
        df_out["IForest_Score"] = iforest.decision_function(X)
    except Exception as e:
        logger.warning("Isolation Forest failed: %s. Defaulting to 0 anomalies.", e)
        df_out["Anomaly_IForest"] = 0
        df_out["IForest_Score"] = 0.0

    # 4. Ensemble Consensus Voting
    # Flag as anomaly if at least 2 out of the 3 detectors agree
    detector_cols = ["Anomaly_Z", "Anomaly_Rolling", "Anomaly_IForest"]
    df_out["Anomaly_Vote_Count"] = df_out[detector_cols].sum(axis=1)
    df_out["Is_Anomaly"] = (df_out["Anomaly_Vote_Count"] >= 2).astype(int)

    # Clean up intermediate cols slightly if needed, but keeping them is useful for debug
    anomaly_table = df_out[df_out["Is_Anomaly"] == 1].copy()

    logger.info(
        "Anomaly detection run: Z-Score found %d, Rolling found %d, IForest found %d. Consensus: %d anomalies.",
        df_out["Anomaly_Z"].sum(),
        df_out["Anomaly_Rolling"].sum(),
        df_out["Anomaly_IForest"].sum(),
        df_out["Is_Anomaly"].sum(),
    )

    return df_out, anomaly_table


def run_and_plot_anomalies(
    df: pd.DataFrame,
    date_col: str = "Date",
    sales_col: str = "Sales",
    save_dir: Optional[Union[str, Path]] = None,
    interactive: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run anomaly detection, save anomaly plots to the charts directory,
    and return the output.
    """
    df_anom, table_anom = detect_anomalies(df, date_col, sales_col)

    # Resolve paths
    if save_dir is None:
        save_dir = PATHS.charts

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img_ext = "html" if interactive else "png"
    save_path = save_dir / f"sales_anomalies.{img_ext}"

    # Draw anomaly chart comparing both methods
    plot_anomalies(
        df_anom,
        date_col=date_col,
        sales_col=sales_col,
        anomaly_col=["Anomaly_IForest", "Anomaly_Rolling"],
        save_path=save_path,
        interactive=interactive,
    )

    # Save anomaly table to CSV
    csv_path = save_dir / "anomalies_table.csv"
    table_anom.to_csv(csv_path, index=False)
    logger.info("Saved anomaly table to %s", csv_path)

    return df_anom, table_anom
