"""
forecasting.py
==============

Forecasting engine for the End-to-End Sales Forecasting & Demand Intelligence System.
Implements SARIMA, Prophet, and XGBoost models for sales forecasting, along with
evaluation metrics, model selection, serialization, and forecast plotting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List

# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor

from src.utils import PATHS, get_logger, ModelTrainingError
from src.visualization import plot_forecast

logger = get_logger(__name__)


# ======================================================================
# 1. Metric Calculations
# ======================================================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics for model evaluation.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Avoid division by zero in MAPE
    y_true_safe = np.where(y_true == 0, 1e-5, y_true)
    mape = mean_absolute_percentage_error(y_true_safe, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


# ======================================================================
# 2. Model Wrappers
# ======================================================================
class SARIMAForecaster:
    """
    Wrapper for the SARIMAX model matching our standard forecaster API.
    
    SARIMA Parameters Justification:
    --------------------------------
    - order=(1, 1, 1): Captures non-seasonal auto-regressive (p=1), first-order differencing (d=1) to ensure stationarity, and moving average (q=1) terms.
    - seasonal_order=(1, 1, 1, 12): Captures seasonal auto-regressive (P=1), seasonal differencing (D=1) to remove seasonal trend, and seasonal moving average (Q=1) terms.
    - m=12: Monthly retail sales data exhibits a strong annual/yearly seasonality pattern (repeating every 12 months), therefore a seasonal period of m=12 was selected.
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None
        self.last_train_data = None
        self.date_col = None
        self.sales_col = None

    def fit(self, df: pd.DataFrame, date_col: str = "Date", sales_col: str = "Sales") -> SARIMAForecaster:
        self.date_col = date_col
        self.sales_col = sales_col
        self.last_train_data = df.copy().sort_values(by=date_col)

        # Set datetime index for statsmodels
        series = self.last_train_data.set_index(date_col)[sales_col]
        # Set frequency to month start if applicable
        if pd.api.types.is_datetime64_any_dtype(series.index):
            series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()

        try:
            model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_fit = model.fit(disp=False)
            logger.info("SARIMA model successfully fitted.")
        except Exception as e:
            raise ModelTrainingError(f"SARIMA fit failed: {str(e)}")
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.model_fit is None:
            raise ModelTrainingError("Model must be fitted before prediction.")

        try:
            forecast_res = self.model_fit.get_forecast(steps=horizon)
            forecast_mean = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int(alpha=0.05)

            forecast_dates = forecast_mean.index

            res = pd.DataFrame({
                self.date_col: forecast_dates,
                "Forecast": forecast_mean.values,
                "Lower_CI": conf_int.iloc[:, 0].values,
                "Upper_CI": conf_int.iloc[:, 1].values,
            })
            # Ensure forecast values are non-negative
            res["Forecast"] = res["Forecast"].clip(lower=0)
            res["Lower_CI"] = res["Lower_CI"].clip(lower=0)
            res["Upper_CI"] = res["Upper_CI"].clip(lower=0)
            return res
        except Exception as e:
            raise ModelTrainingError(f"SARIMA prediction failed: {str(e)}")


class ProphetForecaster:
    """Wrapper for the Meta Prophet model matching our standard forecaster API."""

    def __init__(self, seasonality_mode: str = "additive"):
        self.seasonality_mode = seasonality_mode
        self.model = None
        self.date_col = None
        self.sales_col = None

    def fit(self, df: pd.DataFrame, date_col: str = "Date", sales_col: str = "Sales") -> ProphetForecaster:
        self.date_col = date_col
        self.sales_col = sales_col

        # Prophet requires columns 'ds' and 'y'
        prophet_df = df[[date_col, sales_col]].rename(columns={date_col: "ds", sales_col: "y"})
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        try:
            # Silence logging to keep output clean
            import logging
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
            )
            self.model.fit(prophet_df)
            logger.info("Prophet model successfully fitted.")
        except Exception as e:
            raise ModelTrainingError(f"Prophet fit failed: {str(e)}")
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.model is None:
            raise ModelTrainingError("Model must be fitted before prediction.")

        try:
            future = self.model.make_future_dataframe(periods=horizon, freq="MS")
            # We only want the horizon dates
            future_horizon = future.tail(horizon)

            forecast = self.model.predict(future_horizon)
            res = pd.DataFrame({
                self.date_col: forecast["ds"],
                "Forecast": forecast["yhat"].values,
                "Lower_CI": forecast["yhat_lower"].values,
                "Upper_CI": forecast["yhat_upper"].values,
            })
            res["Forecast"] = res["Forecast"].clip(lower=0)
            res["Lower_CI"] = res["Lower_CI"].clip(lower=0)
            res["Upper_CI"] = res["Upper_CI"].clip(lower=0)
            return res
        except Exception as e:
            raise ModelTrainingError(f"Prophet prediction failed: {str(e)}")


class XGBoostForecaster:
    """Wrapper for XGBoost regressor using recursive forecasting strategy."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.05, lags: int = 3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lags = lags
        self.model = None
        self.date_col = None
        self.sales_col = None
        self.history_sales: List[float] = []
        self.last_date = None
        self.residual_std = 0.0

    def _engineer_features(self, series: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Convert a time series into a supervised learning dataset with lags and calendar features."""
        df = pd.DataFrame(series)
        df.columns = ["y"]

        # Create lags
        for i in range(1, self.lags + 1):
            df[f"lag_{i}"] = df["y"].shift(i)

        # Create calendar features
        df["Month"] = df.index.month
        df["Quarter"] = df.index.quarter

        df = df.dropna()
        X = df.drop(columns=["y"])
        y = df["y"]
        return X, y

    def fit(self, df: pd.DataFrame, date_col: str = "Date", sales_col: str = "Sales") -> XGBoostForecaster:
        self.date_col = date_col
        self.sales_col = sales_col

        df_sorted = df.copy().sort_values(by=date_col)
        series = df_sorted.set_index(date_col)[sales_col]
        if pd.api.types.is_datetime64_any_dtype(series.index):
            series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()

        self.history_sales = series.tolist()
        self.last_date = series.index[-1]

        X, y = self._engineer_features(series)

        try:
            self.model = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
            )
            self.model.fit(X, y)

            # Estimate uncertainty using standard deviation of training residuals
            preds_train = self.model.predict(X)
            residuals = y.values - preds_train
            self.residual_std = np.std(residuals)

            logger.info("XGBoost forecaster successfully fitted.")
        except Exception as e:
            raise ModelTrainingError(f"XGBoost fit failed: {str(e)}")
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.model is None:
            raise ModelTrainingError("Model must be fitted before prediction.")

        try:
            # We predict recursively
            current_lags = list(self.history_sales[-self.lags:])
            predictions = []
            dates = []
            curr_date = self.last_date

            for i in range(horizon):
                # Move date forward 1 month
                curr_date = curr_date + pd.DateOffset(months=1)
                dates.append(curr_date)

                # Prepare feature vector
                feat_dict = {}
                for idx, val in enumerate(reversed(current_lags)):
                    feat_dict[f"lag_{idx+1}"] = [val]
                feat_dict["Month"] = [curr_date.month]
                feat_dict["Quarter"] = [curr_date.quarter]

                X_pred = pd.DataFrame(feat_dict)
                # Ensure correct column ordering matching training
                X_pred = X_pred[[f"lag_{j}" for j in range(1, self.lags + 1)] + ["Month", "Quarter"]]

                pred = float(self.model.predict(X_pred)[0])
                pred = max(0.0, pred)
                predictions.append(pred)

                # Update lags recursive
                current_lags.pop(0)
                current_lags.append(pred)

            predictions = np.array(predictions)
            # Create a simple 95% confidence interval using residual std
            lower_ci = (predictions - 1.96 * self.residual_std).clip(0)
            upper_ci = predictions + 1.96 * self.residual_std

            res = pd.DataFrame({
                self.date_col: dates,
                "Forecast": predictions,
                "Lower_CI": lower_ci,
                "Upper_CI": upper_ci,
            })
            return res
        except Exception as e:
            raise ModelTrainingError(f"XGBoost prediction failed: {str(e)}")


# ======================================================================
# 3. Model Comparison & Best Selection Pipeline
# ======================================================================
def train_and_compare_models(
    df: pd.DataFrame,
    date_col: str = "Date",
    sales_col: str = "Sales",
    val_periods: int = 3,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train SARIMA, Prophet, and XGBoost on historical sales.
    Evaluates them on a validation split (last `val_periods` months).
    Identifies the best model based on MAPE, retrains it on all data,
    and returns metrics, validation predictions, and model comparison details.
    """
    df_clean = df.copy().sort_values(by=date_col)
    n_samples = len(df_clean)

    if n_samples < 2 * val_periods:
        raise ModelTrainingError(
            f"Not enough data points ({n_samples}) to split into train and validation "
            f"sets with val_periods={val_periods}."
        )

    # Train/Val Split
    train_df = df_clean.iloc[:-val_periods]
    val_df = df_clean.iloc[-val_periods:]
    y_val = val_df[sales_col].values

    models = {
        "SARIMA": SARIMAForecaster(),
        "Prophet": ProphetForecaster(),
        "XGBoost": XGBoostForecaster(),
    }

    metrics_dict = {}
    val_predictions = {}

    for name, model in models.items():
        try:
            logger.info("Training %s on validation-train split...", name)
            model.fit(train_df, date_col, sales_col)
            pred_df = model.predict(horizon=val_periods)
            y_pred = pred_df["Forecast"].values

            metrics = calculate_metrics(y_val, y_pred)
            metrics_dict[name] = metrics
            val_predictions[name] = pred_df
            logger.info("%s validation metrics: %s", name, metrics)
        except Exception as err:
            logger.error("Failed to evaluate %s: %s", name, err)
            metrics_dict[name] = {"MAE": np.inf, "RMSE": np.inf, "MAPE": np.inf}

    # Fit all models on full dataset and generate future 3-month forecasts
    forecast_month_1 = {}
    forecast_month_2 = {}
    forecast_month_3 = {}
    fitted_models_full = {}

    for name, model in models.items():
        try:
            logger.info("Fitting %s on full dataset for future forecasting...", name)
            model.fit(df_clean, date_col, sales_col)
            fitted_models_full[name] = model
            
            # Predict future 3 months
            fc_df = model.predict(horizon=3)
            forecast_month_1[name] = float(fc_df["Forecast"].values[0])
            forecast_month_2[name] = float(fc_df["Forecast"].values[1])
            forecast_month_3[name] = float(fc_df["Forecast"].values[2])
        except Exception as e:
            logger.error("Failed to fit %s on full dataset: %s", name, e)
            forecast_month_1[name] = np.nan
            forecast_month_2[name] = np.nan
            forecast_month_3[name] = np.nan

    # Compare models
    comparison_records = []
    for name, metrics in metrics_dict.items():
        comparison_records.append({
            "Model": name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE": metrics["MAPE"],
            "Forecast Month 1": forecast_month_1.get(name, np.nan),
            "Forecast Month 2": forecast_month_2.get(name, np.nan),
            "Forecast Month 3": forecast_month_3.get(name, np.nan),
        })
    # Sort by lowest RMSE
    comparison_df = pd.DataFrame(comparison_records).sort_values("RMSE")

    # Select best model
    best_model_name = comparison_df.iloc[0]["Model"]
    best_rmse = comparison_df.iloc[0]["RMSE"]
    logger.info("--> Best model selected by lowest RMSE: %s (RMSE: %.4f)", best_model_name, best_rmse)

    best_model_full = fitted_models_full[best_model_name]

    # Save to models folder
    PATHS.ensure_all()
    model_save_path = PATHS.models / "best_forecaster.joblib"
    joblib.dump(best_model_full, model_save_path)
    logger.info("Serialized best forecaster to %s", model_save_path)

    # Save comparison dataframe to outputs/model_comparison.csv
    outputs_dir = PATHS.root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(outputs_dir / "model_comparison.csv", index=False)
    logger.info("Saved model comparison to %s", outputs_dir / "model_comparison.csv")

    results = {
        "best_model_name": best_model_name,
        "best_model": best_model_full,
        "metrics": metrics_dict,
        "val_predictions": val_predictions,
        "comparison_df": comparison_df,
        "val_df": val_df,
    }

    return results, comparison_df



def load_best_model() -> Any:
    """Load the serialized best forecasting model from the models directory."""
    path = PATHS.models / "best_forecaster.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Trained forecaster not found at '{path}'. Train the model first."
        )
    return joblib.load(path)
