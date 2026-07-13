"""
preprocessing.py
=================

Data loading, cleaning, validation, and feature engineering for the
End-to-End Sales Forecasting & Demand Intelligence System.

Design notes
------------
- `BasePreprocessor` centralizes cleaning steps that are dataset-agnostic
  (dedup, missing-value handling, type coercion, outlier capping) so every
  concrete preprocessor gets them for free and only implements what's
  actually different about its schema.
- `SuperstoreSalesPreprocessor` handles the primary transactional retail
  dataset (`train.csv`) used for the core forecasting/anomaly/segmentation
  pipeline (Phases 2-7).
- `VideoGameSalesPreprocessor` handles the secondary dataset
  (`videogame_sales.csv`), used as an additional domain example / stretch
  analysis.
- Every public method returns a *new* DataFrame rather than mutating in
  place, which avoids the classic "silent side effect three cells later"
  bug that plagues notebook-driven analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import (
    DataValidationError,
    PATHS,
    get_logger,
    load_csv,
    save_dataframe,
)

logger = get_logger(__name__)


# ======================================================================
# Base class
# ======================================================================
class BasePreprocessor(ABC):
    """
    Abstract base class defining the cleaning contract every dataset-specific
    preprocessor must follow: load -> validate -> clean -> engineer_features.
    """

    def __init__(self, source_path: Path):
        self.source_path = Path(source_path)
        self.raw_: Optional[pd.DataFrame] = None
        self.clean_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Template method: subclasses fill in the specifics
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and return the final engineered DataFrame."""
        self.raw_ = self.load()
        self.validate_schema(self.raw_)
        df = self.clean(self.raw_)
        df = self.engineer_features(df)
        self.clean_ = df
        logger.info(
            "%s pipeline complete -> final shape=%s", self.__class__.__name__, df.shape
        )
        return df

    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    # ------------------------------------------------------------------
    # Shared, dataset-agnostic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove fully duplicated rows, logging how many were found."""
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed:
            logger.info("Dropped %d exact duplicate rows", n_removed)
        return df

    @staticmethod
    def cap_outliers_iqr(
        df: pd.DataFrame, column: str, factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Winsorize a numeric column using the IQR rule instead of deleting
        outliers outright. Deleting rows discards legitimate demand signal
        (a genuine bulk order looks like an "outlier" but is real business
        data); capping preserves the row while limiting its leverage on
        downstream statistical models.
        """
        df = df.copy()
        q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_capped = ((df[column] < lower) | (df[column] > upper)).sum()
        df[column] = df[column].clip(lower=lower, upper=upper)
        if n_capped:
            logger.info(
                "Capped %d outliers in '%s' to range [%.2f, %.2f]",
                n_capped,
                column,
                lower,
                upper,
            )
        return df

    @staticmethod
    def report_missingness(df: pd.DataFrame) -> pd.DataFrame:
        """Return a tidy summary of null counts/percentages per column."""
        missing = df.isnull().sum()
        pct = (missing / len(df) * 100).round(2)
        summary = (
            pd.DataFrame({"missing_count": missing, "missing_pct": pct})
            .query("missing_count > 0")
            .sort_values("missing_count", ascending=False)
        )
        return summary


# ======================================================================
# Superstore-style transactional sales data (primary dataset: train.csv)
# ======================================================================
class SuperstoreSalesPreprocessor(BasePreprocessor):
    """
    Preprocessor for the Superstore-style retail transactions dataset.

    Expected raw columns (standard Kaggle "Sample Superstore" schema):
        Row ID, Order ID, Order Date, Ship Date, Ship Mode, Customer ID,
        Customer Name, Segment, Country, City, State, Postal Code, Region,
        Product ID, Category, Sub-Category, Product Name, Sales, Quantity,
        Discount, Profit

    Only `Order Date` and `Sales` are strictly required for the forecasting
    pipeline; everything else is used opportunistically if present, so the
    class degrades gracefully on slightly different exports of the same
    dataset family.
    """

    REQUIRED_COLUMNS = {"Order Date", "Sales"}

    def load(self) -> pd.DataFrame:
        return load_csv(self.source_path, parse_dates=None)

    def validate_schema(self, df: pd.DataFrame) -> None:
        missing_required = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_required:
            raise DataValidationError(
                f"SuperstoreSalesPreprocessor: missing required column(s) "
                f"{missing_required}. Found columns: {list(df.columns)}"
            )
        if df.empty:
            raise DataValidationError("SuperstoreSalesPreprocessor: input file is empty.")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Standardize column names: strip whitespace only (preserve original
        # naming convention since it's a well-known public schema).
        df.columns = [c.strip() for c in df.columns]

        df = self.drop_exact_duplicates(df)

        # --- Parse dates robustly (Superstore exports mix MM/DD/YYYY and
        # ISO formats depending on locale of export) ---
        df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
        if "Ship Date" in df.columns:
            df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")

        n_bad_dates = df["Order Date"].isna().sum()
        if n_bad_dates:
            logger.warning(
                "Dropping %d rows with unparseable Order Date", n_bad_dates
            )
            df = df.dropna(subset=["Order Date"])

        # --- Sales must be numeric and non-negative (returns/refunds, if
        # present in a given export, are handled as a separate signal rather
        # than silently kept as negative revenue, which would corrupt sums) ---
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
        n_bad_sales = df["Sales"].isna().sum()
        if n_bad_sales:
            logger.warning("Dropping %d rows with non-numeric Sales", n_bad_sales)
            df = df.dropna(subset=["Sales"])

        n_negative = (df["Sales"] < 0).sum()
        if n_negative:
            logger.warning(
                "Found %d rows with negative Sales; clipping to 0 "
                "(treated as return/refund noise, not demand)",
                n_negative,
            )
            df["Sales"] = df["Sales"].clip(lower=0)

        # --- Categorical hygiene ---
        for col in ("Category", "Sub-Category", "Region", "Segment", "State", "City"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # --- Outlier capping on Sales (preserves signal, limits leverage) ---
        df = self.cap_outliers_iqr(df, "Sales", factor=3.0)  # wide factor: retail sales are naturally right-skewed

        return df.reset_index(drop=True)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Calendar features (core inputs for seasonality modeling) ---
        df["Year"] = df["Order Date"].dt.year
        df["Month"] = df["Order Date"].dt.month
        df["MonthName"] = df["Order Date"].dt.strftime("%b")
        df["Quarter"] = df["Order Date"].dt.quarter
        df["Day"] = df["Order Date"].dt.day
        df["DayOfWeek"] = df["Order Date"].dt.dayofweek  # 0=Mon
        df["DayName"] = df["Order Date"].dt.strftime("%a")
        df["WeekOfYear"] = df["Order Date"].dt.isocalendar().week.astype(int)
        df["IsWeekend"] = df["DayOfWeek"].isin([5, 6])
        df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)

        # --- Fulfillment lag, if Ship Date is available ---
        if "Ship Date" in df.columns:
            df["ShippingDays"] = (df["Ship Date"] - df["Order Date"]).dt.days
            df["ShippingDays"] = df["ShippingDays"].clip(lower=0)

        # --- Profitability ratio, if Profit is available ---
        if "Profit" in df.columns:
            df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")
            df["ProfitMargin"] = np.where(
                df["Sales"] > 0, df["Profit"] / df["Sales"], np.nan
            )

        return df

    # ------------------------------------------------------------------
    def aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse transaction-level rows into a continuous daily sales series
        (required by SARIMA/Prophet, which expect one observation per period
        with no gaps). Missing calendar days are filled with 0 sales.
        """
        daily = (
            df.groupby(df["Order Date"].dt.normalize())["Sales"]
            .sum()
            .rename("Sales")
            .reset_index()
            .rename(columns={"Order Date": "Date"})
        )
        full_range = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
        daily = (
            daily.set_index("Date")
            .reindex(full_range)
            .fillna(0.0)
            .rename_axis("Date")
            .reset_index()
        )
        return daily

    def aggregate_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse transaction-level rows into a monthly sales series."""
        monthly = (
            df.set_index("Order Date")
            .resample("MS")["Sales"]  # MS = month start
            .sum()
            .rename("Sales")
            .reset_index()
            .rename(columns={"Order Date": "Date"})
        )
        return monthly

    def aggregate_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse transaction-level rows into a continuous weekly sales series
        (resampled to W-MON: weekly starting on Mondays).
        Missing weeks are filled with 0 sales.
        """
        weekly = (
            df.groupby(df["Order Date"].dt.normalize())["Sales"]
            .sum()
            .rename("Sales")
            .reset_index()
            .rename(columns={"Order Date": "Date"})
        )
        # Resample to weekly starting on Monday
        weekly = weekly.set_index("Date").resample("W-MON")["Sales"].sum().reset_index()
        
        # Fill any missing weeks in the range
        full_range = pd.date_range(weekly["Date"].min(), weekly["Date"].max(), freq="W-MON")
        weekly = (
            weekly.set_index("Date")
            .reindex(full_range)
            .fillna(0.0)
            .rename_axis("Date")
            .reset_index()
        )
        return weekly



# ======================================================================
# Video Game Sales dataset (secondary dataset: videogame_sales.csv)
# ======================================================================
class VideoGameSalesPreprocessor(BasePreprocessor):
    """
    Preprocessor for the Kaggle "Video Game Sales" dataset.

    Expected raw columns:
        Rank, Name, Platform, Year, Genre, Publisher,
        NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales
    (all *_Sales columns are in millions of units)
    """

    REQUIRED_COLUMNS = {"Name", "Year", "Global_Sales"}

    def load(self) -> pd.DataFrame:
        return load_csv(self.source_path, parse_dates=None)

    def validate_schema(self, df: pd.DataFrame) -> None:
        missing_required = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_required:
            raise DataValidationError(
                f"VideoGameSalesPreprocessor: missing required column(s) "
                f"{missing_required}. Found columns: {list(df.columns)}"
            )
        if df.empty:
            raise DataValidationError("VideoGameSalesPreprocessor: input file is empty.")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        df = self.drop_exact_duplicates(df)

        # --- Year: numeric, drop implausible / missing years ---
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        n_bad_year = df["Year"].isna().sum()
        if n_bad_year:
            logger.warning("Dropping %d rows with missing/invalid Year", n_bad_year)
            df = df.dropna(subset=["Year"])
        df = df[(df["Year"] >= 1970) & (df["Year"] <= pd.Timestamp.now().year)]
        df["Year"] = df["Year"].astype(int)

        # --- Sales columns: numeric, non-negative ---
        sales_cols = [
            c for c in ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
            if c in df.columns
        ]
        for col in sales_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0)

        # --- Categorical hygiene ---
        for col in ("Platform", "Genre", "Publisher"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace({"nan": "Unknown"})

        # --- Outlier capping on Global_Sales (blockbuster titles are real,
        # but we limit their leverage on aggregate trend/seasonality stats) ---
        if "Global_Sales" in df.columns:
            df = self.cap_outliers_iqr(df, "Global_Sales", factor=3.0)

        return df.reset_index(drop=True)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Synthetic "release date" (Jan 1 of release year) so this dataset
        # can flow through the same time-series utilities as Superstore data.
        df["ReleaseDate"] = pd.to_datetime(df["Year"].astype(str) + "-01-01")
        df["Decade"] = (df["Year"] // 10 * 10).astype(str) + "s"

        if {"NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"}.issubset(df.columns):
            # Share of global sales coming from each region -- useful for
            # regional demand segmentation later.
            for region_col, share_name in [
                ("NA_Sales", "NA_Share"),
                ("EU_Sales", "EU_Share"),
                ("JP_Sales", "JP_Share"),
                ("Other_Sales", "Other_Share"),
            ]:
                df[share_name] = np.where(
                    df["Global_Sales"] > 0, df[region_col] / df["Global_Sales"], 0.0
                )

        return df

    # ------------------------------------------------------------------
    def aggregate_yearly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse title-level rows into total global sales per year."""
        yearly = (
            df.groupby("Year")["Global_Sales"]
            .sum()
            .rename("Global_Sales")
            .reset_index()
            .sort_values("Year")
        )
        return yearly


# ======================================================================
# Convenience entry point
# ======================================================================
def run_all_preprocessing() -> dict:
    """
    Run both preprocessors end-to-end using the standard project data paths
    and persist cleaned outputs back to disk. Returns a dict of the cleaned
    DataFrames for immediate use in the notebook / dashboard.
    """
    PATHS.ensure_all()
    results = {}

    train_path = PATHS.data / "train.csv"
    if train_path.exists():
        superstore = SuperstoreSalesPreprocessor(train_path)
        clean_superstore = superstore.run()
        save_dataframe(clean_superstore, PATHS.data / "train_clean.csv")
        results["superstore"] = clean_superstore
    else:
        logger.warning("'%s' not found -- skipping Superstore preprocessing", train_path)

    vg_path = PATHS.data / "videogame_sales.csv"
    if vg_path.exists():
        vg = VideoGameSalesPreprocessor(vg_path)
        clean_vg = vg.run()
        save_dataframe(clean_vg, PATHS.data / "videogame_sales_clean.csv")
        results["videogames"] = clean_vg
    else:
        logger.warning("'%s' not found -- skipping Video Game Sales preprocessing", vg_path)

    return results


if __name__ == "__main__":
    run_all_preprocessing()
