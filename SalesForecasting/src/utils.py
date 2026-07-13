"""
utils.py
========

Cross-cutting utilities shared across the project: logging configuration,
lightweight path/config management, IO helpers, and custom exception types.

Keeping these concerns here (rather than duplicated in every module) is what
makes the rest of the codebase read like application code instead of a
notebook dump.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


# ----------------------------------------------------------------------
# Custom Exceptions
# ----------------------------------------------------------------------
class SalesForecastingError(Exception):
    """Base exception for all project-specific errors."""


class DataValidationError(SalesForecastingError):
    """Raised when input data fails schema / integrity checks."""


class ModelTrainingError(SalesForecastingError):
    """Raised when a forecasting/clustering/anomaly model fails to train."""


class ConfigurationError(SalesForecastingError):
    """Raised when required configuration or file paths are missing."""


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger that writes to stdout with a consistent format.

    Using a factory function (instead of `logging.basicConfig` at import time)
    means every module gets its own named logger without fighting over global
    logging state -- the standard pattern in production codebases.

    Parameters
    ----------
    name : str
        Usually `__name__` of the calling module.
    level : int
        Logging level, defaults to INFO.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers on repeated calls
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger


# ----------------------------------------------------------------------
# Project Paths (single source of truth)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectPaths:
    """
    Central registry of project directories, resolved relative to the
    project root (the directory containing this `src/` package).

    Using a frozen dataclass means paths are immutable and can be imported
    anywhere without risk of accidental mutation.
    """

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def charts(self) -> Path:
        return self.root / "charts"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def screenshots(self) -> Path:
        return self.root / "screenshots"

    def ensure_all(self) -> None:
        """Create every managed directory if it does not already exist."""
        for p in (self.data, self.charts, self.models, self.assets, self.screenshots):
            p.mkdir(parents=True, exist_ok=True)


PATHS = ProjectPaths()


# ----------------------------------------------------------------------
# IO Helpers
# ----------------------------------------------------------------------
def load_csv(
    path: Path,
    parse_dates: Optional[list] = None,
    encoding_fallbacks: tuple = ("utf-8", "latin-1", "cp1252"),
) -> pd.DataFrame:
    """
    Load a CSV robustly, trying a sequence of encodings -- real-world retail
    exports (e.g. the Kaggle Superstore dataset) are frequently saved in
    Latin-1/cp1252 rather than UTF-8 and raise `UnicodeDecodeError` otherwise.

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    parse_dates : list, optional
        Columns to parse as datetimes.
    encoding_fallbacks : tuple
        Encodings to attempt, in order.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    DataValidationError
        If the file does not exist or cannot be parsed with any encoding.
    """
    logger = get_logger(__name__)

    if not path.exists():
        raise DataValidationError(
            f"Expected data file not found at '{path}'. "
            f"Place the dataset in the project's data/ directory."
        )

    last_error: Optional[Exception] = None
    for encoding in encoding_fallbacks:
        try:
            df = pd.read_csv(path, parse_dates=parse_dates, encoding=encoding)
            logger.info(
                "Loaded '%s' with encoding='%s' -> shape=%s", path.name, encoding, df.shape
            )
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    raise DataValidationError(
        f"Could not decode '{path}' with any of {encoding_fallbacks}: {last_error}"
    )


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Persist a DataFrame to CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    get_logger(__name__).info("Saved dataframe -> %s (shape=%s)", path, df.shape)
