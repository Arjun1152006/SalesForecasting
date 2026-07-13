"""
src package
===========

Reusable business-logic modules for the End-to-End Sales Forecasting &
Demand Intelligence System.

Modules
-------
preprocessing   : Data loading, cleaning, validation, feature engineering
forecasting     : SARIMA / Prophet / XGBoost forecasting engine   (Phase 4)
anomaly         : Isolation Forest / rolling stats / z-score detectors (Phase 6)
clustering      : KMeans demand segmentation + PCA                 (Phase 7)
visualization   : Centralized Plotly / Matplotlib chart factory    (Phase 2+)
utils           : Logging, config, IO helpers, custom exceptions
"""

__version__ = "0.1.0"
