import sys
import pandas, numpy, streamlit, prophet, joblib, sklearn, matplotlib, plotly
from src.utils import PATHS
from src.forecasting import train_and_compare_models
from src.anomaly import detect_anomalies
from src.clustering import get_recommendation
