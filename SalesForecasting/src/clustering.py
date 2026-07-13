"""
clustering.py
=============

Demand segmentation module for the End-to-End Sales Forecasting & Demand Intelligence System.
Aggregates transactional sales into product-level features (volume, frequency,
volatility, trend), scales features, runs KMeans clustering, performs PCA, and
maps segments to actionable replenishment and business recommendations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils import PATHS, get_logger, DataValidationError
from src.visualization import plot_clusters

logger = get_logger(__name__)


# ======================================================================
# 1. Feature Engineering
# ======================================================================
def engineer_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data to create product-level demand features.

    Features generated:
    - Total_Sales: Total sales volume.
    - YoY_Growth: Year-over-Year sales growth rate.
    - Monthly_Sales_Volatility: Standard deviation of monthly sales.
    - Avg_Order_Value: Average order value (Total Sales / Unique Orders).
    """
    if "Product Name" not in df.columns:
        key_col = "Sub-Category" if "Sub-Category" in df.columns else "Category"
        logger.warning("Product Name column not found. Clustering by '%s' instead.", key_col)
    else:
        key_col = "Product Name"

    # Base aggregations
    grp = df.groupby(key_col)
    features = pd.DataFrame()
    features["Total_Sales"] = grp["Sales"].sum()

    # 1. Year-over-Year Sales Growth
    if "Year" in df.columns:
        yearly = df.groupby([key_col, "Year"])["Sales"].sum().unstack(fill_value=0.0)
        yoy_cols = []
        years = sorted(yearly.columns)
        for i in range(1, len(years)):
            y1 = years[i-1]
            y2 = years[i]
            growth = (yearly[y2] - yearly[y1]) / (yearly[y1] + 1.0)
            col_name = f"growth_{y1}_{y2}"
            yearly[col_name] = growth
            yoy_cols.append(col_name)
        if yoy_cols:
            features["YoY_Growth"] = yearly[yoy_cols].mean(axis=1)
        else:
            features["YoY_Growth"] = 0.0
    else:
        features["YoY_Growth"] = 0.0

    # 2. Monthly Sales Volatility (standard deviation of monthly sales)
    if "YearMonth" in df.columns:
        monthly_sales = df.groupby([key_col, "YearMonth"])["Sales"].sum().unstack(fill_value=0.0)
        features["Monthly_Sales_Volatility"] = monthly_sales.std(axis=1)
    else:
        features["Monthly_Sales_Volatility"] = 0.0

    # 3. Average Order Value (AOV)
    if "Order ID" in df.columns:
        order_counts = df.groupby(key_col)["Order ID"].nunique()
    else:
        order_counts = grp.size()
    features["Avg_Order_Value"] = features["Total_Sales"] / (order_counts + 1e-5)

    # Add metadata (Category and Sub-Category)
    for meta in ["Category", "Sub-Category"]:
        if meta in df.columns:
            features[meta] = grp[meta].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")

    return features


# ======================================================================
# 2. Scaling & Elbow Method
# ======================================================================
def scale_features(features: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Select numerical features for clustering, apply log transformations to skewed
    volume features, and standardize values using StandardScaler.
    """
    clustering_cols = ["Total_Sales", "YoY_Growth", "Monthly_Sales_Volatility", "Avg_Order_Value"]
    df_cluster = features[clustering_cols].copy()

    # Log transform right-skewed features
    df_cluster["Total_Sales"] = np.log1p(df_cluster["Total_Sales"])
    df_cluster["Monthly_Sales_Volatility"] = np.log1p(df_cluster["Monthly_Sales_Volatility"])
    df_cluster["Avg_Order_Value"] = np.log1p(df_cluster["Avg_Order_Value"])

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df_cluster)
    return scaled_matrix, scaler


def run_elbow_method(
    scaled_matrix: np.ndarray,
    max_k: int = 10,
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Calculate and plot within-cluster sum of squares (inertia) for k in [1, max_k]
    to aid in elbow analysis. Saves the chart to disk.
    """
    inertias = []
    ks = list(range(1, max_k + 1))

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_matrix)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, inertias, marker="o", linestyle="-", color="#1f77b4")
    ax.set_title("Demand Segmentation - KMeans Elbow Method", pad=15)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_xticks(ks)

    if save_path is None:
        save_path = PATHS.charts / "clustering_elbow.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Elbow Method plot to %s", save_path)

    return pd.DataFrame({"k": ks, "Inertia": inertias})


# ======================================================================
# 3. KMeans & Business Recommendations
# ======================================================================
# ======================================================================
# 3. KMeans & Business Recommendations
# ======================================================================
def get_recommendation(segment_label: str) -> Dict[str, str]:
    """Get stock and marketing strategy recommendations for a demand segment."""
    recommendations = {
        "High Volume, Stable Demand": {
            "Replenishment": "Automated replenishment (Min-Max system); maintain high safety stock.",
            "Marketing": "Loyalty programs, bulk discounts, and long-term customer contracts.",
            "Strategy": "Core revenue driver. Focus on supplier stability and cost-efficiency.",
        },
        "Low Volume, High Volatility": {
            "Replenishment": "Drop-ship model or make-to-order if possible; clear excess stock immediately.",
            "Marketing": "Clearance sales, deep discount bundles, or liquidations.",
            "Strategy": "Inventory risk. Assess whether to phase out these SKUs entirely.",
        },
        "Growing Demand": {
            "Replenishment": "Dynamic safety stock tied directly to monthly forecasts; build stock ahead of peak seasons.",
            "Marketing": "Pre-season promotions, early-booking discounts, and seasonal bundling.",
            "Strategy": "High-leverage. Plan warehouse capacity carefully around peaks.",
        },
        "Declining Demand": {
            "Replenishment": "Low, steady safety stock. Consolidate orders to reduce shipping costs.",
            "Marketing": "Cross-sell or bundle with 'Consistent Performers' to increase order value.",
            "Strategy": "Stable long-tail. Keep SKU overhead low; do not over-promote.",
        },
    }
    return recommendations.get(segment_label, {
        "Replenishment": "Standard inventory reviews.",
        "Marketing": "General promotions.",
        "Strategy": "Monitor performance.",
    })


def segment_demand(
    features: pd.DataFrame,
    scaled_matrix: np.ndarray,
    n_clusters: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform KMeans demand segmentation and map clusters to business labels.

    To keep labels consistent across runs, we sort cluster centers.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features = features.copy()
    features["Cluster"] = kmeans.fit_predict(scaled_matrix)

    # Let's map cluster indices to human readable business segments.
    # Group by Cluster and calculate mean features
    cluster_stats = features.groupby("Cluster").agg(
        mean_sales=("Total_Sales", "mean"),
        mean_volatility=("Monthly_Sales_Volatility", "mean"),
        mean_growth=("YoY_Growth", "mean")
    )

    # Label assignment based on sales volume and growth rate:
    # 1. Sort by sales volume: lowest sales -> Low Volume, High Volatility, highest sales -> High Volume, Stable Demand
    # 2. Of the middle two, the one with the higher growth rate -> Growing Demand, and the other -> Declining Demand
    sorted_clusters = cluster_stats.sort_values("mean_sales").index.tolist()
    
    label_map = {}
    label_map[sorted_clusters[3]] = "High Volume, Stable Demand"
    label_map[sorted_clusters[0]] = "Low Volume, High Volatility"
    
    mid1, mid2 = sorted_clusters[1], sorted_clusters[2]
    growth_mid1 = cluster_stats.loc[mid1, "mean_growth"]
    growth_mid2 = cluster_stats.loc[mid2, "mean_growth"]
    
    if growth_mid1 > growth_mid2:
        label_map[mid1] = "Growing Demand"
        label_map[mid2] = "Declining Demand"
    else:
        label_map[mid2] = "Growing Demand"
        label_map[mid1] = "Declining Demand"

    features["ClusterLabel"] = features["Cluster"].map(label_map)

    # Apply PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(scaled_matrix)

    pca_df = pd.DataFrame(
        pca_coords,
        columns=["PCA1", "PCA2"],
        index=features.index
    )
    pca_df["Cluster"] = features["Cluster"]
    pca_df["ClusterLabel"] = features["ClusterLabel"]
    pca_df["Total_Sales"] = features["Total_Sales"].round(2)
    pca_df["YoY_Growth"] = features["YoY_Growth"].round(4)
    pca_df["Monthly_Sales_Volatility"] = features["Monthly_Sales_Volatility"].round(3)
    pca_df["Avg_Order_Value"] = features["Avg_Order_Value"].round(2)
    if "Category" in features.columns:
        pca_df["Category"] = features["Category"]

    logger.info("Demand segmentation complete. Segment sizes: \n%s", features["ClusterLabel"].value_counts())
    return features, pca_df


def run_and_plot_segmentation(
    df: pd.DataFrame,
    n_clusters: int = 4,
    save_dir: Optional[Union[str, Path]] = None,
    interactive: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full clustering pipeline, save PCA plots and elbow plots,
    and return the results.
    """
    features = engineer_product_features(df)
    scaled_matrix, scaler = scale_features(features)

    if save_dir is None:
        save_dir = PATHS.charts
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Elbow Chart
    run_elbow_method(scaled_matrix, max_k=8, save_path=save_dir / "clustering_elbow.png")

    # 2. Run Segmentation
    segmented_features, pca_df = segment_demand(features, scaled_matrix, n_clusters=n_clusters)

    # 3. Save PCA Plot
    ext = "html" if interactive else "png"
    pca_plot_path = save_dir / f"demand_segmentation.{ext}"
    plot_clusters(
        segmented_features,
        pca_df,
        cluster_col="Cluster",
        label_col="ClusterLabel",
        save_path=pca_plot_path,
        interactive=interactive,
    )

    # 4. Save segmented products CSV
    segmented_features.to_csv(save_dir / "segmented_products.csv")
    logger.info("Saved segmented products table to %s", save_dir / "segmented_products.csv")

    return segmented_features, pca_df
