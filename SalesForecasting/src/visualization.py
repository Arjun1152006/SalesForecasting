"""
visualization.py
================

Centralized visualization module for the End-to-End Sales Forecasting &
Demand Intelligence System. Provides reusable functions for both Matplotlib
(static, report-friendly) and Plotly (interactive, dashboard-friendly) charts.

All charts can be optionally saved directly to the configured charts directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src.utils import PATHS, get_logger

logger = get_logger(__name__)

# Style settings for Matplotlib
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 150,
})

# Sleek design color palette
PRIMARY_COLOR = "#1f77b4"  # Cool Blue
SECONDARY_COLOR = "#ff7f0e"  # Warm Orange
ACCENT_COLOR = "#2ca02c"  # Soft Green
DARK_BG = "#111111"
LIGHT_BG = "#ffffff"


def _resolve_save_path(filename: str, save_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve and ensure the directory for saving a chart exists."""
    if save_path is None:
        target = PATHS.charts / filename
    else:
        target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


# ======================================================================
# 1. Monthly Sales Trends
# ======================================================================
def plot_monthly_sales(
    df: pd.DataFrame,
    date_col: str = "Date",
    sales_col: str = "Sales",
    save_path: Optional[Union[str, Path]] = None,
    interactive: bool = False,
) -> Union[go.Figure, plt.Figure]:
    """
    Plot monthly historical sales trends with rolling averages.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing monthly aggregated sales.
    date_col : str
        Date column name.
    sales_col : str
        Sales column name.
    save_path : Path, optional
        Path to save the chart.
    interactive : bool
        If True, returns a Plotly Figure, else returns a Matplotlib Figure.
    """
    df = df.copy().sort_values(by=date_col)
    df["Rolling_3M"] = df[sales_col].rolling(window=3, min_periods=1).mean()

    if interactive:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[sales_col],
            mode="lines+markers",
            name="Monthly Sales",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df["Rolling_3M"],
            mode="lines",
            name="3-Month Rolling Average",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        ))
        fig.update_layout(
            title="Monthly Sales Trend & 3-Month Rolling Average",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        if save_path or save_path is None:
            target = _resolve_save_path("monthly_sales_trend.html", save_path)
            fig.write_html(str(target))
            logger.info("Saved interactive monthly sales trend to %s", target)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[date_col], df[sales_col], marker="o", label="Monthly Sales", color="#1f77b4", linewidth=2.5)
        ax.plot(df[date_col], df["Rolling_3M"], linestyle="--", label="3-Month Rolling Average", color="#ff7f0e", linewidth=2)
        ax.set_title("Monthly Sales Trend & 3-Month Rolling Average", pad=15)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales ($)")
        ax.legend()
        plt.tight_layout()
        if save_path or save_path is None:
            target = _resolve_save_path("monthly_sales_trend.png", save_path)
            fig.savefig(target, dpi=300)
            logger.info("Saved static monthly sales trend to %s", target)
        return fig


# ======================================================================
# 2. Regional Sales Breakdown
# ======================================================================
def plot_regional_sales(
    df: pd.DataFrame,
    region_col: str = "Region",
    sales_col: str = "Sales",
    save_path: Optional[Union[str, Path]] = None,
    interactive: bool = False,
) -> Union[go.Figure, plt.Figure]:
    """
    Plot sales distribution across different regions.
    """
    reg_sales = df.groupby(region_col)[sales_col].sum().reset_index().sort_values(by=sales_col, ascending=False)

    if interactive:
        fig = px.bar(
            reg_sales,
            x=region_col,
            y=sales_col,
            title="Sales Distribution by Region",
            labels={region_col: "Region", sales_col: "Total Sales ($)"},
            color=sales_col,
            color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
        if save_path or save_path is None:
            target = _resolve_save_path("regional_sales.html", save_path)
            fig.write_html(str(target))
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=reg_sales, x=region_col, y=sales_col, ax=ax, palette="Blues_r")
        ax.set_title("Sales Distribution by Region", pad=15)
        ax.set_xlabel("Region")
        ax.set_ylabel("Total Sales ($)")
        plt.tight_layout()
        if save_path or save_path is None:
            target = _resolve_save_path("regional_sales.png", save_path)
            fig.savefig(target, dpi=300)
        return fig


# ======================================================================
# 3. Category Sales Breakdown
# ======================================================================
def plot_category_sales(
    df: pd.DataFrame,
    category_col: str = "Category",
    subcat_col: Optional[str] = "Sub-Category",
    sales_col: str = "Sales",
    save_path: Optional[Union[str, Path]] = None,
    interactive: bool = False,
) -> Union[go.Figure, plt.Figure]:
    """
    Plot sales distribution across product categories. If subcategory column is provided,
    creates a nested hierarchical plot (Sunburst/Treemap for Plotly, Grouped Bar for Matplotlib).
    """
    if interactive:
        if subcat_col and subcat_col in df.columns:
            # Sunburst chart for hierarchy
            agg_df = df.groupby([category_col, subcat_col])[sales_col].sum().reset_index()
            fig = px.sunburst(
                agg_df,
                path=[category_col, subcat_col],
                values=sales_col,
                title="Sales Hierarchy by Category & Sub-Category",
                color=sales_col,
                color_continuous_scale="Viridis",
            )
        else:
            agg_df = df.groupby(category_col)[sales_col].sum().reset_index()
            fig = px.pie(
                agg_df,
                names=category_col,
                values=sales_col,
                title="Sales Contribution by Category",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
        fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
        if save_path or save_path is None:
            target = _resolve_save_path("category_sales.html", save_path)
            fig.write_html(str(target))
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        if subcat_col and subcat_col in df.columns:
            agg_df = (
                df.groupby([category_col, subcat_col])[sales_col]
                .sum()
                .reset_index()
                .sort_values(by=sales_col, ascending=False)
            )
            sns.barplot(data=agg_df, x=sales_col, y=subcat_col, hue=category_col, ax=ax, dodge=False)
            ax.set_title("Sales by Sub-Category", pad=15)
            ax.set_xlabel("Sales ($)")
            ax.set_ylabel("Sub-Category")
        else:
            agg_df = df.groupby(category_col)[sales_col].sum().reset_index().sort_values(by=sales_col, ascending=False)
            sns.barplot(data=agg_df, x=category_col, y=sales_col, ax=ax, palette="viridis")
            ax.set_title("Sales by Category", pad=15)
            ax.set_xlabel("Category")
            ax.set_ylabel("Sales ($)")
        plt.tight_layout()
        if save_path or save_path is None:
            target = _resolve_save_path("category_sales.png", save_path)
            fig.savefig(target, dpi=300)
        return fig


# ======================================================================
# 4. Heatmaps
# ======================================================================
def plot_sales_heatmap(
    df: pd.DataFrame,
    index_col: str = "MonthName",
    columns_col: str = "DayName",
    values_col: str = "Sales",
    save_path: Optional[Union[str, Path]] = None,
    interactive: bool = False,
) -> Union[go.Figure, plt.Figure]:
    """
    Generate a heatmap of sales activity, e.g., Month vs Day of Week.
    """
    # Create pivot table
    pivot_df = df.pivot_table(index=index_col, columns=columns_col, values=values_col, aggfunc="sum").fillna(0.0)

    # Sort months/days if standard calendar strings are used
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    days_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    existing_months = [m for m in months_order if m in pivot_df.index]
    if existing_months:
        pivot_df = pivot_df.reindex(existing_months)

    existing_days = [d for d in days_order if d in pivot_df.columns]
    if existing_days:
        pivot_df = pivot_df[existing_days]

    if interactive:
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale="YlOrRd",
            colorbar=dict(title="Sales ($)"),
        ))
        fig.update_layout(
            title=f"Sales Intensity Heatmap ({index_col} vs {columns_col})",
            xaxis_title=columns_col,
            yaxis_title=index_col,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        if save_path or save_path is None:
            target = _resolve_save_path("sales_heatmap.html", save_path)
            fig.write_html(str(target))
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Sales ($)"})
        ax.set_title(f"Sales Intensity Heatmap ({index_col} vs {columns_col})", pad=15)
        ax.set_xlabel(columns_col)
        ax.set_ylabel(index_col)
        plt.tight_layout()
        if save_path or save_path is None:
            target = _resolve_save_path("sales_heatmap.png", save_path)
            fig.savefig(target, dpi=300)
        return fig


# ======================================================================
# 5. Forecast Charts
# ======================================================================
def plot_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    date_col: str = "Date",
    sales_col: str = "Sales",
    title: str = "Sales Forecast",
    save_path: Optional[Union[str, Path]] = None,
    interactive: bool = False,
) -> Union[go.Figure, plt.Figure]:
    """
    Plot historical sales alongside forecasts and uncertainty intervals.

    Parameters
    ----------
    historical_df : pd.DataFrame
        DataFrame with columns [date_col, sales_col]
    forecast_df : pd.DataFrame
        DataFrame with columns [date_col, "Forecast", "Lower_CI", "Upper_CI"]
    """
    historical_df = historical_df.copy().sort_values(by=date_col)
    forecast_df = forecast_df.copy().sort_values(by=date_col)

    if interactive:
        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(
            x=historical_df[date_col],
            y=historical_df[sales_col],
            mode="lines+markers",
            name="Historical Sales",
            line=dict(color="#1f77b4", width=2.5),
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df[date_col],
            y=forecast_df["Forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#d62728", width=2.5),
        ))

        # Confidence bounds (shaded area)
        if "Lower_CI" in forecast_df.columns and "Upper_CI" in forecast_df.columns:
            # Standard Plotly fill pattern requires concatenation or trace ordering
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_df[date_col], forecast_df[date_col][::-1]]),
                y=pd.concat([forecast_df["Upper_CI"], forecast_df["Lower_CI"][::-1]]),
                fill="toself",
                fillcolor="rgba(214, 39, 40, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Confidence Interval",
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        if save_path or save_path is None:
            sanitized = title.lower().replace(" ", "_")
            target = _resolve_save_path(f"forecast_{sanitized}.html", save_path)
            fig.write_html(str(target))
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical_df[date_col], historical_df[sales_col], marker="o", color="#1f77b4", label="Historical Sales", linewidth=2)
        ax.plot(forecast_df[date_col], forecast_df["Forecast"], marker="x", color="#d62728", label="Forecast", linewidth=2)

        if "Lower_CI" in forecast_df.columns and "Upper_CI" in forecast_df.columns:
            ax.fill_between(
                forecast_df[date_col],
                forecast_df["Lower_CI"],
                forecast_df["Upper_CI"],
                color="#d62728",
                alpha=0.15,
                label="95% Confidence Interval",
            )

        ax.set_title(title, pad=15)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales ($)")
        ax.legend()
        plt.tight_layout()
        if save_path or save_path is None:
            sanitized = title.lower().replace(" ", "_")
            target = _resolve_save_path(f"forecast_{sanitized}.png", save_path)
            fig.savefig(target, dpi=300)
        return fig


# ======================================================================
# 6. Cluster Plots
# ======================================================================
def plot_clusters(
    df: pd.DataFrame,
    pca_df: pd.DataFrame,
    cluster_col: str = "Cluster",
    label_col: str = "ClusterLabel",
    save_path: Optional[Union[str, Path]] = None,
    interactive: bool = False,
) -> Union[go.Figure, plt.Figure]:
    """
    Plot demand segmentation clusters on a 2D PCA projection.

    Parameters
    ----------
    df : pd.DataFrame
        Original features dataframe.
    pca_df : pd.DataFrame
        DataFrame with columns ["PCA1", "PCA2", cluster_col, label_col]
        and index/hover information.
    """
    if interactive:
        fig = px.scatter(
            pca_df,
            x="PCA1",
            y="PCA2",
            color=label_col,
            hover_data={
                c: True for c in pca_df.columns if c not in ["PCA1", "PCA2"]
            },
            title="Product Demand Segmentation (PCA 2D Projection)",
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color="DarkSlateGrey")))
        fig.update_layout(
            legend_title="Demand Segment",
            margin=dict(l=40, r=40, t=50, b=40),
            hoverlabel=dict(bgcolor="white", font_size=12),
        )
        if save_path or save_path is None:
            target = _resolve_save_path("demand_segmentation.html", save_path)
            fig.write_html(str(target))
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        # Unique labels
        labels = pca_df[label_col].unique()
        palette = sns.color_palette("Set2", len(labels))

        for i, lbl in enumerate(labels):
            sub_pca = pca_df[pca_df[label_col] == lbl]
            ax.scatter(
                sub_pca["PCA1"],
                sub_pca["PCA2"],
                label=lbl,
                color=palette[i],
                alpha=0.8,
                edgecolors="none",
                s=50,
            )

        ax.set_title("Product Demand Segmentation (PCA 2D Projection)", pad=15)
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.legend(title="Demand Segment", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        if save_path or save_path is None:
            target = _resolve_save_path("demand_segmentation.png", save_path)
            fig.savefig(target, dpi=300, bbox_inches="tight")
        return fig


# ======================================================================
# 7. Anomaly Charts
# ======================================================================
def plot_anomalies(
    df: pd.DataFrame,
    date_col: str = "Date",
    sales_col: str = "Sales",
    anomaly_col: Union[str, list, tuple] = "Is_Anomaly",
    save_path: Optional[Union[str, Path]] = None,
    interactive: bool = False,
) -> Union[go.Figure, plt.Figure]:
    """
    Plot time series showing sales volume with anomalous points highlighted.
    Supports list/tuple of anomaly_col to display subplots for each method.
    """
    df = df.copy().sort_values(by=date_col)

    if isinstance(anomaly_col, (list, tuple)):
        # Multi-method subplots
        if interactive:
            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=len(anomaly_col),
                cols=1,
                shared_xaxes=True,
                subplot_titles=[f"Outlier Detection Method: {col}" for col in anomaly_col]
            )
            for idx, col in enumerate(anomaly_col):
                anomalies = df[df[col] == 1]
                fig.add_trace(go.Scatter(
                    x=df[date_col],
                    y=df[sales_col],
                    mode="lines",
                    name="Normal Sales",
                    line=dict(color="#1f77b4", width=1.5),
                    showlegend=(idx == 0)
                ), row=idx+1, col=1)
                fig.add_trace(go.Scatter(
                    x=anomalies[date_col],
                    y=anomalies[sales_col],
                    mode="markers",
                    name=f"{col} Anomaly",
                    marker=dict(color="#d62728" if idx == 0 else "#ff7f0e", size=8),
                    showlegend=(idx == 0)
                ), row=idx+1, col=1)
            
            fig.update_layout(
                title="Comparative Sales Anomalies by Method",
                height=300 * len(anomaly_col),
                margin=dict(l=40, r=40, t=60, b=40),
                hovermode="x unified",
            )
            if save_path:
                target = _resolve_save_path("sales_anomalies.html", save_path)
                fig.write_html(str(target))
            return fig
        else:
            fig, axes = plt.subplots(len(anomaly_col), 1, figsize=(12, 4.5 * len(anomaly_col)), sharex=True)
            for idx, col in enumerate(anomaly_col):
                ax = axes[idx]
                anomalies = df[df[col] == 1]
                ax.plot(df[date_col], df[sales_col], color="#1f77b4", label="Sales Trend", linewidth=1.5)
                ax.scatter(anomalies[date_col], anomalies[sales_col], color="#d62728" if idx == 0 else "#ff7f0e",
                           label=f"{col} Outlier", s=40, zorder=5)
                ax.set_title(f"Anomaly Detection: {col}", pad=10)
                ax.set_ylabel("Sales ($)")
                ax.legend()
            
            plt.xlabel("Date")
            plt.tight_layout()
            if save_path:
                target = _resolve_save_path("sales_anomalies.png", save_path)
                fig.savefig(target, dpi=300)
            return fig
    else:
        # Single column plot
        anomalies = df[df[anomaly_col] == 1]
        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[sales_col],
                mode="lines",
                name="Normal Sales",
                line=dict(color="#1f77b4", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=anomalies[date_col],
                y=anomalies[sales_col],
                mode="markers",
                name="Detected Anomaly",
                marker=dict(color="#d62728", size=9, symbol="circle"),
            ))
            fig.update_layout(
                title="Sales Anomalies Over Time",
                xaxis_title="Date",
                yaxis_title="Sales ($)",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode="x unified",
                margin=dict(l=40, r=40, t=50, b=40),
            )
            if save_path:
                target = _resolve_save_path("sales_anomalies.html", save_path)
                fig.write_html(str(target))
            return fig
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df[date_col], df[sales_col], color="#1f77b4", label="Sales Trend", linewidth=1.5)
            ax.scatter(anomalies[date_col], anomalies[sales_col], color="#d62728", label="Anomaly Spike/Drop", s=50, zorder=5)
            ax.set_title("Sales Anomalies Over Time", pad=15)
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales ($)")
            ax.legend()
            plt.tight_layout()
            if save_path:
                target = _resolve_save_path("sales_anomalies.png", save_path)
                fig.savefig(target, dpi=300)
            return fig


