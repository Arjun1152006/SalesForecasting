"""
app.py
======

Interactive Streamlit dashboard for the End-to-End Sales Forecasting & Demand
Intelligence System. Contains four dashboards:
1. Sales Dashboard (KPIs, regional, and categorical distributions)
2. Forecast Explorer (3-month forecast visualizations, adjustable horizons, metrics)
3. Anomaly Detection (consensus voting thresholds, anomaly time-series, tables)
4. Product Demand Segmentation (PCA cluster visualizations, replenishment rules)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils import PATHS
from src.anomaly import detect_anomalies
from src.clustering import get_recommendation

# Set page config
st.set_page_config(
    page_title="End-to-End Sales Forecasting & Demand Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Charcoal Dark Side, Clean White Card Grid Layout)
st.markdown("""
<style>
    /* Main body background styling */
    .reportview-container {
        background: #f5f7fa;
    }
    
    /* Top Header styling */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1b365d;
        font-weight: 700;
        font-size: 38px;
        margin-bottom: 2px;
    }
    
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #7b8a97;
        font-size: 16px;
        margin-bottom: 25px;
    }
    
    /* KPI Card styling */
    .kpi-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    
    .kpi-title {
        font-size: 14px;
        color: #7b8a97;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        font-size: 28px;
        color: #1b365d;
        font-weight: 700;
        margin-top: 5px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-image: linear-gradient(180deg, #1b365d 0%, #11223b 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# Data Loading & Caching Helpers
# ======================================================================
@st.cache_data
def load_clean_data():
    """Load the preprocessed transactional sales data."""
    path = PATHS.data / "train_clean.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Order Date", "Ship Date"])
        return df
    return None


@st.cache_data
def load_monthly_sales():
    """Load aggregated monthly sales."""
    path = PATHS.data / "monthly_sales.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"])
        return df
    return None


@st.cache_data
def load_weekly_sales():
    """Load aggregated weekly sales."""
    path = PATHS.data / "weekly_sales.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"])
        return df
    return None


def get_saved_model():
    """Attempt to load the saved forecasting model."""
    path = PATHS.models / "best_forecaster.joblib"
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None


def get_segmented_products():
    """Load clustered product segments."""
    path = PATHS.data / "segmented_products.csv"
    if path.exists():
        return pd.read_csv(path, index_col="Product Name")
    return None


# ======================================================================
# Main Layout / Sidebar Navigation
# ======================================================================
st.sidebar.markdown(
    "<h2 style='color:#1f77b4; font-weight:700; margin-bottom:5px;'>NAVIGATOR</h2>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Go to page:",
    ["Sales Dashboard", "Forecast Explorer", "Anomaly Detection", "Product Demand Segmentation"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:12px; color:#7b8a97;'>Author: <b>Arjun Peddi</b><br>Role: Senior ML Engineer</div>",
    unsafe_allow_html=True
)

# Load data
df_clean = load_clean_data()
df_monthly = load_monthly_sales()
df_weekly = load_weekly_sales()

# Graceful degradation check
if df_clean is None or df_monthly is None or df_weekly is None:
    st.markdown("<div class='main-title'>Demand Intelligence Portal</div>", unsafe_allow_html=True)
    st.warning("⚠️ Baseline pipeline files not found in the `data/` directory.")
    st.info(
        "Please run the pipeline script first to generate clean datasets and serialize models:\n\n"
        "```bash\n"
        "python run_pipeline.py\n"
        "```"
    )
    st.stop()


# ======================================================================
# PAGE 1: Sales Dashboard
# ======================================================================
if page == "Sales Dashboard":
    st.markdown("<div class='main-title'>SALES DASHBOARD</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Enterprise Sales Performance & Key Distribution Metrics</div>", unsafe_allow_html=True)

    # Sidebar Filter Controls
    st.sidebar.markdown("### Dashboard Filters")
    regions = ["All"] + sorted(df_clean["Region"].unique().tolist())
    categories = ["All"] + sorted(df_clean["Category"].unique().tolist())
    
    selected_region = st.sidebar.selectbox("Select Region:", regions)
    selected_category = st.sidebar.selectbox("Select Category:", categories)

    # Filter dataset
    df_filtered = df_clean.copy()
    if selected_region != "All":
        df_filtered = df_filtered[df_filtered["Region"] == selected_region]
    if selected_category != "All":
        df_filtered = df_filtered[df_filtered["Category"] == selected_category]

    # Recalculate monthly trend on filtered data
    # Aggregate transaction levels to monthly series
    monthly_trend = (
        df_filtered.set_index("Order Date")
        .resample("MS")["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Order Date": "Date"})
    )
    monthly_trend["Rolling_3M"] = monthly_trend["Sales"].rolling(window=3, min_periods=1).mean()

    # KPI Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    total_revenue = df_filtered["Sales"].sum()
    total_orders = len(df_filtered)
    avg_order_value = df_filtered["Sales"].mean()
    profit_margin = df_filtered["ProfitMargin"].mean() if "ProfitMargin" in df_filtered.columns else 0.0

    with kpi_col1:
        st.markdown(
            f"<div class='kpi-container'>"
            f"<div class='kpi-title'>Total Sales</div>"
            f"<div class='kpi-value'>${total_revenue:,.2f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with kpi_col2:
        st.markdown(
            f"<div class='kpi-container' style='border-left-color: #ff7f0e;'>"
            f"<div class='kpi-title'>Order Count</div>"
            f"<div class='kpi-value'>{total_orders:,}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with kpi_col3:
        st.markdown(
            f"<div class='kpi-container' style='border-left-color: #2ca02c;'>"
            f"<div class='kpi-title'>Average Order Value</div>"
            f"<div class='kpi-value'>${avg_order_value:,.2f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with kpi_col4:
        st.markdown(
            f"<div class='kpi-container' style='border-left-color: #d62728;'>"
            f"<div class='kpi-title'>Avg Profit Margin</div>"
            f"<div class='kpi-value'>{profit_margin:.2%}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Charts Grid
    st.markdown("### Sales Trend & Distribution Breakdown")
    row_col1, row_col2 = st.columns([2, 1])

    with row_col1:
        # Line chart of monthly sales
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_trend["Date"],
            y=monthly_trend["Sales"],
            mode="lines+markers",
            name="Monthly Sales",
            line=dict(color="#1f77b4", width=3),
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_trend["Date"],
            y=monthly_trend["Rolling_3M"],
            mode="lines",
            name="3-Month Rolling Avg",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        ))
        fig_trend.update_layout(
            title="Monthly Sales & 3-Month Rolling Average (Filtered Data)",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with row_col2:
        # Pie chart / Sunburst of Product Category sales
        if "Sub-Category" in df_filtered.columns:
            agg_cat = df_filtered.groupby(["Category", "Sub-Category"])["Sales"].sum().reset_index()
            fig_cat = px.sunburst(
                agg_cat,
                path=["Category", "Sub-Category"],
                values="Sales",
                color="Sales",
                color_continuous_scale="Viridis",
            )
            fig_cat.update_layout(
                title="Sales Breakdown by Category Hierarchy",
                margin=dict(l=10, r=10, t=40, b=10)
            )
        else:
            agg_cat = df_filtered.groupby("Category")["Sales"].sum().reset_index()
            fig_cat = px.pie(agg_cat, names="Category", values="Sales", hole=0.4)
            fig_cat.update_layout(
                title="Sales Breakdown by Category",
                margin=dict(l=10, r=10, t=40, b=10)
            )
        st.plotly_chart(fig_cat, use_container_width=True)

    # Secondary Breakdown Row (Region vs Heatmap)
    st.markdown("### Geographical & Temporal Breakdowns")
    row2_col1, row2_col2 = st.columns([1, 1])

    with row2_col1:
        # Region distribution
        reg_sales = df_filtered.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
        fig_reg = px.bar(
            reg_sales,
            x="Region",
            y="Sales",
            color="Sales",
            color_continuous_scale="Blues",
            labels={"Sales": "Total Revenue ($)"}
        )
        fig_reg.update_layout(
            title="Revenue Contribution by Region",
            margin=dict(l=30, r=30, t=40, b=30),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    with row2_col2:
        # Heatmap of Sales intensity (Month vs Day of Week)
        pivot_sales = df_filtered.pivot_table(index="MonthName", columns="DayName", values="Sales", aggfunc="sum").fillna(0.0)
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        days_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        existing_months = [m for m in months_order if m in pivot_sales.index]
        existing_days = [d for d in days_order if d in pivot_sales.columns]
        if existing_months:
            pivot_sales = pivot_sales.reindex(existing_months)
        if existing_days:
            pivot_sales = pivot_sales[existing_days]

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot_sales.values,
            x=pivot_sales.columns,
            y=pivot_sales.index,
            colorscale="YlOrRd"
        ))
        fig_heat.update_layout(
            title="Temporal Demand Heatmap (Month vs Day of Week)",
            margin=dict(l=30, r=30, t=40, b=30),
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ======================================================================
# PAGE 2: Forecast Explorer
# ======================================================================
elif page == "Forecast Explorer":
    st.markdown("<div class='main-title'>FORECAST EXPLORER</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Simulate 3-Month Inventory and Sales Horizons</div>", unsafe_allow_html=True)

    # Load serialized model
    best_model = get_saved_model()
    
    if best_model is None:
        st.error("No saved forecasting model found.")
        st.info("Please execute the training pipeline first: `python run_pipeline.py`.")
        st.stop()

    model_name = best_model.__class__.__name__.replace("Forecaster", "")

    st.success(f"🤖 **Active Model:** Automated selection chose **{model_name}** as the primary forecaster.")

    # Horizon slider
    st.markdown("### Forecast Settings")
    col1, col2 = st.columns([1, 2])
    with col1:
        horizon = st.slider("Select Forecast Horizon (Months):", min_value=1, max_value=6, value=3)
        
        # Calculate dynamic prediction
        forecast_df = best_model.predict(horizon=horizon)
        
        # Display forecast values in table
        st.markdown("#### Forecast Output Data")
        disp_fc = forecast_df.copy()
        disp_fc["Date"] = pd.to_datetime(disp_fc["Date"]).dt.strftime("%Y-%b")
        disp_fc = disp_fc.rename(columns={"Forecast": "Pred Sales ($)", "Lower_CI": "Lower Limit ($)", "Upper_CI": "Upper Limit ($)"})
        st.dataframe(disp_fc.style.format(precision=2), height=200)

        # Download CSV link
        csv_data = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast CSV",
            data=csv_data,
            file_name=f"sales_forecast_{horizon}m.csv",
            mime="text/csv",
        )
        
    with col2:
        # Plot forecast dynamically
        fig_fc = go.Figure()
        # Historical line
        fig_fc.add_trace(go.Scatter(
            x=df_monthly["Date"],
            y=df_monthly["Sales"],
            mode="lines+markers",
            name="Historical Sales",
            line=dict(color="#1f77b4", width=2.5),
        ))

        # Forecast line
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast"],
            mode="lines+markers",
            name="Predicted Sales",
            line=dict(color="#d62728", width=2.5),
        ))

        # Confidence limits
        if "Lower_CI" in forecast_df.columns and "Upper_CI" in forecast_df.columns:
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
                y=pd.concat([forecast_df["Upper_CI"], forecast_df["Lower_CI"][::-1]]),
                fill="toself",
                fillcolor="rgba(214, 39, 40, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="95% Confidence Interval",
            ))

        fig_fc.update_layout(
            title=f"{horizon}-Month Sales Projection Horizon (Model: {model_name})",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=40, r=40, t=50, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fc, use_container_width=True)


# ======================================================================
# PAGE 3: Anomaly Detection
# ======================================================================
elif page == "Anomaly Detection":
    st.markdown("<div class='main-title'>ANOMALY DETECTION</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Real-Time Outlier Identification & Fraud/Spike Prevention</div>", unsafe_allow_html=True)

    st.sidebar.markdown("### Anomaly Model Controls")
    z_thresh = st.sidebar.slider("Z-Score Standard Deviation Limit:", min_value=1.5, max_value=4.0, value=2.0, step=0.1)
    rolling_wind = st.sidebar.slider("Rolling Mean Window (Weeks):", min_value=2, max_value=24, value=8, step=1)
    rolling_k_mult = st.sidebar.slider("Rolling Std Dev Multiplier:", min_value=1.5, max_value=4.0, value=2.0, step=0.1)
    contamination = st.sidebar.slider("IForest Contamination Rate:", min_value=0.01, max_value=0.10, value=0.03, step=0.01)

    # Dynamically recompute anomalies based on user thresholds!
    # This matches "Implement rolling mean, z-score, Isolation Forest" in interactive app.
    with st.spinner("Recomputing anomalies with user thresholds..."):
        df_anom, table_anom = detect_anomalies(
            df=df_weekly,
            date_col="Date",
            sales_col="Sales",
            iforest_contamination=contamination,
            rolling_window=rolling_wind,
            rolling_k=rolling_k_mult,
            z_threshold=z_thresh,
        )

    # KPI blocks
    anom_col1, anom_col2, anom_col3 = st.columns(3)
    
    with anom_col1:
        st.markdown(
            f"<div class='kpi-container' style='border-left-color: #d62728;'>"
            f"<div class='kpi-title'>Consensus Anomalies</div>"
            f"<div class='kpi-value'>{len(table_anom)}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with anom_col2:
        prop_anom = len(table_anom) / len(df_weekly)
        st.markdown(
            f"<div class='kpi-container' style='border-left-color: #ff7f0e;'>"
            f"<div class='kpi-title'>Anomaly Ratio</div>"
            f"<div class='kpi-value'>{prop_anom:.2%}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with anom_col3:
        max_spike = table_anom["Sales"].max() if not table_anom.empty else 0.0
        st.markdown(
            f"<div class='kpi-container' style='border-left-color: #1f77b4;'>"
            f"<div class='kpi-title'>Max Outlier Spike</div>"
            f"<div class='kpi-value'>${max_spike:,.2f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Anomaly Chart
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=df_anom["Date"],
        y=df_anom["Sales"],
        mode="lines",
        name="Weekly Sales",
        line=dict(color="#1f77b4", width=1.5),
    ))
    
    anomalies_only = df_anom[df_anom["Is_Anomaly"] == 1]
    fig_anom.add_trace(go.Scatter(
        x=anomalies_only["Date"],
        y=anomalies_only["Sales"],
        mode="markers",
        name="Consensus Anomaly Spike",
        marker=dict(color="#d62728", size=8, symbol="circle"),
    ))
    fig_anom.update_layout(
        title="Consensus Anomaly Detections on Weekly Sales Series",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    # Table & Details
    st.markdown("### Anomaly Event Audit Log")
    if not table_anom.empty:
        disp_table = table_anom[["Date", "Sales", "Rolling_Mean", "Z_Score", "Anomaly_Vote_Count"]].copy()
        disp_table["Date"] = pd.to_datetime(disp_table["Date"]).dt.strftime("%Y-%m-%d")
        disp_table = disp_table.rename(columns={"Sales": "Sales ($)", "Rolling_Mean": "Local Average ($)", "Anomaly_Vote_Count": "Ensemble Vote Score (Max 3)"})
        st.dataframe(disp_table.style.format(precision=2), use_container_width=True)
    else:
        st.info("No anomalies detected at current threshold configurations.")


# ======================================================================
# PAGE 4: Product Demand Segmentation
# ======================================================================
elif page == "Product Demand Segmentation":
    st.markdown("<div class='main-title'>PRODUCT DEMAND SEGMENTATION</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Catalog Clustering & Actionable Replenishment Strategy</div>", unsafe_allow_html=True)

    # Load clusters
    df_clustered = get_segmented_products()
    
    if df_clustered is None:
        st.error("No clustered product segment files found.")
        st.info("Please execute the main pipeline first: `python run_pipeline.py`.")
        st.stop()

    st.markdown("### PCA Segment Clusters")
    
    # Render interactive Plotly scatter of cluster projection
    # Recompute PCA coords if not loaded or read from file
    pca_file = PATHS.charts / "demand_segmentation.html"
    
    # We can rebuild the Plotly figure directly in streamlit to match filters!
    pca_plot_df = df_clustered.reset_index()
    # Check if PCA coords are columns. If not, recalculate PCA
    if "PCA1" not in pca_plot_df.columns:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        features_cols = ["Total_Sales", "Order_Count", "Sales_Volatility", "Sales_Trend"]
        df_cluster = pca_plot_df[features_cols].copy()
        df_cluster["Total_Sales"] = np.log1p(df_cluster["Total_Sales"])
        df_cluster["Order_Count"] = np.log1p(df_cluster["Order_Count"])
        X_scaled = StandardScaler().fit_transform(df_cluster)
        coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
        pca_plot_df["PCA1"] = coords[:, 0]
        pca_plot_df["PCA2"] = coords[:, 1]

    fig_pca = px.scatter(
        pca_plot_df,
        x="PCA1",
        y="PCA2",
        color="ClusterLabel",
        hover_data=["Product Name", "Total_Sales", "Order_Count", "Sales_Volatility", "Sales_Trend"],
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig_pca.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.5, color="DarkSlateGrey")))
    fig_pca.update_layout(
        title="K-Means Catalog Segments Projected on 2D PCA Space",
        legend_title="Demand Segment",
        margin=dict(l=40, r=40, t=50, b=40),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # Dynamic Strategy lookup
    st.markdown("### Custom Segment Replenishment Playbook")
    
    segments = sorted(df_clustered["ClusterLabel"].unique().tolist())
    selected_seg = st.selectbox("Select Demand Segment:", segments)

    # Recommendations
    # Strip Tier B suffix if present to match get_recommendation key
    clean_key = selected_seg.split(" (")[0]
    rec = get_recommendation(clean_key)
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    with rec_col1:
        st.info(f"📦 **Replenishment Strategy**\n\n{rec['Replenishment']}")
    with rec_col2:
        st.success(f"📣 **Marketing Strategy**\n\n{rec['Marketing']}")
    with rec_col3:
        st.warning(f"🎯 **Operational Strategy**\n\n{rec['Strategy']}")

    # Filter product list
    products_in_seg = df_clustered[df_clustered["ClusterLabel"] == selected_seg].copy()
    
    st.markdown(f"#### Products in {selected_seg} ({len(products_in_seg)} items)")
    # Show summary details
    disp_prod = products_in_seg[["Category", "Total_Sales", "Order_Count", "Sales_Volatility", "Sales_Trend"]].copy()
    disp_prod = disp_prod.rename(columns={"Total_Sales": "Revenue ($)", "Order_Count": "Order Frequency", "Sales_Volatility": "Volatility Index", "Sales_Trend": "Growth Slope"})
    st.dataframe(disp_prod.style.format(precision=2), use_container_width=True)
