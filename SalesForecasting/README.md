# End-to-End Sales Forecasting & Demand Intelligence System

A production-grade data science system for retail/product sales analytics, combining
**time series forecasting**, **anomaly detection**, and **demand segmentation** into a
single decision-support platform — delivered as a research notebook, a reusable Python
package, an interactive Streamlit dashboard, and an executive report.

---

## 1. Business Problem

Retail and e-commerce businesses lose revenue when they:

- **Under-forecast demand** → stockouts, lost sales, disappointed customers
- **Over-forecast demand** → excess inventory, tied-up capital, markdowns
- **Miss abnormal sales events** → fraud, supply shocks, viral demand, or data errors go unnoticed
- **Treat all products the same** → high-velocity and slow-moving SKUs need different
  replenishment, marketing, and pricing strategies

This system addresses all four problems in one pipeline.

## 2. What This Project Delivers

| Capability | Technique | Business Value |
|---|---|---|
| Sales Forecasting | SARIMA, Prophet, XGBoost (auto-selected by lowest error) | Plan inventory & staffing 3 months ahead |
| Trend/Seasonality Decomposition | STL decomposition, ADF test, ACF/PACF | Understand *why* sales move the way they do |
| Category & Region Forecasting | Per-segment time series models | Localized, actionable planning |
| Anomaly Detection | Isolation Forest, Rolling Mean, Z-score (ensemble) | Catch spikes/drops before they become problems |
| Demand Segmentation | KMeans + PCA | Tier products/SKUs for differentiated strategy |
| Interactive Dashboard | Streamlit + Plotly | Self-service exploration for non-technical stakeholders |
| Executive Report | python-docx generated `.docx` | Board/stakeholder-ready summary, zero jargon |

## 3. Project Structure

```
SalesForecasting_Kanakalakshmi/
│
├── README.md                  # This file
├── requirements.txt           # Pinned dependencies
├── analysis.ipynb             # Full research notebook (EDA → modeling → insights)
├── app.py                     # Streamlit dashboard entry point
├── summary.docx               # Auto-generated executive report (Phase 9)
│
├── charts/                    # All exported static/plotly charts (PNG/HTML)
├── assets/                    # Dashboard assets (logos, custom CSS, icons)
├── models/                    # Serialized trained models (joblib/pickle)
├── data/
│   ├── train.csv              # Superstore-style transactional sales data
│   └── videogame_sales.csv    # Video game sales dataset (secondary domain example)
│
├── src/                       # Reusable, importable business-logic modules
│   ├── __init__.py
│   ├── preprocessing.py       # Data cleaning, validation, feature engineering
│   ├── forecasting.py         # SARIMA / Prophet / XGBoost forecasting engine
│   ├── anomaly.py             # Isolation Forest / rolling stats / z-score detectors
│   ├── clustering.py          # KMeans demand segmentation + PCA
│   ├── visualization.py       # Centralized Plotly/Matplotlib chart factory
│   └── utils.py               # Logging, config, IO helpers, custom exceptions
│
└── screenshots/               # Dashboard screenshots for README/portfolio
```

## 4. Tech Stack

- **Data**: Pandas, NumPy
- **Time Series**: Statsmodels (SARIMA, STL, ADF), Prophet
- **ML**: Scikit-learn (KMeans, PCA, Isolation Forest), XGBoost
- **Viz**: Plotly, Matplotlib, Seaborn
- **App**: Streamlit
- **Reporting**: python-docx
- **Persistence**: Joblib

## 5. How to Run

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place data files in data/
#    data/train.csv
#    data/videogame_sales.csv

# 4. Explore the full analysis
jupyter notebook analysis.ipynb

# 5. Launch the dashboard
streamlit run app.py
```

## 6. Project Roadmap (Build Phases)

- [x] **Phase 1** — Project scaffolding, README, requirements, `preprocessing.py`, notebook skeleton
- [ ] **Phase 2** — Data cleaning, EDA, feature engineering, chart exports, business insights
- [ ] **Phase 3** — Time series decomposition: trend, seasonality, residuals, ADF, ACF/PACF
- [ ] **Phase 4** — SARIMA / Prophet / XGBoost forecasting + model comparison
- [ ] **Phase 5** — Category & regional forecasting
- [ ] **Phase 6** — Anomaly detection (Isolation Forest, rolling mean, z-score)
- [ ] **Phase 7** — Demand segmentation (KMeans, elbow method, PCA)
- [ ] **Phase 8** — Streamlit dashboard
- [ ] **Phase 9** — Executive report (`summary.docx`)

## 7. Author

Arjun Peddi — Data Science Internship Project

## 8. License

MIT License — free to use for portfolio and educational purposes.
