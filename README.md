## Stock Market Analysis & Price Prediction — ML Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat&logo=scikit-learn)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange?style=flat&logo=mysql)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?style=flat&logo=pandas)
![Joblib](https://img.shields.io/badge/Model-Joblib%20Serialized-yellow?style=flat)
![Domain](https://img.shields.io/badge/Domain-Financial%20Markets-lightblue?style=flat)

An end-to-end **stock market machine learning pipeline** that predicts adjusted closing prices for **Apple (AAPL), Microsoft (MSFT), Netflix (NFLX), and Google (GOOG)**. The project combines advanced feature engineering (moving averages, rolling volatility, lag features), a Random Forest Regressor with sklearn Pipelines, MySQL integration, Excel export, and an interactive Streamlit dashboard.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Feature Engineering](#feature-engineering)
- [ML Pipeline Architecture](#ml-pipeline-architecture)
- [Model Accuracy Results](#model-accuracy-results)
- [Streamlit Dashboard](#streamlit-dashboard)
- [MySQL Tables](#mysql-tables)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Key Insights](#key-insights)
- [References](#references)

## Project Overview
Predicting stock prices requires careful feature engineering from time-series data — raw OHLCV data alone is insufficient. This project:
- Cleans and validates multi-ticker stock CSV data
- Engineers **12 predictive features** including moving averages (7/30/90 day), rolling volatility, lag prices, and price range ratios
- Trains a **Random Forest Regressor** inside a full `sklearn` Pipeline with `ColumnTransformer` (numeric imputation + scaling, categorical OHE for Ticker)
- Uses a **time-based 80/20 train-test split** (chronological order preserved)
- Computes **RMSE, MAE, and R²** for both train and test sets
- Generates **Actual vs Predicted** line charts per ticker
- Saves the trained pipeline as `stock_rf_model.joblib`
- Pushes raw data, features, predictions, and metrics to **MySQL** (4 tables)
- Exports everything to a multi-sheet **Excel workbook** with embedded charts
- Serves all functionality through a **Streamlit app** with CSV upload, training, and download buttons

## Dataset
| Property | Detail |
| :--- | :--- |
| **Stocks Covered** | Apple (AAPL), Microsoft (MSFT), Netflix (NFLX), Google (GOOG) |
| **Format** | CSV with multi-ticker rows |
| **Required Columns** | `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `AdjClose` (or `Adj Close`), `Volume` |
| **Target Variable** | `AdjClose` — adjusted closing price |
| **Min Rows Required** | 20+ rows after feature engineering (50+ per ticker recommended) |

### Column Handling (from source code)
| Column | Processing |
| :--- | :--- |
| `Date` | `pd.to_datetime()` |
| `Adj Close` | Auto-renamed to `AdjClose` |
| `Close` | Used as `AdjClose` fallback if `AdjClose` missing |
| All OHLCV cols | `pd.to_numeric(errors='coerce')` |
| Duplicates | `drop_duplicates(subset=['Ticker', 'Date'])` |

## Technologies Used
| Technology | Version | Purpose |
| :--- | :---: | :--- |
| **Python** | 3.9+ | Core programming language |
| **Pandas** | 2.x | Data loading, cleaning, feature engineering, Excel export |
| **NumPy** | latest | Feature calculations, RMSE computation |
| **Matplotlib** | latest | Actual vs Predicted charts, Excel embedded charts |
| **Scikit-Learn** | 1.x | Pipeline, ColumnTransformer, RandomForest, StandardScaler, SimpleImputer, OHE, metrics |
| **Joblib** | latest | Serialise and load trained pipeline |
| **SQLAlchemy** | latest | MySQL ORM — write 4 tables |
| **PyMySQL** | latest | MySQL driver |
| **Streamlit** | 1.x | Interactive upload → train → predict → export dashboard |
| **XlsxWriter** | latest | Multi-sheet Excel with embedded charts |
| **urllib.parse** | built-in | URL-encode MySQL password |
| **math / tempfile / warnings** | built-in | RMSE calculation, temp chart files, warning suppression |

### Python Libraries (from source code)
```python
import os, math, tempfile, warnings, urllib.parse, joblib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sqlalchemy import create_engine
import pymysql
import streamlit as st
```

## Feature Engineering
All features are computed **per Ticker group** using `groupby('Ticker')` transforms:
### Feature Set (12 features extracted from source code)
| Feature | Formula | Window | Purpose |
| :--- | :--- | :---: | :--- |
| `Daily_Return` | `AdjClose.pct_change()` | 1 day | Day-over-day % change |
| `MA_7` | `AdjClose.rolling(7).mean()` | 7 days | Short-term trend |
| `MA_30` | `AdjClose.rolling(30).mean()` | 30 days | Medium-term trend |
| `MA_90` | `AdjClose.rolling(90).mean()` | 90 days | Long-term trend |
| `Vol_30d` | `Daily_Return.rolling(30).std()` | 30 days | Rolling volatility |
| `Lag_1` | `AdjClose.shift(1)` | 1 day | Yesterday's price |
| `Lag_2` | `AdjClose.shift(2)` | 2 days | Price 2 days ago |
| `HL_range` | `(High - Low) / AdjClose` | — | Intraday price range ratio |
| `OC_change` | `(Close - Open) / AdjClose` | — | Open-to-close change ratio |
| `dayofweek` | `Date.dt.dayofweek` | — | 0=Monday to 6=Sunday |
| `month` | `Date.dt.month` | — | Seasonality signal |
| `Ticker` | Categorical | — | Stock identifier (OHE encoded) |

## ML Pipeline Architecture
```
Uploaded CSV
      │
      ▼
load_and_clean()
├── Strip column names
├── Parse Date → datetime
├── Rename Adj Close → AdjClose
├── Convert OHLCV → numeric
├── dropna(subset=['Ticker','Date','AdjClose'])
└── sort_values(['Ticker','Date']) + drop_duplicates

      │
      ▼
create_features()
├── Daily_Return (pct_change per ticker)
├── MA_7, MA_30, MA_90 (rolling mean per ticker)
├── Vol_30d (rolling std of returns per ticker)
├── Lag_1, Lag_2 (shifted AdjClose per ticker)
├── HL_range, OC_change (price ratios)
└── dayofweek, month (date parts)

      │
      ▼
prepare_ml_data()
├── Target = AdjClose
└── Features = [Lag_1, Lag_2, MA_7, MA_30, MA_90,
                Vol_30d, HL_range, OC_change,
                Volume, dayofweek, month, Ticker]

      │
      ▼
Time-based Train/Test Split (80% / 20%)
(chronological order preserved)

      │
      ▼
ColumnTransformer
├── num → SimpleImputer(median) → StandardScaler
└── cat (Ticker) → SimpleImputer(constant) → OneHotEncoder

      │
      ▼
RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)

      │
      ▼
Evaluate: RMSE, MAE, R² (train + test)

      │
      ▼
Save: stock_rf_model.joblib

      │
      ├──────────────────────────────┐
      ▼                              ▼
Export to Excel (6 sheets)    Push to MySQL (4 tables)
```

## Model Accuracy Results
**Model:** `RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)`
**Split:** Time-based 80/20 (chronological — no data leakage)
**Target:** `AdjClose` (raw price, not log-transformed)
| Split | RMSE | MAE | R² Score |
| :---: | :---: | :---: | :---: |
| **Train** | Low | Low | ~0.99 |
| **Test** | Higher | Higher | ~0.95–0.99 |

> Random Forest achieves high R² on stock price prediction because `Lag_1` (yesterday's price) is a very strong predictor — prices are highly autocorrelated. This is expected behaviour in financial time series.

### Metrics computed using (from source code)
```python
def metrics(y_true, y_pred):
    return {
        'rmse': math.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
```

## MySQL Tables
Database: `stock_market`
| Table | Contents |
| :--- | :--- |
| `raw_data` | Full cleaned dataset (all tickers, all dates) |
| `features` | Feature-engineered dataset (12 features per row) |
| `predictions` | Ticker, Date, AdjClose (actual), y_true, y_pred |
| `model_metrics` | Split (train/test), RMSE, MAE, R² |

## 🖥️ Streamlit Dashboard
### Main Buttons
| Button | Action |
| :--- | :--- |
| **Upload CSV** | File uploader → loads and validates CSV, shows row count and ticker count |
| **Train Model** | Runs full pipeline: features → train/test split → fit → metrics → predictions |
| **Export to Excel** | Generates `stock_analysis_ml_output.xlsx` with 6 sheets + embedded charts |
| **Save to MySQL** | Pushes all 4 tables to MySQL `stock_market` database |

### After Training — Displayed Results
| Section | Content |
| :--- | :--- |
| **Dataset preview** | First 5 rows of loaded CSV |
| **Model Performance table** | RMSE, MAE, R² for Train and Test splits |
| **Predictions line chart** | Actual vs Predicted `AdjClose` per ticker (first 2 tickers shown) |
| **Download Excel** | Button to download full results workbook |

### Excel Workbook Sheets (6 sheets)
| Sheet | Contents |
| :--- | :--- |
| `raw_data` | Cleaned raw CSV data |
| `features` | All 12 engineered features |
| `train_eval` | Train and test metrics (RMSE, MAE, R²) |
| `predictions` | Actual vs predicted per ticker per date |
| `correlation` | Ticker-to-ticker price correlation matrix |
| `charts` | Actual vs Predicted line charts embedded for up to 4 tickers |

## Installation and Setup
### Step 1 — Clone the Repository
```bash
git clone https://github.com/abhi-1009/Stock-Market-Analysis.git
cd Stock-Market-Analysis
```
### Step 2 — Install Required Libraries
```bash
pip install streamlit pandas numpy matplotlib scikit-learn joblib sqlalchemy pymysql xlsxwriter openpyxl
```
### Step 3 — Prepare Your CSV
Ensure your stock CSV contains these columns:
```
Date, Ticker, Open, High, Low, Close, AdjClose (or Adj Close), Volume
```
You can download historical data from [Yahoo Finance](https://finance.yahoo.com) or [Kaggle](https://www.kaggle.com).

### Step 4 — Configure MySQL (optional)
Update `DB_SETTINGS` in the script or enter credentials in the Streamlit app:
```python
DB_SETTINGS = {
    "user": "your_user",
    "password": "your_password",
    "host": "127.0.0.1",
    "port": 3306,
    "database": "stock_market"
}
```
### Step 5 — Launch the App
```bash
streamlit run stock_project_full_pipeline_streamlit.py
```

## Usage
1. **Upload your stock CSV** → app validates columns and shows preview
2. Click **Train Model** → pipeline runs, metrics table and prediction charts appear
3. Click **Export to Excel** → download full 6-sheet workbook with embedded charts
4. Click **Save to MySQL** → all 4 tables pushed to `stock_market` database
5. Trained model saved as `stock_rf_model.joblib` — reloadable for future predictions

## Key Insights
- **Lag_1** (yesterday's price) is the strongest single predictor — stock prices are highly autocorrelated
- **Moving averages (MA_7, MA_30, MA_90)** capture trend direction across short, medium, and long horizons
- **Rolling volatility (Vol_30d)** flags high-uncertainty periods that affect prediction confidence
- **HL_range and OC_change** encode intraday momentum — days with large spreads often precede trend reversals
- **Time-based split** (not random) is critical — random splitting in time series causes data leakage and inflated R² scores
- **Ticker as a feature** (OHE) allows the single model to learn stock-specific price level differences

## References
- [Scikit-Learn Pipeline Documentation](https://scikit-learn.org/stable/modules/compose.html)
- [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Yahoo Finance Data](https://finance.yahoo.com)
- [Joblib Documentation](https://joblib.readthedocs.io/)

## Author
**Abhijit Sinha**
- GitHub: [@abhi-1009](https://github.com/abhi-1009)
- LinkedIn: [abhijit-sinha-053b159a](https://linkedin.com/in/abhijit-sinha-053b159a)
- Email: sinhaabhijit12@yahoo.com
