"""
stock_project_full_pipeline_streamlit.py
Streamlit-enabled version of your Stock Market ML pipeline
"""

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

warnings.filterwarnings("ignore")
plt.ioff()

# -----------------------
# CONFIG
# -----------------------
OUTPUT_EXCEL = "stock_analysis_ml_output.xlsx"
MODEL_PATH = "stock_rf_model.joblib"
MOVING_WINDOWS = [7, 30, 90]
ROLLING_VOL_WINDOW = 30

DB_SETTINGS = {
    "user": "root",
    "password": "Abhi@100982",
    "host": "127.0.0.1",
    "port": 3306,
    "database": "stock_market"
}

# -----------------------
# DB ENGINE
# -----------------------
def get_sqlalchemy_engine(user, password, host, port, database):
    password_enc = urllib.parse.quote_plus(password)
    conn_str = f"mysql+pymysql://{user}:{password_enc}@{host}:{port}/{database}"
    safe_str = conn_str.replace(password_enc, "****")
    print("Connecting with:", safe_str)
    return create_engine(conn_str, pool_recycle=3600)

# -----------------------
# BACKEND FUNCTIONS
# -----------------------

def load_and_clean(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]
    if 'Date' not in df.columns:
        raise ValueError("CSV must contain 'Date' column.")
    df['Date'] = pd.to_datetime(df['Date'])
    for col in ['Adj Close', 'AdjClose']:
        if col in df.columns:
            df.rename(columns={col: 'AdjClose'}, inplace=True)
            break
    if 'AdjClose' not in df.columns and 'Close' in df.columns:
        df['AdjClose'] = df['Close']
    numeric_cols = ['Open','High','Low','Close','AdjClose','Volume']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'Ticker' not in df.columns:
        raise ValueError("CSV must contain 'Ticker' column (symbol).")
    df = df.dropna(subset=['Ticker','Date','AdjClose'])
    df = df.sort_values(['Ticker','Date']).reset_index(drop=True)
    df = df.drop_duplicates(subset=['Ticker','Date'])
    return df

def create_features(df):
    df = df.copy().sort_values(['Ticker','Date'])
    df['Daily_Return'] = df.groupby('Ticker')['AdjClose'].pct_change()
    for w in MOVING_WINDOWS:
        df[f'MA_{w}'] = df.groupby('Ticker')['AdjClose'].transform(lambda s: s.rolling(window=w, min_periods=1).mean())
    df[f'Vol_{ROLLING_VOL_WINDOW}d'] = df.groupby('Ticker')['Daily_Return'].transform(lambda s: s.rolling(window=ROLLING_VOL_WINDOW, min_periods=1).std())
    df['Lag_1'] = df.groupby('Ticker')['AdjClose'].shift(1)
    df['Lag_2'] = df.groupby('Ticker')['AdjClose'].shift(2)
    df['HL_range'] = (df['High'] - df['Low']) / df['AdjClose']
    df['OC_change'] = (df['Close'] - df['Open']) / df['AdjClose']
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df = df.dropna(subset=['Lag_1'])
    return df

def prepare_ml_data(df):
    df = df.copy()
    df['Target'] = df['AdjClose']
    feature_cols = ['Lag_1','Lag_2','MA_7','MA_30','MA_90',
                    'Vol_30d','HL_range','OC_change','Volume','dayofweek','month','Ticker']
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].copy()
    y = df['Target'].copy()
    meta = df[['Ticker','Date','AdjClose']].copy()
    return X, y, meta, available_features

def build_pipeline():
    return RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)

def train_and_evaluate(X, y, meta, features):
    n = len(meta)
    if n < 20:  # enforce a safer minimum
        raise ValueError(f"Dataset too small for training. Need at least 20 rows after feature engineering, got {n}.")

    cat_cols = [c for c in features if c == 'Ticker' or X[c].dtype == object]
    num_cols = [c for c in features if c != 'Ticker' and c not in cat_cols]

    # split safely
    #split_idx = int(n * 0.8)
    #if split_idx <= 0 or split_idx >= n:
    #    raise ValueError(f"Invalid split with {n} rows. Cannot create both train and test sets.")

    #train_idx = meta.index[:split_idx]
    #test_idx = meta.index[split_idx:]

    #X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    #y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    #meta_test = meta.iloc[test_idx].copy()

    #if len(X_train) == 0 or len(X_test) == 0:
    #    raise ValueError(f"Train/test split failed: train={len(X_train)} rows, test={len(X_test)} rows.")
    
        # split safely (use positions instead of index labels)
    split_idx = int(n * 0.8)
    if split_idx <= 0 or split_idx >= n:
        raise ValueError(f"Invalid split with {n} rows. Cannot create both train and test sets.")

    train_idx = list(range(split_idx))
    test_idx = list(range(split_idx, n))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    meta_test = meta.iloc[test_idx].copy()


    # preprocess
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    rf = build_pipeline()
    model_pipeline = Pipeline(steps=[('preproc', preprocessor), ('rf', rf)])

    model_pipeline.fit(X_train, y_train)

    y_pred_train = model_pipeline.predict(X_train)
    y_pred_test = model_pipeline.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            'rmse': math.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    train_metrics = metrics(y_train, y_pred_train)
    test_metrics = metrics(y_test, y_pred_test)

    results_test = meta_test.copy().reset_index(drop=True)
    results_test['y_true'] = y_test.reset_index(drop=True)
    results_test['y_pred'] = y_pred_test

    joblib.dump(model_pipeline, MODEL_PATH)

    return model_pipeline, train_metrics, test_metrics, results_test

def compute_correlations(df):
    pivot = df.pivot(index='Date', columns='Ticker', values='AdjClose')
    returns = pivot.pct_change().dropna(how='all')
    corr = returns.corr()
    return pivot, returns, corr

def save_to_mysql(df_raw, df_features, metrics_dict, df_predictions, engine):
    with engine.begin() as conn:
        df_raw.to_sql('raw_data', con=conn, if_exists='replace', index=False)
        df_features.to_sql('features', con=conn, if_exists='replace', index=False)
        df_predictions.to_sql('predictions', con=conn, if_exists='replace', index=False)
        metrics_rows = []
        for k, v in metrics_dict.items():
            row = {'split': k}
            row.update(v)
            metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_sql('model_metrics', con=conn, if_exists='replace', index=False)

def export_to_excel(df_raw, df_features, train_metrics, test_metrics, df_predictions, corr, out_path):
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        df_raw.to_excel(writer, sheet_name='raw_data', index=False)
        df_features.to_excel(writer, sheet_name='features', index=False)
        pd.DataFrame([{'split':'train', **train_metrics},{'split':'test', **test_metrics}]).to_excel(writer, sheet_name='train_eval', index=False)
        df_predictions.to_excel(writer, sheet_name='predictions', index=False)
        corr.to_excel(writer, sheet_name='correlation')

        workbook = writer.book
        charts_ws = workbook.add_worksheet("charts")
        writer.sheets['charts'] = charts_ws

        tickers = df_predictions['Ticker'].unique()[:4]
        start_row = 0
        for t in tickers:
            sub = df_predictions[df_predictions['Ticker']==t].sort_values('Date')
            if sub.empty: continue
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(sub['Date'], sub['y_true'], label='Actual')
            ax.plot(sub['Date'], sub['y_pred'], label='Predicted', linestyle='--')
            ax.set_title(f'{t} Actual vs Predicted')
            ax.legend()
            tmpfile = os.path.join(tempfile.gettempdir(), f"chart_{t}.png")
            fig.savefig(tmpfile, dpi=150)
            plt.close(fig)
            charts_ws.insert_image(start_row, 0, tmpfile)
            start_row += 20

# -----------------------
# STREAMLIT APP
# -----------------------
st.set_page_config(page_title="Stock Market ML Project", layout="wide")
st.title("Stock Market ML Project")

uploaded_file = st.file_uploader("Upload Stock CSV", type=["csv"])

if uploaded_file:
    df_raw = load_and_clean(uploaded_file)
    st.session_state.df_raw = df_raw
    n_rows = len(df_raw)
    n_tickers = df_raw['Ticker'].nunique()

    st.success(f"Loaded {n_rows} rows and {n_tickers} tickers.")
    st.dataframe(df_raw.head())

    if n_rows < 30:
        st.warning(f"Dataset has only {n_rows} rows. For reliable training, please upload at least 30 rows (ideally 50+) per ticker.")

if st.button("Train Model"):
    if "df_raw" not in st.session_state:
        st.error("Please upload a CSV first!")
    else:
        try:
            df_features = create_features(st.session_state.df_raw)

            if len(df_features) < 10:
                st.warning(f"⚠️ Dataset too small after feature engineering. Got only {len(df_features)} rows, need at least 10.")
            else:
                X, y, meta, features = prepare_ml_data(df_features)
                model_pipeline, train_metrics, test_metrics, df_preds = train_and_evaluate(X, y, meta, features)

                # Save in session_state
                st.session_state.df_features = df_features
                st.session_state.df_preds = df_preds
                st.session_state.train_metrics = train_metrics
                st.session_state.test_metrics = test_metrics

                st.success("✅ Model trained successfully!")

                # Metrics table
                metrics_df = pd.DataFrame([
                    {"Split": "Train", **train_metrics},
                    {"Split": "Test", **test_metrics}
                ])
                st.subheader("Model Performance")
                st.dataframe(metrics_df)

                # Predictions plot
                st.subheader("Predictions (Actual vs Predicted)")
                for t in df_preds["Ticker"].unique()[:2]:
                    sub = df_preds[df_preds["Ticker"] == t].sort_values("Date")
                    st.line_chart(sub.set_index("Date")[["y_true","y_pred"]])

        except Exception as e:
            st.error(f"Training failed: {e}")

if st.button("Export to Excel"):
    if "df_preds" not in st.session_state:
        st.error("Please train the model first.")
    else:
        try:
            _,_,corr = compute_correlations(st.session_state.df_raw)
            export_to_excel(
                st.session_state.df_raw,
                st.session_state.df_features,
                st.session_state.train_metrics,
                st.session_state.test_metrics,
                st.session_state.df_preds,
                corr,
                OUTPUT_EXCEL
            )
            with open(OUTPUT_EXCEL, "rb") as f:
                st.download_button("Download Excel", f, OUTPUT_EXCEL)
            st.success("Excel exported successfully!")
        except Exception as e:
            st.error(f"Excel export failed: {e}")

if st.button("Save to MySQL"):
    if "df_preds" not in st.session_state:
        st.error("Please train the model first.")
    else:
        try:
            engine = get_sqlalchemy_engine(**DB_SETTINGS)
            metrics_dict = {
                "train": st.session_state.train_metrics,
                "test": st.session_state.test_metrics
            }
            save_to_mysql(
                st.session_state.df_raw,
                st.session_state.df_features,
                metrics_dict,
                st.session_state.df_preds,
                engine
            )
            st.success("Data saved to MySQL successfully!")
        except Exception as e:
            st.error(f"MySQL save failed: {e}")
