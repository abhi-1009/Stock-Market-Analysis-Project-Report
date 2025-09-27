## Stock Market Analysis Project

## Overview
This project analyzes stock market data for **Apple (AAPL), Microsoft (MSFT), Netflix (NFLX), and Google (GOOG)**.  
It covers **Exploratory Data Analysis (EDA)**, **SQL integration**, **Machine Learning model training**, and a **Streamlit app deployment**.

The goal is to identify stock trends, compute moving averages & volatility, and build predictive models for closing prices.

## Tools & Technologies
- **Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)**
- **SQL (MySQL for storing raw data, features, predictions, and metrics)**
- **Machine Learning (Random Forest Regressor)**
- **Streamlit (for interactive dashboard)**
- **Excel Export (via Pandas ExcelWriter)**

## Steps Performed

### 1. Data Preprocessing
- Loaded raw CSV dataset
- Cleaned column names & handled missing values
- Feature engineering (Moving Averages, Volatility, Lag features)
- Train/test split for ML model

### 2. Exploratory Data Analysis (EDA)
- Distribution of closing prices
- Total traded volume by company
- Scatter plot (Volume vs Closing Price)
- Correlation heatmap

### 3. SQL Integration
- Stored raw, feature-engineered, and predictions data into **MySQL**
- Example Query: *Average Closing Price by Ticker*

### 4. Machine Learning
- Trained a **Random Forest Regressor**
- Evaluated performance with **RMSE, MAE, R²**
- Visualized **Actual vs Predicted closing prices**

### 5. Streamlit App
- Upload CSV & preview data
- Train ML model interactively
- Visualize predictions (Actual vs Predicted)
- Export results to Excel or save to MySQL

## Sample Outputs

- **EDA Charts**: Price distribution, correlations, and volume analysis  
- **SQL Query Visualization**: Average closing prices per ticker  
- **ML Model Results**: Random Forest regression with R² score  
- **Streamlit App UI**: Upload → Train Model → Predictions → Export  

## Learnings & Challenges
- Importance of feature engineering (lags, volatility, moving averages) in stock prediction  
- Handling missing values & ensuring consistent ticker data  
- Balancing interpretability vs predictive performance in ML models  
- Deploying with Streamlit for interactive exploration  
---

