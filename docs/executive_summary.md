# Executive Summary — Automated Daily Trading System

**Master of Business Analytics and Data Science, IE School of Science and Technology**

## Introduction

This project implements an automated daily trading system that predicts stock market movements using machine learning. The system covers five major US companies (Apple, Microsoft, Google, Amazon, and Tesla) and is accessible through a web application deployed on Streamlit Cloud.

## Data Sources

All financial data is sourced from SimFin (simfin.com), a platform providing free access to historical and real-time stock market data. Two main datasets were used:

- **Share Prices (Bulk Download):** Five years of daily prices including open, high, low, close, and volume for all US-listed companies. Used to train the machine learning model.
- **SimFin Data API:** Used in the web application to fetch fresh price data in real time for generating live predictions.

## ETL Process

The ETL (Extract, Transform, Load) pipeline processes raw share price data into a format suitable for machine learning. Key transformations include cleaning missing data, filling absent volume values, computing technical indicators (moving averages, RSI, volatility), and creating the target variable (whether the next day's price goes up or down). The pipeline is reusable across any ticker symbol by accepting the ticker as a parameter.

## Machine Learning Model

We chose a Logistic Regression model for its simplicity and interpretability. The model takes eight technical indicators as input and outputs a binary prediction: UP (price increases) or DOWN (price decreases). The data is split chronologically (80% training, 20% testing) to avoid data leakage. Features are standardized using a StandardScaler. While the model's accuracy is modest (as expected for stock prediction), the focus of this project is on the engineering and methodology rather than prediction quality.

| Feature | Description |
|---|---|
| Returns | Daily percentage change in closing price |
| SMA_5 | Simple Moving Average over 5 days |
| SMA_20 | Simple Moving Average over 20 days |
| EMA_12 | Exponential Moving Average over 12 days |
| RSI_14 | Relative Strength Index (14-day momentum) |
| Volatility_20 | Rolling standard deviation of returns (20 days) |
| Volume_Change | Daily percentage change in trading volume |
| High_Low_Range | Difference between daily high and low prices |

## Web Application

The Streamlit web application has three pages:

1. **Home Page:** Provides an overview of the system, explains how it works, and lists the supported companies.
2. **Go Live Page:** Allows users to select a stock ticker and time horizon, view real-time price data fetched from SimFin, see candlestick charts with moving average overlays, and receive the model's prediction for the next trading day.
3. **Backtesting Page:** Runs the ML strategy against a buy-and-hold baseline on historical data, displaying portfolio value over time alongside performance metrics (total return, Sharpe ratio, max drawdown).

The application uses a custom Python API wrapper class (PySimFin) to interact with SimFin's REST API, handling authentication and rate limiting automatically. The wrapper follows object-oriented design principles with proper error handling.

## Challenges

- **Rate Limiting:** SimFin's free tier requires careful caching and request management in the web app.
- **Data Quality:** The free dataset had missing volume data for most tickers, requiring the ETL pipeline to fill missing values rather than discard rows.
- **Column Naming:** The SimFin API uses different column names than the bulk download CSV files, requiring a renaming step in the web application to maintain compatibility with the trained model.

## Conclusions

This project demonstrates an end-to-end data engineering workflow: from raw data extraction through machine learning to a deployed web application. The system is modular, well-documented, and easily extensible to include more companies or more sophisticated models. The complete source code is available on GitHub and the live application is deployed on Streamlit Cloud.
