# Time Series Forecasting using ARIMA

## Overview
This project focuses on **time series analysis and forecasting** using the **ARIMA (AutoRegressive Integrated Moving Average) model**. The goal is to analyze a time-dependent feature, transform it into a stationary series, and predict future values using statistical modeling.

## Dataset
The dataset consists of:
- **Train Data (`train_csv.csv`)**: Historical time series data with indexed timestamps.
- **Test Data (`test_csv.csv`)**: Future time points where predictions are needed.

## Workflow

### 1. **Data Preprocessing**
   - Load train and test datasets
   - Handle missing values
   - Check basic statistics of the feature

### 2. **Time Series Analysis**
   - **Visualization**: Plot raw data and differenced series to check trends
   - **Stationarity Check**: Apply **Augmented Dickey-Fuller (ADF) test**
   - **Autocorrelation and Partial Autocorrelation Analysis**: Use **ACF and PACF plots** to determine AR and MA terms

### 3. **Feature Engineering**
   - Apply differencing to remove trends
   - Check for stationarity post-transformation

### 4. **Model Selection & Training**
   - Fit an **ARIMA(p,d,q)** model with optimal parameters
   - Use **AIC (Akaike Information Criterion)** for model selection
   - Train the model on the historical data

### 5. **Evaluation**
   - Predict values for the test set
   - Compute **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** for accuracy assessment

### 6. **Final Predictions**
   - Generate predictions for future time steps
   - Merge predictions with the test dataset

## Dependencies
- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- Statsmodels
