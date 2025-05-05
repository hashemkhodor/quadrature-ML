import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

def preprocess_data(seq_length=10, test_size=2000, dt=0.01, input_file="forecaster/data/lorenz_data.csv"):
    df = pd.read_csv(input_file)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    dt_column = np.full((len(data_scaled), 1), dt)
    data_with_dt = np.hstack((data_scaled, dt_column))

    X, y = [], []
    for i in range(len(data_with_dt) - seq_length - 1):
        X.append(data_with_dt[i:i + seq_length])
        delta = data_scaled[i + seq_length] - data_scaled[i + seq_length - 1]
        y.append(delta)

    X = np.array(X)
    y = np.array(y)

    if test_size < len(X):
        X_train, y_train = X[:-test_size], y[:-test_size]
        X_test, y_test = X[-test_size:], y[-test_size:]
    else:
        X_train = X_test = X
        y_train = y_test = yc

    np.savez("forecaster/data/processed_data_dt.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    joblib.dump(scaler, "forecaster/data/scaler.pkl")
    return scaler
