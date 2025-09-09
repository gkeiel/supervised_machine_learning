# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def time_series(n):
    # generate time series data with an exogenous input
    np.random.seed(0)
    t = np.arange(n)
    u = np.sin(0.2*t) + np.random.normal(scale=0.2, size=n)
    y = np.zeros(n)
    for k in range(2, n):
        y[k] = 0.6*y[k-1] -0.3*y[k-2] +0.5*u[k-1] +np.random.normal(scale=0.1)

    data = pd.DataFrame({'y':y, 'u':u, 't':t})
    return data


def arima_model(data, n, n_a):
    # ARIMA model
    ar_model  = ARIMA(data['y'], order=(n_a,0,0)).fit()
    y_pred_ar = ar_model.predict(start=n_a, end=n-1)
    return y_pred_ar


def decision_tree(data):
    u = data['u']
    y = data['y']

    # build features for ML model: past y and u values
    x_ml = np.column_stack((y[:-2], y[1:-1], u[1:-1]))  # lag-2 y and lag-1 u
    y_ml = y[2:]

    # train/test split
    split_idx = int(0.8*len(x_ml))
    x_train, x_test = x_ml[:split_idx], x_ml[split_idx:]
    y_train, y_test = y_ml[:split_idx], y_ml[split_idx:]

    # train decision tree
    tree = DecisionTreeRegressor(max_depth=5)
    tree.fit(x_train, y_train)
    y_pred_tree = tree.predict(x_test)
    return y_pred_tree, y_test, split_idx


def metrics(y_pred_ar, y_pred_tree, y, y_test):
    mse_ar   = mean_squared_error(y[2:], y_pred_ar)
    mse_tree = mean_squared_error(y_test, y_pred_tree)
    return mse_ar, mse_tree


def plot_res(y, t, y_pred_ar, y_pred_tree, mse_ar, mse_tree, split_idx):
    # plot predictions
    plt.figure(figsize=(12, 5))
    plt.plot(t[2:], y[2:], label="True y", color='black', linewidth=1.5)
    plt.plot(t[2:], y_pred_ar, label=f"AR model (MSE={mse_ar:.3f})", linestyle='--')
    plt.plot(t[split_idx+2:], y_pred_tree, label=f"Decision Tree (MSE={mse_tree:.3f})", linestyle=':')
    plt.legend()
    plt.title("AR vs Decision Tree Prediction")
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.tight_layout()
    plt.grid(True)
    plt.show()