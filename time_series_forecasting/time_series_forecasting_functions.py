# import modules
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def download_data(ticker, start, end):
    # collect OHLCVDS data from Yahoo Finance
    df = yf.download(ticker, start, end, auto_adjust=True)
    df.columns = df.columns.droplevel(1)    
    df = df[["Close", "Volume"]]
    return df


def clean_data(df):
    # clean time series
    df = df.rename(columns={"Close":"y", "Volume":"u"}).asfreq("B")
    df["y"] = df["y"].ffill()
    df["u"] = df["u"].fillna(0)
    return df  


def arima_model(df, n_a, idx):
    # ARIMA model
    ar_model  = ARIMA(df["y"], order=(n_a,0,0)).fit()

    # start and end
    start = idx +n_a
    end   = len(df) -1

    # predictions
    y_pred_ar = ar_model.predict(start=start, end=end)

    # add to dataframe
    df["y_ARIMA"] = np.nan
    df.iloc[start:end+1, df.columns.get_loc("y_ARIMA")] = y_pred_ar.values
    return df


def decision_tree(df):
    u = df["u"]
    y = df["y"]

    # build features for ML model: past y and u values
    x_ml = np.column_stack((y[:-2], y[1:-1], u[1:-1]))  # lag-2 y and lag-1 u
    y_ml = y[2:]

    # train/test split
    idx = int(0.5*len(x_ml))
    x_train, x_test = x_ml[:idx], x_ml[idx:]
    y_train, y_test = y_ml[:idx], y_ml[idx:]

    # train decision tree
    tree = DecisionTreeRegressor(max_depth=5)
    tree.fit(x_train, y_train)
    y_pred_tree = tree.predict(x_test)

    # add to dataframe
    indices    = df.index[2+idx: 2+idx +len(y_pred_tree)]
    df["y_DT"] = np.nan
    df.loc[indices, "y_DT"] = y_pred_tree    
    return df, idx


def metrics(df):
    valid_ar = df["y_ARIMA"].notna()
    valid_tree = df["y_DT"].notna()

    mse_ar = mean_squared_error(df.loc[valid_ar, "y"], df.loc[valid_ar, "y_ARIMA"])
    mse_tree = mean_squared_error(df.loc[valid_tree, "y"], df.loc[valid_tree, "y_DT"])
    return mse_ar, mse_tree


def plot_res(df, mse_ar, mse_tree, label, idx):
    # plot predictions
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["y"], label="True y", color='black', linewidth=1.5)
    plt.plot(df.index, df["y_ARIMA"], label=f"AR model (MSE={mse_ar:.3f})", linestyle='--')
    plt.plot(df.index, df["y_DT"], label=f"Decision Tree (MSE={mse_tree:.3f})", linestyle=':')
    plt.title("AR vs Decision Tree Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{label}.png", dpi=300, bbox_inches="tight")
    plt.close()


def export_dataframe(pro_data):
    # export dataframe for further analysis
    for ticker, ticker_debug in pro_data.items():
        with pd.ExcelWriter(f"data/{ticker}.xlsx", engine="openpyxl") as writer:
            for sheet_name, df in ticker_debug.items():
                # write to .xlsx
                df.to_excel(writer, sheet_name=sheet_name[:20])