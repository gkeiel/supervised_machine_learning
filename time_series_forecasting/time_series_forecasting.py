import os
import time_series_forecasting_functions as tsf
from datetime import datetime
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    # defines start and end time
    start = "2025-01-01"
    end   = datetime.now()

    # initialize cache dictionaries
    raw_data = {}
    pro_data = {}

    # lists of parameters
    ticker   = "B3SA3.SA"

    # model order
    n_a = 2

    # download time series data
    df = tsf.download_data(ticker, start, end)

    # clean time series
    df = tsf.clean_data(df)
    
    # prediction from decision tree
    df, idx = tsf.decision_tree(df)

    # prediction from ARIMA model
    df = tsf.arima_model(df, n_a, idx)

    # exports dataframe for analysis
    label = f"{ticker}_{n_a}"
    pro_data[ticker]        = {}
    pro_data[ticker][label] = df.copy()
    tsf.export_dataframe(pro_data)

    # metrics
    mse_ar, mse_tree = tsf.metrics(df)

    # plot
    tsf.plot_res(df, mse_ar, mse_tree, label, idx)


if __name__ == "__main__":
    main()