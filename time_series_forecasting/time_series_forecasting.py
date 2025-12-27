import os, itertools
from datetime import datetime
from time_series_forecasting_classes import Loader, Forecaster, Backtester, Exporter
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    # define start and end time
    start = "2025-01-01"
    end   = datetime.now()

    # initialize cache dictionaries
    raw_data = {}
    pro_data = {}
    res_data = {}

    # import lists of parameters
    loader  = Loader("tickers.txt", "method.txt")
    tickers = loader.load_tickers()
    methods = loader.load_methods()

    # download data and run forecast
    for ticker, method in itertools.product(tickers, methods):
        
        # download data (only once)
        if ticker not in raw_data:
            raw_data[ticker] = loader.download_data(ticker, start, end)
        df = raw_data[ticker]
        
        # predictions
        df = Forecaster(method, df).predictions()
        
        # run backtest
        backtest = Backtester(df)
        df = backtest.run_strategy(method)
        
        
        if ticker not in res_data:
            pro_data[ticker] = {}
            res_data[ticker] = {}

        # store processed data and result data
        ind_t  = method["ind_t"]  # method name
        ind_p  = method["ind_p"]  # method parameters
        params = "_".join(str(p) for p in ind_p)
        label  = f"{ticker}_{ind_t}_{params}"
        
        pro_data[ticker][label] = df.copy()
        res_data[ticker][label] = {
            "Indicator": ind_t,
            "Parameters": ind_p,
            "Return_Market": df["Cumulative_Market"].iloc[-1],
            "Return_Strategy": df["Cumulative_Strategy"].iloc[-1]
        }
        backtest.plot(label)
    
    # exports dataframe for analysis
    Exporter.export_dataframe(pro_data)

    # metrics
    #mse_ar, mse_tree = tsf.metrics(df)

    # plot
    #tsf.plot_res(df, mse_ar, mse_tree, label, idx)


if __name__ == "__main__":
    main()