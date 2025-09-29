import os
import itertools
import time_series_analysis_functions as tsf
from datetime import datetime
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    # defines start and end time
    start = "2024-01-01"
    end   = datetime.now()

    # initialize cache dictionaries
    raw_data = {}
    pro_data = {}
    res_data = {}

    # import lists of parameters:
    tickers    = ["PETR4.SA"]
    indicators = tsf.load_indicators("indicators.txt")
    
    # download data and run backtest
    for ticker, (ind_t, ind_s, ind_l) in itertools.product(tickers, indicators):

        # download data (only once)
        if ticker not in raw_data:
            raw_data[ticker] = tsf.download_data(ticker, start, end)
        df = raw_data[ticker]

        # calculate indicator
        if ind_t == "SMA":
            df["Short"], df["Long"] = tsf.sma( df["Close"], ind_s, ind_l)
        elif ind_t == "EMA":
            df["Short"], df["Long"] = tsf.ema( df["Close"], ind_s, ind_l)
        elif ind_t == "WMA":
            df["Short"], df["Long"] = tsf.wma( df["Close"], ind_s, ind_l)

        # run backtest
        df = tsf.run_strategy(df)

        if ticker not in res_data:
            res_data[ticker] = {}
            pro_data[ticker] = {}

        # store processed data and result data
        label = f"{ticker}_{ind_t}_{ind_s}_{ind_l}"
        pro_data[ticker][label] = df.copy()
        res_data[ticker][label] = {
            "Indicator": ind_t,
            "MA_Short": ind_s,
            "MA_Long": ind_l,
            "Return_Market": df["Cumulative_Market"].iloc[-1],
            "Return_Strategy": df["Cumulative_Strategy"].iloc[-1]
        }
        tsf.plot_res(df, label)

    # exports dataframe for analysis
    tsf.export_dataframe(pro_data)

    # exports backtesting results
    tsf.export_results(res_data)


if __name__ == "__main__":
    main()