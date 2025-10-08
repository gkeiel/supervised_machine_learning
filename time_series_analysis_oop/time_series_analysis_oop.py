import os
import itertools
from datetime import datetime
from time_series_analysis_functions_oop import Loader
from time_series_analysis_functions_oop import Indicator
from time_series_analysis_functions_oop import Backtest
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    # defines start and end time
    start = "2024-01-01"
    end   = datetime.now()

    # initialize cache dictionaries
    raw_data = {}
    pro_data = {}
    res_data = {}

    # import lists of parameters
    loader = Loader("tickers.txt", "indicators.txt")
    tickers    = ["B3SA3.SA"]
    indicators = loader.load_indicators()
    
    # download data and run backtest
    for ticker, (ind_t, ind_s, ind_l) in itertools.product(tickers, indicators):

        # download data (only once)
        if ticker not in raw_data:
            raw_data[ticker] = loader.download_data(ticker, start, end)
        df = raw_data[ticker]

        # calculate indicator
        indicator = Indicator(ind_t, ind_s, ind_l)
        df = indicator.calculate(df)

        # run backtest
        backtest = Backtest(df)
        df = backtest.run()

        if ticker not in res_data:
            pro_data[ticker] = {}
            res_data[ticker] = {}

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
        backtest.plot(label)


if __name__ == "__main__":
    main()