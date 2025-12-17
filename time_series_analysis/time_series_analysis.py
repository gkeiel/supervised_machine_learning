import os, itertools
from datetime import datetime
from time_series_analysis_classes import Loader, Indicator, Backtester, Exporter
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
    tickers    = loader.load_tickers()
    indicators = loader.load_indicators()
    
    # download data and run backtest
    for ticker, indicator in itertools.product(tickers, indicators):

        # download data (only once)
        if ticker not in raw_data:
            raw_data[ticker] = loader.download_data(ticker, start, end)
        df = raw_data[ticker]

        # setup indicator
        df = Indicator(indicator).setup_indicator(df)

        # run backtest
        backtest = Backtester(df)
        df = backtest.run_strategy(indicator)

        if ticker not in res_data:
            pro_data[ticker] = {}
            res_data[ticker] = {}

        # store processed data and result data
        ind_t  = indicator["ind_t"]  # indicator title
        ind_p  = indicator["ind_p"]  # indicator parameters
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

    # exports backtesting results
    Exporter.export_results(res_data)


if __name__ == "__main__":
    main()