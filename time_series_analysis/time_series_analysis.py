
import time_series_analysis_functions as tsf
from datetime import datetime


def main():
    # select data, period
    ticker = "PETR4.SA"
    start  = "2023-01-01"
    end    = datetime.now()
    
    # collect data from Yahoo Finance
    df = tsf.download_data(ticker, start, end)

    # select short and long MA
    ma_s = 14
    ma_l = 28

    # calculate indicators
    lab_ma_s = "SMA"+str(ma_s)
    lab_ma_l = "SMA"+str(ma_l)
    df[lab_ma_s] = df["Close"].rolling(window=ma_s).mean()  # short MA
    df[lab_ma_l] = df["Close"].rolling(window=ma_l).mean()  # long MA

    # generate signals
    df["Signal"] = 0
    df.loc[df[lab_ma_s] > df[lab_ma_l], "Signal"] = 1   # buy
    df.loc[df[lab_ma_s] < df[lab_ma_l], "Signal"] = -1  # sell

    # simulate execution (backtest)
    df["Position"] = df["Signal"].shift(1)
    df["Return"] = df["Close"].pct_change()
    df["Strategy"] = df["Position"]*df["Return"]

    # compare buy & hold vs strategy
    df["Cumulative_Market"] = (1 + df["Return"]).cumprod()
    df["Cumulative_Strategy"] = (1 + df["Strategy"]).cumprod()

    # plot result
    tsf.plot_res(df)


if __name__ == "__main__":
    main()