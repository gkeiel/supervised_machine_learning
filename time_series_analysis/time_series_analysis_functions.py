import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use("Agg")


def load_tickers(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers


def load_indicators(filepath):
    indicators = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t, s, l = line.strip().split(",")
                indicators.append((t, int(s), int(l)))
    return indicators


def download_data(ticker, start, end):
    # collect OHLCVDS data from Yahoo Finance
    df = yf.download(ticker, start, end, auto_adjust=True)
    df.columns = df.columns.droplevel(1)    
    df = df[["Close", "Volume"]]
    return df


def sma(series:pd.Series, short:int, long:int) -> tuple[pd.Series, pd.Series]:
    # simple moving average (SMA)
    return (series.rolling(window=short).mean(),
            series.rolling(window=long).mean())


def ema(series:pd.Series, short:int, long:int) -> tuple[pd.Series, pd.Series]:
    # exponential moving average (EMA)
    return (series.ewm(span=short, adjust=False).mean(),
            series.ewm(span=long, adjust=False).mean())


# def wma(series: pd.Series, span:int) -> pd.Series:
    # exponential moving average (EMA)
    # return series.ewm(span=span, adjust=False).mean()


def run_strategy(df):
    df = df.copy()
    
    # generate buy/sell signals
    df["Signal"] = 0
    df.loc[df["Short"] > df["Long"], "Signal"] = 1              # buy signal  ->  1
    df.loc[df["Short"] < df["Long"], "Signal"] = -1             # sell signal -> -1
    df["Signal_Length"] = df["Signal"].groupby((df["Signal"] != df["Signal"].shift()).cumsum()).cumcount() +1  # consecutive samples of same signal (signal length)
    df.loc[df["Signal"] == 0, "Signal_Strength"] = 0                                                           # strength is zero while there is no signal

    # simulate execution (backtest)
    df["Position"] = df["Signal"].shift(1)                      # simulate position (using previous sample)
    df["Trade"] = df["Position"].diff().abs()                   # simulate trade
    df["Return"] = df["Close"].pct_change()                     # asset percentage variation (in relation to previous sample)
    df["Strategy"] = df["Position"]*df["Return"]                # return of the strategy
    
    # compare buy & hold vs current strategy
    df["Cumulative_Market"] = (1 +df["Return"]).cumprod()       # cumulative return buy & hold strategy
    df["Cumulative_Strategy"] = (1 +df["Strategy"]).cumprod()   # cumulative return current strategy
    df["Cumulative_Trades"] = df["Trade"].cumsum()              # cumulative number of trades
    return df


def plot_res(df, label):
    ticker, ind_t, ind_s, ind_l = label.split("_")

    # save results
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["Close"], label=f"{ticker}")
    plt.plot(df.index, df[f"Short"], label=f"{ind_t}{ind_s}")
    plt.plot(df.index, df[f"Long"], label=f"{ind_t}{ind_l}")
    plt.title(f"{ticker} - Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{label}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["Cumulative_Market"], label="Buy & Hold")
    plt.plot(df.index, df["Cumulative_Strategy"], label="Strategy")
    plt.title(f"{ticker} - Backtest {ind_t}{ind_s}/{ind_l}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/backtest_{label}.png", dpi=300, bbox_inches="tight")
    plt.close()


def export_dataframe(pro_data):
    # export dataframe for further analysis
    for ticker, ticker_debug in pro_data.items():
        with pd.ExcelWriter(f"data/{ticker}.xlsx", engine="openpyxl") as writer:
            for sheet_name, df in ticker_debug.items():
                # write to .xlsx
                df.to_excel(writer, sheet_name=sheet_name[:20])


def export_results(res_data):
    # export backtesting results (a spreadsheet for each ticker)
    with pd.ExcelWriter("results/results_backtest.xlsx", engine="openpyxl") as writer:
        for ticker, ticker_results in res_data.items():
            # orient combinations to rows
            ticker_results_df = pd.DataFrame.from_dict(ticker_results, orient="index")

            # write to .xlsx
            ticker_results_df.to_excel(writer, sheet_name=ticker[:10], index=False)