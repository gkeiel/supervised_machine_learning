import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_tickers(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers


def load_indicators(filepath):
    indicators = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = [p.strip() for p in line.split(",") if p.strip()]
                ind_t = parts[0]                        # indicator title
                ind_p = [int(x) for x in parts[1:]]     # indicator parameters
                indicators.append({"ind_t":ind_t, "ind_p":ind_p})
    return indicators


def download_data(ticker, start, end):
    # collect OHLCVDS data from Yahoo Finance
    try:
        df = yf.download(ticker, start, end, auto_adjust=True)
    except Exception as err:
        raise RuntimeError("Unexpected error in download_data.") from err
    df.columns = df.columns.droplevel(1)    
    df = df[["Close", "Volume"]]
    return df


def sma(series:pd.Series, window:int) -> pd.Series:
    # simple moving average (SMA)
    return series.rolling(window=window).mean()


def wma(series:pd.Series, window:int) -> pd.Series:
    # weighted moving average (WMA)
    w      = pd.Series(range(1, window+1), dtype=float)
    return series.rolling(window=window).apply(lambda x: (x*w).sum()/w.sum(), raw=True)


def ema(series:pd.Series, window:int) -> pd.Series:
    # exponential moving average (EMA)
    return series.ewm(span=window, adjust=False).mean()


def setup_indicator(df, indicator):
    """
    parameters:
    - df: dataframe with column 'Close'
    - indicator: dictionary with
        - ind_t: str with indicator name ("SMA", "WMA", "EMA" or "BB")
        - ind_p: list with indicator values (10, 20)
    """
    df     = df.copy()
    ind_t  = indicator.get("ind_t", "")
    params = indicator.get("ind_p", [])

    if ind_t in ["SMA", "WMA", "EMA"]:
        fn = globals().get(ind_t.lower())
        # 1 MA
        if len(params) == 1:
            short = params[0]
            df["Short"] = fn(df["Close"], short)
        # 2 MAs
        elif len(params) == 2:
            short, long = params
            df["Short"] = fn(df["Close"], short)
            df["Long"]  = fn(df["Close"], long)
        # 3 MAs
        elif len(params) == 3:
            short, medium, long = params
            df["Short"] = fn(df["Close"], short)
            df["Mid"]   = fn(df["Close"], medium)
            df["Long"]  = fn(df["Close"], long)
        else:
            raise ValueError(f"Unsupported indicator: {ind_t}.")    
        return df


def run_strategy(df, indicator):
    df = df.copy()
    ind_t  = indicator["ind_t"]
    params = indicator["ind_p"]
    
    # generate buy/sell signals
    df["Signal"] = 0
    if ind_t in ["SMA", "EMA", "WMA"]:
        if len(params) == 1:
            # 1 MA crossover
            df.loc[df["Short"] > df["Close"], "Signal"] = 1         # buy signal
            df.loc[df["Short"] < df["Close"], "Signal"] = -1        # sell signal
        elif len(params) == 2:
            # 2 MAs crossover
            df.loc[df["Short"] > df["Long"], "Signal"] = 1          # buy signal
            df.loc[df["Short"] < df["Long"], "Signal"] = -1         # sell signal
        elif len(params) == 3:
            # 3 MAs crossover
            df.loc[(df["Short"] > df["Med"]) & (df["Med"] > df["Long"]), "Signal"] = 1                              # buy signal
            df.loc[(df["Short"] < df["Med"]) & (df["Med"] < df["Long"]), "Signal"] = -1                             # sell signal
    df["Signal_Length"] = df["Signal"].groupby((df["Signal"] != df["Signal"].shift()).cumsum()).cumcount() +1  # consecutive samples of same signal (signal length)

    # simulate execution (backtest)
    df["Position"] = df["Signal"].shift(1)                      # simulate position (using previous sample)
    df["Return"] = df["Close"].pct_change()                     # asset percentage variation (in relation to previous sample)
    df["Strategy"] = df["Position"]*df["Return"]                # return of the strategy
    
    # compare buy & hold vs current strategy
    df["Cumulative_Market"] = (1 +df["Return"]).cumprod()       # cumulative return buy & hold strategy
    df["Cumulative_Strategy"] = (1 +df["Strategy"]).cumprod()   # cumulative return current strategy
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