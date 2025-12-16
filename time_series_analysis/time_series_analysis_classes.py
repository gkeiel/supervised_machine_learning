import yfinance as yf
import pandas as pd
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Loader:
    def __init__(self, file_tickers=None, file_indicators=None):
        self.file_tickers = file_tickers
        self.file_indicators = file_indicators

    # load tickers from .txt
    def load_tickers(self):
        with open(self.file_tickers, "r", encoding="utf-8") as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    
    # load indicators from .txt
    def load_indicators(self):
        indicators = []
        with open(self.file_indicators, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    ind_t = parts[0]                        # indicator title
                    ind_p = [int(x) for x in parts[1:]]     # indicator parameters
                    indicators.append({"ind_t":ind_t, "ind_p":ind_p})
            return indicators
    
    # collect OHLCVDS data from Yahoo Finance
    def download_data(self, ticker, start, end):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        except Exception as err:
            raise RuntimeError("Unexpected error in download_data.") from err
        df.columns = df.columns.droplevel(1)    
        df = df[["Close", "Volume"]]
        return df
    

class Indicator:
    def __init__(self, indicator):
        self.indicator = indicator
        
    @staticmethod
    def sma(series:pd.Series, window:int) -> pd.Series:
        # simple moving average (SMA)
        return series.rolling(window=window).mean()

    @staticmethod
    def wma(series:pd.Series, window:int) -> pd.Series:
        # weighted moving average (WMA)
        w      = pd.Series(range(1, window+1), dtype=float)
        return series.rolling(window=window).apply(lambda x: (x*w).sum()/w.sum(), raw=True)

    @staticmethod
    def ema(series:pd.Series, window:int) -> pd.Series:
        # exponential moving average (EMA)
        return series.ewm(span=window, adjust=False).mean()

    # calculate indicator
    def setup_indicator(self, df):
        df     = df.copy()
        ind_t  = self.indicator.get("ind_t", "")
        params = self.indicator.get("ind_p", [])

        if ind_t in ["SMA", "WMA", "EMA"]:
            fn = getattr(self, ind_t.lower())
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


class Backtester:
    def __init__(self, df):
        self.df = df.copy()

    def run_strategy(self, indicator):
        df     = self.df
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
    
    # save results
    def plot(self, label):
        ticker, ind_t, *params = label.split("_")
        
        plt.figure(figsize=(12,6))
        plt.plot(self.df.index, self.df["Close"], label=f"{ticker}")
        if ind_t in ["SMA", "EMA", "WMA"]:
            if "Short" in self.df and len(params) >= 1:
                plt.plot(self.df.index, self.df["Short"], label=f"{ind_t}{params[0]}")
            if "Long" in self.df and len(params) == 2:
                plt.plot(self.df.index, self.df["Long"], label=f"{ind_t}{params[1]}")
            if "Long" in self.df and len(params) >= 3:
                plt.plot(self.df.index, self.df["Long"], label=f"{ind_t}{params[1]}")
            if "Med" in self.df and len(params) >= 3:
                plt.plot(self.df.index, self.df["Med"], label=f"{ind_t}{params[2]}")
        plt.title(f"{ticker} - Price")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"data/results/{label}.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12,6))
        plt.plot(self.df.index, self.df["Cumulative_Market"], label="Buy & Hold")
        plt.plot(self.df.index, self.df["Cumulative_Strategy"], label="Strategy")
        plt.title(f"{ticker} - Backtest {ind_t}{'/'.join(params)}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"data/results/{label}_backtest.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        
class Exporter:
    def export_dataframe(pro_data):
        # export dataframe for further analysis
        for ticker, ticker_debug in pro_data.items():
            with pd.ExcelWriter(f"data/debug/{ticker}.xlsx", engine="openpyxl") as writer:
                for sheet_name, df in ticker_debug.items():
                    # write to .xlsx
                    df.to_excel(writer, sheet_name=sheet_name[:20])

    def export_results(res_data):
        # export backtesting results (a spreadsheet for each ticker)
        with pd.ExcelWriter("data/results/results_backtest.xlsx", engine="openpyxl") as writer:
            for ticker, ticker_results in res_data.items():
                # orient combinations to rows
                ticker_results_df = pd.DataFrame.from_dict(ticker_results, orient="index")

                # write to .xlsx
                ticker_results_df.to_excel(writer, sheet_name=ticker[:10], index=False)