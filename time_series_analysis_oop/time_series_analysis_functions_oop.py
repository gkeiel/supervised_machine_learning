import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use("Agg")


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
                    t, s, l = line.strip().split(",")
                    indicators.append((t, int(s), int(l)))
        return indicators
    
    # collect OHLCVDS data from Yahoo Finance
    def download_data(self, ticker, start, end):
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        df.columns = df.columns.droplevel(1)    
        df = df[["Close", "Volume"]]
        return df
    

class Indicator:
    def __init__(self, name, short, long):
        self.name = name
        self.short = short
        self.long = long

    # calculate indicator
    def calculate(self, df):
        df = df.copy()
        df["Short"] = df["Close"].rolling(window=self.short).mean()
        df["Long"] = df["Close"].rolling(window=self.long).mean()
        return df


class Backtest:
    def __init__(self, df):
        self.df = df.copy()

    def run(self):
        df = self.df

        # generate buy/sell signals
        df["Signal"] = 0
        df.loc[df["Short"] > df["Long"], "Signal"] = 1              # buy signal  ->  1
        df.loc[df["Short"] < df["Long"], "Signal"] = -1             # sell signal -> -1
        df["Signal_Length"] = df["Signal"].groupby((df["Signal"] != df["Signal"].shift()).cumsum()).cumcount() +1  # consecutive samples of same signal (signal length)
        df.loc[df["Signal"] == 0, "Signal_Strength"] = 0                                                           # strength is zero while there is no signal

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
        ticker, ind_t, ind_s, ind_l = label.split("_")

        plt.figure(figsize=(12,6))
        plt.plot(self.df.index, self.df["Close"], label=f"{ticker}")
        plt.plot(self.df.index, self.df[f"Short"], label=f"{ind_t}{ind_s}")
        plt.plot(self.df.index, self.df[f"Long"], label=f"{ind_t}{ind_l}")
        plt.title(f"{ticker} - Price")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results/{label}.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12,6))
        plt.plot(self.df.index, self.df["Cumulative_Market"], label="Buy & Hold")
        plt.plot(self.df.index, self.df["Cumulative_Strategy"], label="Strategy")
        plt.title(f"{ticker} - Backtest {ind_t}{ind_s}/{ind_l}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results/backtest_{label}.png", dpi=300, bbox_inches="tight")
        plt.close()