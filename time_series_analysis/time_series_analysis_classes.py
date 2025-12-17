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
                    ind_t = parts[0]                                                # indicator title
                    ind_p = [float(x) if "." in x else int(x) for x in parts[1:]]   # indicator parameters
                    indicators.append({"ind_t":ind_t, "ind_p":ind_p})
            return indicators
    
    # collect OHLCVDS data from Yahoo Finance
    def download_data(self, ticker, start, end):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        except Exception as err:
            raise RuntimeError("Unexpected error in download_data.") from err
        df.columns = df.columns.droplevel(1)    
        df = df[["High", "Low", "Close", "Volume"]]
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
    
    @staticmethod
    def bollinger_bands(series:pd.Series, window:int, std_dev:float=2.0):
        # bollinger bands (BB)
        middle = series.rolling(window=window).mean()
        std    = series.rolling(window=window).std()
        upper  = middle +(std_dev*std)
        lower  = middle -(std_dev*std)
        return middle, upper, lower
           
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        # moving average convergence divergence (MACD)
        macd_line   = series.ewm(span=fast, adjust=False).mean() -series.ewm(span=slow, adjust=False).mean()
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram   = macd_line -signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def parabolic_sar(high:pd.Series, low:pd.Series, step:float=0.02, max_step:float=0.2) -> pd.Series:
        # parabolic SAR
        sar = low.copy()
        trend, af = 1, step
        ep = high.iloc[0]
        
        for i in range(1, len(high)):
            sar.iloc[i] = sar.iloc[i-1] + af*(ep -sar.iloc[i-1])

            if trend == 1:
                if low.iloc[i] < sar.iloc[i]:
                    trend, af, ep = -1, step, low.iloc[i]
                    sar.iloc[i] = ep
                else:
                    ep = max(ep, high.iloc[i])
                    af = min(af +step, max_step)
            else:
                if high.iloc[i] > sar.iloc[i]:
                    trend, af, ep = 1, step, high.iloc[i]
                    sar.iloc[i] = ep
                else:
                    ep = min(ep, low.iloc[i])
                    af = min(af + step, max_step)
        return sar
    
    @staticmethod
    def supertrend(high:pd.Series, low:pd.Series, close:pd.Series, atr_window:int=10, multiplier:float=3.0) -> pd.Series:
        # supertrend
        tr    = pd.concat([high -low, (high -close.shift()).abs(), (low -close.shift()).abs()], axis=1).max(axis=1)
        atr   = tr.rolling(atr_window).mean() 
        upper = (high +low)/2 +multiplier*atr
        lower = (high +low)/2 -multiplier*atr
        
        st    = pd.Series(index=close.index, dtype=float)
        trend = 1
        for i in range(1, len(close)):
            if close.iloc[i] > upper.iloc[i-1]:   trend = 1
            elif close.iloc[i] < lower.iloc[i-1]: trend = -1
            st.iloc[i] = lower.iloc[i] if trend == 1 else upper.iloc[i]
        return st
        
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
        elif ind_t == "BB":
            window, std_dev = params
            df["BB_Mid"], df["BB_Upper"], df["BB_Lower"] = self.bollinger_bands(df["Close"], window, std_dev)
        elif ind_t == "MACD":
            fast, slow, signal = params
            df["MACD"], df["MACD_Signal"], df["MACD_Histogram"] = self.macd(df["Close"], fast, slow, signal)
        elif ind_t == "SAR":
            step, max_step = params
            df["SAR"] = self.parabolic_sar(df["High"], df["Low"], step, max_step)            
        elif ind_t == "ST":
            atr_w, mult = params
            df["Supertrend"] = self.supertrend(df["High"], df["Low"], df["Close"], atr_w, mult)
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
                df.loc[df["Short"] > df["Close"], "Signal"] = 1         # buy signal (MA)
                df.loc[df["Short"] < df["Close"], "Signal"] = -1        # sell signal (MA)
            elif len(params) == 2:
                # 2 MAs crossover
                df.loc[df["Short"] > df["Long"], "Signal"] = 1
                df.loc[df["Short"] < df["Long"], "Signal"] = -1
            elif len(params) == 3:
                # 3 MAs crossover
                df.loc[(df["Short"] > df["Mid"]) & (df["Mid"] > df["Long"]), "Signal"] = 1
                df.loc[(df["Short"] < df["Mid"]) & (df["Mid"] < df["Long"]), "Signal"] = -1
        elif ind_t == "BB":
            df.loc[df["Close"] < df["BB_Lower"], "Signal"] = 1          # buy signal (BB)
            df.loc[df["Close"] > df["BB_Lower"], "Signal"] = -1         # seel signal (BB)
        elif ind_t == "MACD":
            df.loc[df["MACD"] > df["MACD_Signal"], "Signal"] = 1        # buy signal (MACD)
            df.loc[df["MACD"] < df["MACD_Signal"], "Signal"] = -1       # sell signal (MACD)
        elif ind_t == "SAR":
            df.loc[df["Close"] > df["SAR"], "Signal"] = 1               # buy signal (SAR)
            df.loc[df["Close"] < df["SAR"], "Signal"] = -1              # sell signal (SAR)
        elif ind_t == "ST":
            df.loc[df["Close"] > df["Supertrend"], "Signal"] = 1        # buy signal (Supertrend)
            df.loc[df["Close"] < df["Supertrend"], "Signal"] = -1       # sell signal (Supertrend)
               
        # simulate execution (backtest)
        df["Position"] = df["Signal"].shift(1)                      # simulate position (using previous sample)
        df["Return"] = df["Close"].pct_change()                     # asset percentage variation (in relation to previous sample)
        df["Strategy"] = df["Position"]*df["Return"]                # return of the strategy
    
        # compare buy & hold vs current strategy
        df["Cumulative_Market"] = (1 +df["Return"]).cumprod()       # cumulative return buy & hold strategy
        df["Cumulative_Strategy"] = (1 +df["Strategy"]).cumprod()   # cumulative return current strategy
        return df
    
    def plot_price(self, axis, ticker):
        axis.plot(self.df.index, self.df["Close"], label=ticker)
        axis.grid(True)
        
    def plot_ma(self, axis, ind_t, params):
        if "Short" in self.df and len(params) >= 1:
            axis.plot(self.df.index, self.df["Short"], label=f"{ind_t}{params[0]}")
        if "Long" in self.df and len(params) == 2:
            axis.plot(self.df.index, self.df["Long"], label=f"{ind_t}{params[1]}")
        if "Long" in self.df and len(params) >= 3:
            axis.plot(self.df.index, self.df["Long"], label=f"{ind_t}{params[1]}")
        if "Med" in self.df and len(params) >= 3:
             axis.plot(self.df.index, self.df["Mid"], label=f"{ind_t}{params[2]}")
        
    def plot_bb(self, axis, params):
        axis.plot(self.df.index, self.df["BB_Mid"], label=f"BB mean {params[0]}")
        axis.plot(self.df.index, self.df["BB_Upper"], color='r', label=f"BB std {params[1]}")
        axis.plot(self.df.index, self.df["BB_Lower"], color='r')
        
    def plot_macd(self, axis):
        axis.plot(self.df.index, self.df["MACD"], label="MACD")
        axis.plot(self.df.index, self.df["MACD_Signal"], label="MACD_Signal")
        axis.bar(self.df.index, self.df["MACD_Histogram"], color='r', label="Histogram", alpha=0.4)
        axis.axhline(0, linewidth=1)
        axis.grid(True)
        
    def plot_sar(self, axis):
        axis.scatter(self.df.index, self.df["SAR"], color='orange', s=10, label="Parabolic SAR")
        
    def plot_supertrend(self, axis):
        axis.plot(self.df.index, self.df["Supertrend"], label="Supertrend")
    
    def plot(self, label):
        ticker, ind_t, *params = label.split("_")
        
        # plot price and indicator
        if ind_t in ["SMA", "EMA", "WMA"]:
            fig, axis = plt.subplots(figsize=(12,6))
            self.plot_price(axis, ticker)
            self.plot_ma(axis, ind_t, params)
            axis.legend()
            axis.set_title(f"{ticker} - Price")
        elif ind_t == "BB":
            fig, axis = plt.subplots(figsize=(12,6))
            self.plot_price(axis, ticker)
            self.plot_bb(axis, params)
            axis.legend()
            axis.set_title(f"{ticker} - Price")
        elif ind_t == "MACD":
            fig, (axis_price, axis_macd) = plt.subplots(2, 1, figsize=(12,8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            self.plot_price(axis_price, ticker)
            axis_price.set_title(f"{ticker} - Price")
            self.plot_macd(axis_macd)
        elif ind_t == "SAR":
            fig, axis = plt.subplots(figsize=(12,6))
            self.plot_price(axis, ticker)
            self.plot_sar(axis)
            axis.legend()
            axis.set_title(f"{ticker} - Price")
        elif ind_t == "ST":
            fig, axis = plt.subplots(figsize=(12,6))
            self.plot_price(axis, ticker)
            self.plot_supertrend(axis)
            axis.legend()
            axis.set_title(f"{ticker} - Price")
        plt.tight_layout()
        plt.savefig(f"data/results/{label}.png", dpi=300, bbox_inches="tight")
        plt.close()

        # plot returns
        plt.figure(figsize=(12,6))
        plt.plot(self.df.index, self.df["Cumulative_Market"], label="Buy & Hold")
        plt.plot(self.df.index, self.df["Cumulative_Strategy"], label="Strategy")
        plt.legend()
        plt.title(f"{ticker} - Backtest {ind_t}{'/'.join(params)}")
        plt.grid(True)
        plt.savefig(f"data/results/{label}_backtest.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        
class Exporter:
    @staticmethod
    def round_dataframe(df, n=4):
        df = df.copy()
        float_cols = df.select_dtypes(include=["float"]).columns
        df[float_cols] = df[float_cols].round(n)
        return df
            
    def export_dataframe(pro_data):
        # export dataframe for further analysis
        for ticker, ticker_debug in pro_data.items():
            with pd.ExcelWriter(f"data/debug/{ticker}.xlsx", engine="xlsxwriter") as writer:
                for sheet_name, df in ticker_debug.items():
                    # write to .xlsx
                    df = Exporter.round_dataframe(df)
                    df.to_excel(writer, sheet_name=sheet_name[:20])

    def export_results(res_data):
        # export backtesting results (a spreadsheet for each ticker)
        with pd.ExcelWriter("data/results/results.xlsx", engine="xlsxwriter") as writer:
            for ticker, res_df in res_data.items():
                # write to .xlsx
                res_df = pd.DataFrame.from_dict(res_df, orient="index")
                res_df = Exporter.round_dataframe(res_df)
                res_df.to_excel(writer, sheet_name=ticker[:10], index=False)