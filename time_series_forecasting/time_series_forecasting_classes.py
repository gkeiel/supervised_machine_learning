# import modules
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


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
    def load_methods(self):
        methods = []
        with open(self.file_indicators, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    ind_t = parts[0]                                                # indicator title
                    ind_p = [float(x) if "." in x else int(x) for x in parts[1:]]   # indicator parameters
                    methods.append({"ind_t":ind_t, "ind_p":ind_p})
            return methods
        
    # collect OHLCVDS data from Yahoo Finance
    def download_data(self, ticker, start, end):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        except Exception as err:
            raise RuntimeError("Unexpected error in download_data.") from err
        df.columns = df.columns.droplevel(1)    
        df = df[["High", "Low", "Close", "Volume"]]
        return df


class Forecaster:
    def __init__(self, method, df):
        self.method = method
        self.df = df.copy()
        self.model  = None
        
    def build_features(self, y):
        X, Y = [], []
        for i in range(self.n_lags, len(y)):
            X.append(y.iloc[i-self.n_lags:i])
            Y.append(y.iloc[i])
        return np.array(X), np.array(Y)
        
    def predictions(self):
        df = self.df.coppy()
        ind_t  = self.method.get("ind_t", "")
        params = self.method.get("ind_p", [])
        y  = df["Close"]

        if ind_t != "ARIMA":
            self.n_estimators, self.max_depth, self.n_lags = params
            
            # build features for ML models:
            X, Y = self.build_features(y)
            
            # train data and test data
            X_train, Y_train = X[:self.N], Y[:self.N]
            X_test, Y_test   = X[self.N:], Y[self.N:]
            
            # define method and trainning
            model = self.MODELS[method](params)
            model.fit(X_train, Y_train)

            # predictions
            y_hat = model.predict(X_test)

        # add to dataframe
        pred = np.full(len(df), np.nan)
        pred[self.N+self.n_lags: self.N+self.n_lags+len(y_hat)] = y_hat
        df["Predicted_Close"] = pred
        return df, idx

    def metrics(df):
        valid_ar = df["y_ARIMA"].notna()
        valid_tree = df["y_DT"].notna()

        mse_ar = mean_squared_error(df.loc[valid_ar, "y"], df.loc[valid_ar, "y_ARIMA"])
        mse_tree = mean_squared_error(df.loc[valid_tree, "y"], df.loc[valid_tree, "y_DT"])
        return mse_ar, mse_tree

    def plot_res(df, mse_ar, mse_tree, label, idx):
        # plot predictions
        plt.figure(figsize=(12,6))
        plt.plot(df.index, df["y"], label="True y", color='black', linewidth=1.5)
        plt.plot(df.index, df["y_ARIMA"], label=f"AR model (MSE={mse_ar:.3f})", linestyle='--')
        plt.plot(df.index, df["y_DT"], label=f"Decision Tree (MSE={mse_tree:.3f})", linestyle=':')
        plt.title("AR vs Decision Tree Prediction")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{label}.png", dpi=300, bbox_inches="tight")
        plt.close()
        

class Backtester:
    def __init__(self, df):
        self.df = df.copy()

    def run_strategy(self, method):
        df     = self.df
        params = method["ind_p"]   
    
        # generate buy/sell signals
        df["Signal"] = 0
        df.loc[df["Predicted_Close"] > (1 +self.hysteresis/100)*df["Close"], "Signal"] = 1
        df.loc[df["Predicted_Close"] < (1 -self.hysteresis/100)*df["Close"], "Signal"] = -1
                    
        # simulate execution (backtest)
        df["Position"] = df["Signal"].shift(1)                      # simulate position (using previous sample)
        #df.loc[df["Position"] == -1, "Position"] = 0                # comment if also desired selling operations  
        df["Return"] = df["Close"].pct_change()                     # asset percentage variation (in relation to previous sample)
        df["Strategy"] = df["Position"]*df["Return"]                # return of the strategy
            
        # compare buy & hold vs current strategy
        df["Cumulative_Market"] = (1 +df["Return"]).cumprod()       # cumulative return buy & hold strategy
        df["Cumulative_Strategy"] = (1 +df["Strategy"]).cumprod()   # cumulative return current strategy
        return df


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