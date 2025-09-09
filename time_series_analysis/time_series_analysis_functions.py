import yfinance as yf
import matplotlib.pyplot as plt


def download_data(ticker, start, end):
    # collect OHLCVDS data from Yahoo Finance
    df = yf.download(ticker, start, end)
    df = df[["Close"]]
    return df


def plot_res(df):
    # plot results
    plt.figure(1, figsize=(12,6))
    plt.plot(df.index, df["Close"], label="PETR4")
    plt.title(f"Data")
    plt.grid(True)

    plt.figure(2, figsize=(12,6))
    plt.plot(df.index, df["Cumulative_Market"], label="Buy & Hold")
    plt.plot(df.index, df["Cumulative_Strategy"], label="Strategy")
    plt.title(f"Backtest")
    plt.legend()
    plt.grid(True)
    plt.show()