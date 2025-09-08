import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # collect data from Yahoo Finance
    ticker = "PETR4.SA"
    df = yf.download(ticker, start="2024-01-01", end="2025-09-05")
    df = df[["Close"]]

    # calculate indicators
    df["SMA20"] = df["Close"].rolling(window=10).mean()  # short MA
    df["SMA50"] = df["Close"].rolling(window=20).mean()  # long MA

    # generate signals
    df["Signal"] = 0
    df.loc[df["SMA20"] > df["SMA50"], "Signal"] = 1   # buy
    df.loc[df["SMA20"] < df["SMA50"], "Signal"] = -1  # sell

    # simulate execution (backtest)
    df["Position"] = df["Signal"].shift(1)
    df["Return"] = df["Close"].pct_change()
    df["Strategy"] = df["Position"] * df["Return"]

    # compare buy & hold vs strategy
    df["Cumulative_Market"] = (1 + df["Return"]).cumprod()
    df["Cumulative_Strategy"] = (1 + df["Strategy"]).cumprod()

    # plot result
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["Cumulative_Market"], label="Buy & Hold")
    plt.plot(df.index, df["Cumulative_Strategy"], label="Strategy")
    plt.legend()
    plt.title(f"Backtest {ticker} - Cruzamento de Médias")
    plt.show()


if __name__ == "__main__":
    main()