import yfinance as yf
import pandas as pd

def get_stock_data(symbol: str, period: str = "7d", interval: str = "1h"):
    df = yf.download(tickers=symbol, interval=interval, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    df['Change'] = df['Close'].pct_change().fillna(0)
    df['MA7'] = df['Close'].rolling(window=7).mean().fillna(method='bfill')

    latest = df.iloc[-1]
    return pd.DataFrame([{
        "Open": latest['Open'],
        "High": latest['High'],
        "Low": latest['Low'],
        "Close": latest['Close'],
        "Volume": latest['Volume'],
        "Change": latest['Change'],
        "MA7": latest['MA7']
    }])
