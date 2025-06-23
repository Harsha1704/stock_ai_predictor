import yfinance as yf
import pandas as pd
import numpy as np

def calculate_technical_indicators(df):
    # SMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # RSI
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    # Bollinger Bands
    df['BB_upper'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()

    # Drop NA rows
    df = df.dropna()

    return df

def get_stock_data(symbol="RELIANCE.NS", interval="1m", period="60m"):
    df = yf.download(tickers=symbol, interval=interval, period=period, progress=False)
    if df.empty:
        raise ValueError("No data fetched.")
    
    df = calculate_technical_indicators(df)
    latest = df.iloc[-1]

    features = {
        "close": latest["Close"],
        "sma_20": latest["SMA_20"],
        "rsi": latest["RSI"],
        "macd": latest["MACD"],
        "bb_upper": latest["BB_upper"],
        "bb_lower": latest["BB_lower"]
    }

    return pd.DataFrame([features])
