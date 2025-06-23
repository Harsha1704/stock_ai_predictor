import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

def calculate_technical_indicators(df):
    # Adjust column names to remove symbol suffix if present
    if any(col.endswith('AAPL') for col in df.columns):
        df.columns = [col.replace(' AAPL', '') for col in df.columns]
    # Convert columns to float to avoid issues
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = df[col].astype(float)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0).flatten()
    loss = np.where(delta < 0, -delta, 0).flatten()
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['BB_upper'] = df['SMA_20'] + 2 * pd.Series(df['Close'].values.flatten()).rolling(window=20).std()
    df['BB_lower'] = df['SMA_20'] - 2 * pd.Series(df['Close'].values.flatten()).rolling(window=20).std()
    print(f"Shape before dropna in calculate_technical_indicators: {df.shape}")
    # Instead of dropna, fill NaN values with method 'bfill' or 'ffill'
    df = df.fillna(method='bfill').fillna(method='ffill')
    print(f"Shape after filling NaNs in calculate_technical_indicators: {df.shape}")
    return df

def create_lag_features(df, lag=5):
    print(f"Shape before creating lag features: {df.shape}")
    print(f"Columns before creating lag features: {df.columns}")
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    print(f"Shape before dropna in create_lag_features: {df.shape}")
    df = df.dropna()
    print(f"Shape after dropna in create_lag_features: {df.shape}")
    return df

def load_data(symbol="AAPL", start="2023-01-01", end="2025-01-01", interval="1d"):
    df = yf.download(symbol, start=start, end=end, interval=interval)
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]
    print(f"Raw data shape: {df.shape}")
    print(f"Raw data head:\n{df.head()}")
    df = calculate_technical_indicators(df)
    print(f"Shape after calculating technical indicators: {df.shape}")
    print(f"Head after calculating technical indicators:\n{df.head()}")
    df = create_lag_features(df)
    print(f"Shape after creating lag features: {df.shape}")
    print(f"Head after creating lag features:\n{df.head()}")
    return df

def prepare_data(df):
    features = ['Close', 'SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower'] + [f'lag_{i}' for i in range(1,6)]
    X = df[features].values
    y = (df['Close'].shift(-1) > df['Close']).astype(int).values[:-1]
    X = X[:-1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    return X_scaled, y, scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    df = load_data()
    print(f"Dataframe shape after loading and processing: {df.shape}")
    X, y, scaler = prepare_data(df)
    print(f"Feature matrix shape: {X.shape}, Target vector shape: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")
    model.save("model/lstm_stock_model.h5")
    joblib.dump(scaler, "model/scaler.pkl")

if __name__ == "__main__":
    train_model()
