import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load historical stock data (simulate for now)
df = pd.read_csv("historical_data.csv")

# Simulate target: 1 = Profit, 0 = Loss
# You can refine this with real rules later
df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)  # 5-minute future

# Select features
features = ['Close', 'SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
X = df[features].dropna()
y = df['target'].loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Save
joblib.dump(model, "model/stock_model.pkl")
