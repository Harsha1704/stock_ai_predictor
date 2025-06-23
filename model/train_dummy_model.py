import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Dummy data
X = pd.DataFrame({
    "Open": [100, 105, 95],
    "High": [110, 108, 97],
    "Low": [90, 102, 93],
    "Close": [105, 103, 94],
    "Volume": [10000, 15000, 12000],
    "Change": [0.05, -0.01, -0.03],
    "MA7": [102, 101, 100]
})

y = [1, 0, 0]  # 1 = Profit, 0 = Loss

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model/stock_model.pkl")