import sys
import os
from fastapi import FastAPI

# Fix module path to import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.predictor import predict_profit_or_loss
from tensorflow.keras.models import load_model
import joblib

app = FastAPI()

# Load model and scaler once at startup
model = load_model("model/lstm_stock_model.h5")
scaler = joblib.load("model/scaler.pkl")

# ✅ Optional root route to avoid 404 on "/"
@app.get("/")
def root():
    return {
        "message": "Welcome to AI Stock Predictor. Use /predict?symbol=RELIANCE.NS"
    }

# ✅ Prediction endpoint
@app.get("/predict")
def predict(symbol: str = "RELIANCE.NS"):
    return predict_profit_or_loss(symbol, model, scaler)
