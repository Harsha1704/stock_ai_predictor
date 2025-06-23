import numpy as np
import joblib
from tensorflow.keras.models import load_model
from data.fetch_live_data import get_stock_data

def predict_profit_or_loss(symbol: str = "RELIANCE.NS", model=None, scaler=None):
    try:
        if model is None or scaler is None:
            from tensorflow.keras.models import load_model
            import joblib
            model = load_model("model/lstm_stock_model.h5")
            scaler = joblib.load("model/scaler.pkl")

        features_df = get_stock_data(symbol, period="7d", interval="1h")

        if features_df is None or features_df.empty:
            return {"error": f"No data found for {symbol}"}

        features = features_df.values
        features_scaled = scaler.transform(features)
        features_scaled = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))

        prediction_prob = model.predict(features_scaled)
        prediction = (prediction_prob > 0.5).astype(int)[0][0]

        confidence = float(prediction_prob[0][0])

        return {
            "prediction": "Profit" if prediction == 1 else "Loss",
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        return {"error": str(e)}
