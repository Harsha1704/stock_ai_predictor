import joblib
from data.fetch_live_data import get_stock_data

def predict_profit_or_loss(symbol: str = "RELIANCE.NS"):
    try:
        model = joblib.load("model/stock_model.pkl")
        features = get_stock_data(symbol, period="7d", interval="1h")

        if features is None or features.empty:
            return {"error": f"No data found for {symbol}"}

        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0].max()

        return {
            "prediction": "Profit" if prediction == 1 else "Loss",
            "confidence": round(float(confidence), 2)
        }
    except Exception as e:
        return {"error": str(e)}
