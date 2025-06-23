import joblib
from data.fetch_live_data import get_stock_data

def predict_profit_or_loss(symbol="RELIANCE.NS"):
    model = joblib.load("model/stock_model.pkl")
    features = get_stock_data(symbol)

    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0].max()

    result = "Profit" if prediction == 1 else "Loss"
    return {
        "prediction": result,
        "confidence": round(float(confidence), 2)
    }

# Example
if __name__ == "__main__":
    print(predict_profit_or_loss("RELIANCE.NS"))
