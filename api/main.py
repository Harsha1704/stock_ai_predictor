from fastapi import FastAPI
from model.predictor import predict_profit_or_loss

app = FastAPI()  # ✅ THIS is what FastAPI is looking for

@app.get("/predict")
def predict(symbol: str = "RELIANCE.NS"):
    try:
        result = predict_profit_or_loss(symbol)
        return result
    except Exception as e:
        return {"error": str(e)}
