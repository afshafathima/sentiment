from fastapi import FastAPI
from app.predict import predict_sentiment

app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API"}

@app.post("/predict")
def predict(text: str):
    result = predict_sentiment(text)
    return {
        "review": text,
        "prediction": result["label"],
        "confidence": result["confidence"]
    }