# pyrefly: ignore [missing-import]
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
# pyrefly: ignore [missing-import]  
import torch

# Load your trained model
MODEL_PATH = "models/distilbert-imdb"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

labels = {
    0: "NEGATIVE",
    1: "POSITIVE"
}

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()

    return {
        "label": labels[prediction],
        "confidence": round(confidence, 4)
    }