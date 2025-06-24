from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import torch.nn.functional as F

# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Map model output labels to human-friendly form
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def analyze_sentiment(reviews):
    results = []
    for text in reviews:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            label_id = torch.argmax(probs, dim=1).item()
            label = label_map[f"LABEL_{label_id}"]
            confidence = probs[0][label_id].item()

        results.append({
            "review": text,
            "sentiment": label,
            "confidence": round(confidence, 2)
        })

    return pd.DataFrame(results)
