from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import torch.nn.functional as F

# Load model and tokenizer
model_name = "LiYuan/amazon-review-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Label mapping
label_map = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

def analyze_sentiment(reviews):
    results = []
    for text in reviews:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            label_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][label_id].item()

        label = label_map[label_id]
        results.append({
            "review": text,
            "sentiment": label,
            "confidence": round(confidence, 2)
        })

    return pd.DataFrame(results)
