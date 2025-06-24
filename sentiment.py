import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Define model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

# 🔥 FIX: Disable fast tokenizer to avoid merges_file error
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Model labels
labels = ['Negative', 'Neutral', 'Positive']

def analyze_sentiment(reviews):
    results = []
    for text in reviews:
        text = text.strip().replace('\n', ' ')
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            output = model(**encoded_input)
        scores = output.logits[0].numpy()
        probs = softmax(scores)
        label_id = probs.argmax()
        sentiment = labels[label_id]
        confidence = float(probs[label_id])
        results.append({
            "review": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 2)
        })
    return pd.DataFrame(results)
