import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load model and tokenizer as per official Hugging Face docs
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Labels defined by the model
labels = ['Negative', 'Neutral', 'Positive']

def analyze_sentiment(reviews):
    results = []
    for text in reviews:
        text = text.strip().replace('\n', ' ')
        encoded_input = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = model(**encoded_input)
        scores = output.logits[0].numpy()
        scores = softmax(scores)
        label_id = scores.argmax()
        sentiment = labels[label_id]
        confidence = float(scores[label_id])

        results.append({
            "review": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 2)
        })
    return pd.DataFrame(results)
