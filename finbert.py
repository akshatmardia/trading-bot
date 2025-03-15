import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
labels = ["positive", "negative", "neutral"]

def sentiment_from_model(news):
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        result = nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability.item(), sentiment
    else:
        return 0, labels[2]