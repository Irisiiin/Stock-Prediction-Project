import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI
from time import sleep
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#data and split
df = pd.read_csv("sp500_headlines_with_prices.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Create binary label: next open vs current close
df["next_open"] = df["open_price"].shift(-1)

df["label"] = (df["next_open"] > df["close_price"]).astype(int)

df = df.dropna().reset_index(drop=True)

# Split
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

X_test = test_df[["combined_headline"]]
y_test = test_df["label"]

model_name = "ProsusAI/finbert"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

def finbert_predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    labels = ["up", "down", "flat"]
    pred_label = labels[torch.argmax(probs).item()]

    return pred_label

def map_to_binary(sentiment):
    if sentiment == "up":
        return 1
    else:
        return 0

preds_finbert = []
labels_finbert = []

for i in tqdm(range(len(X_test))):

    text = X_test.iloc[i]["combined_headline"]

    sentiment = finbert_predict(text)
    pred = map_to_binary(sentiment)

    preds_finbert.append(pred)
    labels_finbert.append(y_test.iloc[i])

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Results:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["0", "1"],
        digits=2
    ))
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 2))

evaluate_model("FinBERT", labels_finbert, preds_finbert)
