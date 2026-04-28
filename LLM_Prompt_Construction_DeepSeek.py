# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI
from time import sleep

#data and split
df = pd.read_csv("sp500_headlines_with_prices.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

df["next_open"] = df["open_price"].shift(-1)

df["label"] = (df["next_open"] > df["close_price"]).astype(int)

df = df.dropna().reset_index(drop=True)

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

X_train = train_df[["combined_headline"]]
y_train = train_df["label"]

X_test = test_df[["combined_headline"]]
y_test = test_df["label"]


#setup
LLM_API_KEY = "sk-j49dRg-v03_qWy-1xijenA"
CACHE_FILE = "deepseek_cache.json"

import json, os

# Persistent cache: survives crashes and restarts
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        cache = json.load(f)
else:
    cache = {}

def run_llm(prompt, model="deepseek-r1", return_first_line=False):
    cache_key = f"{model}||{prompt}"

    if cache_key in cache:
        return cache[cache_key]

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url="https://ol.sci.pitt.edu",
        timeout=60.0,
    )

    response_text = None

    for attempt, wait in enumerate([0, 6, 15]):
        try:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=0,
                max_tokens=1024,
                top_p=1,
                timeout=60,
            )
            response_text = response.choices[0].text.strip()
            break
        except Exception as e:
            print(f"Retry {attempt+1}/3 error: {e}")
            if wait:
                sleep(wait)

    if response_text is None:
        return None

    if return_first_line:
        response_text = response_text.split('\n')[0]
        response_text = response_text.split('.')[0] + '.'

    cache[cache_key] = response_text
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    return response_text

#prompts and helpers
def build_zero_prompt(headlines):
    return f"""
### Instruction:

You are a financial market analyst.

Given financial news headlines from one trading day, predict whether the S&P 500 will go UP or DOWN/FLAT the next trading day. You are predicting closing to openeing price.

Respond with exactly ONE word:
up, down, or flat

### Input:
{headlines}

### Output:
"""

def get_few_shot_examples(train_df, n_per_class=3, seed=42):
    
    class_1 = train_df[train_df["label"] == 1].sample(n_per_class, random_state=seed)
    class_0 = train_df[train_df["label"] == 0].sample(n_per_class, random_state=seed)

    examples = pd.concat([class_1, class_0]).sample(frac=1, random_state=seed)

    return examples.reset_index(drop=True)

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Results:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["0", "1"],
        digits=2
    ))
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 2))

def build_few_prompt(headlines, examples):

    example_text = ""

    for _, row in examples.iterrows():
        example_text += f"""
Input:
{row['combined_headline']}
Output:
{"up" if row['label'] == 1 else "down"}
"""

    return f"""
### Instruction:

You are a financial market analyst.

Given financial news headlines from one trading day, predict whether the S&P 500 will go UP or DOWN/FLAT the next trading day. You are predicting closing to openeing price.

Respond with exactly ONE word:
up, down, or flat

---

### Examples:
{example_text}

---

### Input:
{headlines}

### Output:
"""

UP = ["up", "increase", "rise"]
DOWN = ["down", "decrease", "fall"]
FLAT = ["flat", "same", "neutral"]

def parse_output(text):
    if text is None:
        return None
    
    text = text.lower()

    if any(w in text for w in UP):
        return 1
    elif any(w in text for w in DOWN + FLAT):
        return 0
    else:
        return None

#run zero
preds = []
labels = []

#X_test_small = X_test.head(30)
#y_test_small = y_test.head(30)
X_test_full = X_test
y_test_full = y_test

for i in tqdm(range(len(X_test_full))):

    headlines = X_test_full.iloc[i]["combined_headline"]
    
    prompt = build_zero_prompt(headlines)
    output = run_llm(prompt)
    
    pred = parse_output(output)

    if pred is not None:
        preds.append(pred)
        labels.append(y_test_full.iloc[i])

print("Zero-Shot")
print("Accuracy:", accuracy_score(labels, preds))
print("Precision:", precision_score(labels, preds))
print("Recall:", recall_score(labels, preds))
print("F1:", f1_score(labels, preds))
evaluate_model("deepseek‑r1: Zero-Shot", labels, preds)

#few shot
predsF = []
labelsF = []

X_test_full = X_test
y_test_full = y_test

few_shot_examples = get_few_shot_examples(train_df)  # fixed seed=42, compute once

for i in tqdm(range(len(X_test_full))):

    headlines = X_test_full.iloc[i]["combined_headline"]

    prompt = build_few_prompt(headlines, few_shot_examples)
    output = run_llm(prompt)
    
    predF = parse_output(output)

    if predF is not None:
        predsF.append(predF)
        labelsF.append(y_test_full.iloc[i])

print("Few-Shot")
print("Accuracy:", accuracy_score(labelsF, predsF))
print("Precision:", precision_score(labelsF, predsF))
print("Recall:", recall_score(labelsF, predsF))
print("F1:", f1_score(labelsF, predsF))
evaluate_model(" deepseek‑r1: Few-Shot", labelsF, predsF)

# %%
