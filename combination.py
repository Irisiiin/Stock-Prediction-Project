# %%
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
import os

# =========================================================
# CONFIG
# =========================================================
DATA_FILE  = "sp500_headlines_with_prices.csv"
CACHE_FILE = "finbert_probs_combo.npy"
SEQ_LEN    = 10

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

np.random.seed(42)

# =========================================================
# 1. LOAD DATA & TARGET
# =========================================================
df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Target: close >= open (intraday direction, consistent with StockPredict 64.1% baseline)
df["target"] = (df["close_price"] >= df["open_price"]).astype(int)

print(f"Total rows: {len(df)}")
print(f"Target distribution: {df['target'].value_counts().to_dict()}")

# =========================================================
# 2. FINBERT SENTIMENT (batch inference, cached to disk)
# =========================================================
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert   = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
finbert.eval()

def get_finbert_probs(texts, batch_size=32):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt",
            truncation=True, max_length=512, padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = finbert(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)

if os.path.exists(CACHE_FILE):
    print("Loading cached FinBERT probs...")
    finbert_probs = np.load(CACHE_FILE)
else:
    print("Running FinBERT inference (this takes a few minutes on GPU)...")
    texts = df["combined_headline"].fillna("neutral").tolist()
    finbert_probs = get_finbert_probs(texts)
    np.save(CACHE_FILE, finbert_probs)
    print(f"Saved to {CACHE_FILE}")

# FinBERT label mapping: {0: positive, 1: negative, 2: neutral}
df["sent_pos"] = finbert_probs[:, 0]
df["sent_neg"] = finbert_probs[:, 1]
df["sent_neu"] = finbert_probs[:, 2]

# Free FinBERT from GPU/MPS memory before heavy feature work
del finbert, tokenizer
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
import gc; gc.collect()

print("FinBERT features added. Model freed from memory.")

# =========================================================
# 3. TECHNICAL INDICATORS (pure pandas, no TA-Lib required)
# =========================================================
df["daily_return"] = (df["close_price"] - df["open_price"]) / df["open_price"]
df["daily_range"]  = (df["high_price"]  - df["low_price"])  / df["open_price"]
df["ma5_ratio"]    = df["close_price"] / df["close_price"].rolling(5).mean()
df["ma10_ratio"]   = df["close_price"] / df["close_price"].rolling(10).mean()

def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs  = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100.0

df["rsi14"]     = compute_rsi(df["close_price"], 14)
df["vol_ratio"] = df["volume"] / df["volume"].rolling(5).mean()

FEATURE_COLS = [
    "sent_pos", "sent_neg", "sent_neu",        # FinBERT sentiment:  3 dims
    "daily_return", "ma5_ratio", "ma10_ratio",  # price features:     3 dims
    "rsi14", "vol_ratio",                       # technical:          2 dims
]

df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)

print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Rows after dropna: {len(df)}")

# =========================================================
# 4. TRAIN / TEST SPLIT (chronological, no shuffle)
# =========================================================
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].reset_index(drop=True)
test_df  = df.iloc[split_index:].reset_index(drop=True)

scaler = StandardScaler()
train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

# =========================================================
# 5. SLIDING WINDOW SEQUENCES → flatten for XGBoost
#    X shape before flatten: (samples, SEQ_LEN, n_features)
#    X shape after  flatten: (samples, SEQ_LEN * n_features)
#    Each sequence = previous SEQ_LEN days' features → predict next day
# =========================================================
def make_sequences(df, seq_len, feature_cols, label_col="target"):
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[feature_cols].iloc[i-seq_len:i].values)
        y.append(df[label_col].iloc[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train_seq, y_train = make_sequences(train_df, SEQ_LEN, FEATURE_COLS)
X_test_seq,  y_test  = make_sequences(test_df,  SEQ_LEN, FEATURE_COLS)

print(f"X_train (seq): {X_train_seq.shape} | X_test (seq): {X_test_seq.shape}")

# Flatten: (N, 10, 8) → (N, 80)
X_train = X_train_seq.reshape(len(X_train_seq), -1)
X_test  = X_test_seq.reshape(len(X_test_seq),  -1)

# Feature names: sent_pos_t-9, ..., vol_ratio_t0
flat_cols = [f"{feat}_t{t - SEQ_LEN + 1}" for t in range(SEQ_LEN) for feat in FEATURE_COLS]
print(f"XGBoost input: {X_train.shape}  ({len(flat_cols)} features)")

# =========================================================
# 6. XGBOOST CLASSIFIER
# =========================================================
n_pos = int(y_train.sum())
n_neg = len(y_train) - n_pos
print(f"Class balance — pos: {n_pos}, neg: {n_neg}, ratio: {n_neg/n_pos:.3f}")

xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

clf = XGBClassifier(
    n_estimators        = 500,
    max_depth           = 4,
    learning_rate       = 0.05,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    min_child_weight    = 5,
    gamma               = 0.1,
    random_state        = 42,
    eval_metric         = "logloss",
    early_stopping_rounds = 30,
    device              = xgb_device,
    verbosity           = 0,
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

print(f"Best iteration: {clf.best_iteration}")

# =========================================================
# 7. EVALUATE
# =========================================================
all_probs = clf.predict_proba(X_test)[:, 1]
all_preds = (all_probs > 0.5).astype(int)
all_labels = y_test.astype(int)

print("\n================ COMBINATION MODEL RESULTS ================")
print(classification_report(all_labels, all_preds))
print("Accuracy :", round(accuracy_score(all_labels, all_preds), 4))
print("ROC-AUC  :", round(roc_auc_score(all_labels, all_probs),  4))
print("\n--- Baseline comparison ---")
print(f"Random baseline (always up)     : {sum(all_labels)/len(all_labels):.1%}")
print(f"StockPredict FinBERT→LinearSVC  : 64.1%")
print(f"Combination  FinBERT+Price→XGB  : {accuracy_score(all_labels, all_preds):.1%}")

# =========================================================
# 8. FEATURE IMPORTANCE (top 20)
# =========================================================
fi = pd.Series(clf.feature_importances_, index=flat_cols).sort_values(ascending=False)
print("\n--- Top 20 feature importances ---")
print(fi.head(20).to_string())

# %%
