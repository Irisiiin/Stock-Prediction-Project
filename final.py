import pandas as pd
import numpy as np
import re
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

import shap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "ProsusAI/finbert"

np.random.seed(42)
torch.manual_seed(42)

# =========================================================
# LOAD MODELS
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(model_name)

finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
bert_model = AutoModel.from_pretrained(model_name).to(device)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("sp500_headlines_with_prices sep ~.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

price_cols = ["open_price", "high_price", "low_price", "close_price", "volume"]

df[price_cols] = df[price_cols].shift(1)
df["combined_headline"] = df["combined_headline"].shift(1)

df["target"] = (df["open_price"].shift(-1) > df["close_price"]).astype(int)

df["headline_list"] = df["combined_headline"].fillna("").apply(
    lambda x: x.split("~") if isinstance(x, str) else []
)

df["news_t"] = df["combined_headline"].fillna("")
df["news_lag1"] = df["news_t"].shift(1).fillna("")
df["news_lag2"] = df["news_t"].shift(2).fillna("")
df["news_lag3"] = df["news_t"].shift(3).fillna("")

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.2,
    random_state=42,
    stratify=df["target"]
)

train_df = df.loc[train_idx].copy().reset_index(drop=True)
test_df = df.loc[test_idx].copy().reset_index(drop=True)

# =========================================================
# 🆕 CLS (FIXED)
# =========================================================

def get_cls_embeddings(texts, batch_size=16):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = bert_model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

train_text = (
    train_df["news_t"].fillna("") + " " +
    train_df["news_lag1"].fillna("")
).tolist()

test_text = (
    test_df["news_t"].fillna("") + " " +
    test_df["news_lag1"].fillna("")
).tolist()

train_cls = get_cls_embeddings(train_text)
test_cls = get_cls_embeddings(test_text)

pca = PCA(n_components=12)

train_cls = pca.fit_transform(train_cls)
test_cls = pca.transform(test_cls)

cls_train_df = pd.DataFrame(train_cls, columns=[f"emb_{i}" for i in range(12)])
cls_test_df = pd.DataFrame(test_cls, columns=[f"emb_{i}" for i in range(12)])

train_df = pd.concat([train_df.reset_index(drop=True), cls_train_df], axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), cls_test_df], axis=1)

# =========================================================
# 🆕 TF-IDF (FIXED)
# =========================================================

tfidf = TfidfVectorizer(max_features=5)

train_text = (
    train_df["news_t"] + " " +
    train_df["news_lag1"] + " " 

)

test_text = (
    test_df["news_t"] + " " +
    test_df["news_lag1"] + " " 

)

tfidf_train = tfidf.fit_transform(train_text).toarray()
tfidf_test = tfidf.transform(test_text).toarray()

tfidf_train_df = pd.DataFrame(tfidf_train, columns=[f"tfidf_{i}" for i in range(5)])
tfidf_test_df = pd.DataFrame(tfidf_test, columns=[f"tfidf_{i}" for i in range(5)])

train_df = pd.concat([train_df, tfidf_train_df], axis=1)
test_df = pd.concat([test_df, tfidf_test_df], axis=1)

# =========================================================
# SENTIMENT MODEL
# (UNCHANGED)
# =========================================================

def get_sentiment_batch(texts, batch_size=32):
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = finbert_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)

def compute_sentiment(df):
    neg_mean, neu_mean, pos_mean = [], [], []
    neg_max, pos_max = [], []
    sent_vol = []

    for headlines in df["headline_list"]:
        headlines = [h for h in headlines if isinstance(h, str) and h.strip()]
        if len(headlines) == 0:
            headlines = ["neutral market"]

        sentiments = get_sentiment_batch(headlines)

        neg = sentiments[:, 0]
        neu = sentiments[:, 1]
        pos = sentiments[:, 2]

        neg_mean.append(np.mean(neg))
        neu_mean.append(np.mean(neu))
        pos_mean.append(np.mean(pos))

        neg_max.append(np.max(neg))
        pos_max.append(np.max(pos))

        sent_vol.append(np.std(pos - neg))

    df["sent_neg_mean"] = neg_mean
    df["sent_neu_mean"] = neu_mean
    df["sent_pos_mean"] = pos_mean
    df["sent_neg_max"] = neg_max
    df["sent_pos_max"] = pos_max
    df["sent_volatility"] = sent_vol
    df["sent_polarity"] = df["sent_pos_mean"] - df["sent_neg_mean"]


    return df

train_df = compute_sentiment(train_df)
test_df = compute_sentiment(test_df)

# =========================================================
# SENTIMENT LAGS + FEATURES (UNCHANGED)
# =========================================================

sent_cols = [
    "sent_neg_mean",
    "sent_neu_mean",
    "sent_pos_mean",
    "sent_polarity",
    "sent_volatility"
]

for df_ in [train_df, test_df]:
    for lag in [1,2]:
        for col in sent_cols:
            df_[f"{col}_lag{lag}"] = df_[col].shift(lag)

    lag_cols = []
    for lag in [1,2]:
        lag_cols += [
            f"sent_neg_mean_lag{lag}",
            f"sent_neu_mean_lag{lag}",
            f"sent_pos_mean_lag{lag}",
            f"sent_polarity_lag{lag}",
            f"sent_volatility_lag{lag}"
        ]

    df_[lag_cols] = df_[lag_cols].fillna(0)


def get_market_signals(text):
    text = str(text).lower()

    patterns = {
        "flag_index": r"\b(s&p|s&p 500|sp500|spx|spy|voo|gspc|market)\b",
        "flag_fed": r"\b(fed|federal reserve|powell|fomc|interest rate(s)?|hawkish|dovish|yield(s)?)\b",
        "flag_macro": r"\b(cpi|ppi|inflation|gdp|unemployment|nfp|retail sales|pce)\b",
        "flag_mag7": r"\b(aapl|apple|msft|microsoft|nvda|nvidia|goog|google|amzn|amazon|meta|tsla|tesla)\b",
        "flag_risk": r"\b(war|tariff(s)?|sanction(s)?|recession|crisis|election|default|vix|geopolitical)\b"
    }

    return pd.Series({
        flag: int(bool(re.search(pattern, text)))
        for flag, pattern in patterns.items()
    })


# =========================================================
# FEATURE ENGINEERING (UNCHANGED)
# =========================================================

for df_ in [train_df, test_df]:

    signals = df_["combined_headline"].apply(get_market_signals)
    df_ = pd.concat([df_, signals], axis=1)
        

    df_["sent_polarity"] = df_["sent_pos_mean"] - df_["sent_neg_mean"]
        

    df_["daily_range"] = df_["high_price"] - df_["low_price"]
    df_["close_open_diff"] = df_["close_price"] - df_["open_price"]
    df_["daily_return"] = df_["close_open_diff"] / df_["open_price"]

    df_["volatility_3"] = df_["daily_return"].shift(1).rolling(3).std()
    df_["volatility_5"] = df_["daily_return"].shift(1).rolling(5).std()

    df_["momentum_3"] = df_["close_price"].pct_change(3, fill_method=None)
    df_["momentum_5"] = df_["close_price"].pct_change(5, fill_method=None)

    df_["sma_5"] = df_["close_price"].rolling(5).mean()
    df_["sma_10"] = df_["close_price"].rolling(10).mean()

    df_["close_sma_ratio"] = df_["close_price"] / df_["sma_5"]

    df_["ema_5"] = df_["close_price"].ewm(span=5).mean()
    df_["ema_10"] = df_["close_price"].ewm(span=10).mean()

    df_["ema_ratio"] = df_["ema_5"] / df_["ema_10"]

    df_["price_acceleration"] = df_["close_price"].diff().diff()
    
    signals = df_["combined_headline"].apply(get_market_signals)
    df_ = pd.concat([df_, signals], axis=1)
    

    #df_["sent_momentum"] = df_["sent_pos_mean"] - df_["sent_pos_mean"].shift(1)
    #df_["sent_acceleration"] = df_["sent_momentum"].diff()
    #df_["sent_extremity"] = df_["sent_pos_max"] - df_["sent_neg_max"]

    for col in ["open_price", "high_price", "low_price", "close_price", "volume"]:
        for lag in [1,2]:
            df_[f"{col}_lag{lag}"] = df_[col].shift(lag)
    


# =========================================================
# FEATURE SELECTION (FIXED)
# =========================================================

#drop_cols = ["Date", "target", "combined_headline", "headline_list","news_t","news_lag1", "news_lag2","news_lag3"]+ list(tfidf_train_df.columns) + list(cls_train_df.columns)
drop_cols = ["Date", "target", "combined_headline", "headline_list","news_t","news_lag1", "news_lag2","news_lag3"] 

feature_cols = [c for c in train_df.columns if c not in drop_cols]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)


y_train = train_df["target"]
y_test = test_df["target"]

# =========================================================
# MODEL
# =========================================================

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.9,
    reg_lambda=0.9,
    tree_method="hist",
    device="cuda"
)

xgb_model.fit(X_train, y_train)

# =========================================================
# RESULTS
# =========================================================

# 1. Get the probabilities for the 'Up' class (Class 1)
test_probs = xgb_model.predict_proba(X_test)[:, 1]

custom_threshold = 0.539
test_preds_conservative = (test_probs >= custom_threshold).astype(int)

# 3. Evaluate
print(classification_report(y_test, test_preds_conservative))

print("\n================ FINAL TEST RESULTS ================")
print("Accuracy:", accuracy_score(y_test,test_preds_conservative))
print("ROC-AUC:", roc_auc_score(y_test, test_preds_conservative))



# =========================================================
# SHAP
# =========================================================

X_test = X_test.astype(np.float32)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n===== TOP 20 FEATURES =====")
print(importance.head(20))