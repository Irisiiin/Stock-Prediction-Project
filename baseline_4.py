import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("sp500_headlines_with_prices.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# -----------------------------
# 2. Basic Feature Engineering
# -----------------------------
df["daily_range"] = df["high_price"] - df["low_price"]
df["close_open_diff"] = df["close_price"] - df["open_price"]
df["daily_return"] = df["close_open_diff"] / df["open_price"]

# -----------------------------
# 3. Volatility Features
# -----------------------------
# Rolling volatility (standard deviation of returns)
df["volatility_3"] = df["daily_return"].rolling(window=3).std()
df["volatility_5"] = df["daily_return"].rolling(window=5).std()
df["volatility_10"] = df["daily_return"].rolling(window=10).std()

# Parkinson volatility (uses high/low)
df["parkinson_vol"] = np.sqrt(
    (1 / (4 * np.log(2))) * (np.log(df["high_price"] / df["low_price"]) ** 2)
)

# -----------------------------
# 4. Momentum Features
# -----------------------------
# Price momentum (rate of change)
df["momentum_3"] = df["close_price"].pct_change(periods=3)
df["momentum_5"] = df["close_price"].pct_change(periods=5)
df["momentum_10"] = df["close_price"].pct_change(periods=10)

# Volume momentum
df["volume_momentum_3"] = df["volume"].pct_change(periods=3)

# -----------------------------
# 5. Trend Indicators
# -----------------------------
# Moving averages
df["sma_5"] = df["close_price"].rolling(window=5).mean()
df["sma_10"] = df["close_price"].rolling(window=10).mean()
df["sma_20"] = df["close_price"].rolling(window=20).mean()

# Price relative to moving averages
df["close_sma5_ratio"] = df["close_price"] / df["sma_5"]
df["close_sma10_ratio"] = df["close_price"] / df["sma_10"]

# EMA (trend following)
df["ema_5"] = df["close_price"].ewm(span=5, adjust=False).mean()
df["ema_10"] = df["close_price"].ewm(span=10, adjust=False).mean()

# Trend direction (binary)
df["trend_sma"] = (df["sma_5"] > df["sma_10"]).astype(int)
df["trend_ema"] = (df["ema_5"] > df["ema_10"]).astype(int)

# -----------------------------
# 6. Lag Features
# -----------------------------
lag_features = ["open_price", "high_price", "low_price", "close_price", "volume"]

for col in lag_features:
    for lag in [1, 2, 3, 4, 5]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

for lag in [1, 2, 3, 4, 5]:
    df[f"daily_return_lag{lag}"] = df["daily_return"].shift(lag)
    df[f"daily_range_lag{lag}"] = df["daily_range"].shift(lag)

# -----------------------------
# 7. Targets
# -----------------------------
df["target_close"] = (df["close_price"].shift(-1) > df["close_price"]).astype(int)
df["target_open"] = (df["open_price"].shift(-1) > df["close_price"]).astype(int)

# Drop NaNs
df = df.dropna().reset_index(drop=True)

# -----------------------------
# 8. Feature Selection
# -----------------------------
X = df[[
    "combined_headline",

    # raw prices
    "open_price", "high_price", "low_price", "close_price", "volume",

    # engineered
    "daily_range", "close_open_diff", "daily_return",

    # volatility
    "volatility_3", "volatility_5", "volatility_10", "parkinson_vol",

    # momentum
    "momentum_3", "momentum_5", "momentum_10", "volume_momentum_3",

    # trend
    "sma_5", "sma_10", "sma_20",
    "ema_5", "ema_10",
    "close_sma5_ratio", "close_sma10_ratio",
    "trend_sma", "trend_ema",

    # lag features
    "open_price_lag1", "high_price_lag1", "low_price_lag1", "close_price_lag1", "volume_lag1",
    "daily_return_lag1", "daily_range_lag1",

    "open_price_lag2", "high_price_lag2", "low_price_lag2", "close_price_lag2", "volume_lag2",
    "daily_return_lag2", "daily_range_lag2",

    "open_price_lag3", "high_price_lag3", "low_price_lag3", "close_price_lag3", "volume_lag3",
    "daily_return_lag3", "daily_range_lag3",

    "open_price_lag4", "high_price_lag4", "low_price_lag4", "close_price_lag4", "volume_lag4",
    "daily_return_lag4", "daily_range_lag4",

    "open_price_lag5", "high_price_lag5", "low_price_lag5", "close_price_lag5", "volume_lag5",
    "daily_return_lag5", "daily_range_lag5"
]]

# -----------------------------
# 9. Train/Test Split
# -----------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train_close = df["target_close"].iloc[:split_index]
y_test_close = df["target_close"].iloc[split_index:]

y_train_open = df["target_open"].iloc[:split_index]
y_test_open = df["target_open"].iloc[split_index:]

# -----------------------------
# 10. Preprocessing
# -----------------------------
numeric_features = [col for col in X.columns if col != "combined_headline"]
text_feature = "combined_headline"

preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words="english"
        ), text_feature),

        ("num", StandardScaler(), numeric_features)
    ]
)

# -----------------------------
# 11. Model
# -----------------------------
def create_model():
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight={0: 2.5, 1: 1}
        ))
    ])

# -----------------------------
# 12. Train Close Model
# -----------------------------
model_close = create_model()
model_close.fit(X_train, y_train_close)

y_pred_close = model_close.predict(X_test)

print("===== CLOSE-TO-CLOSE TARGET =====")
print("Accuracy:", accuracy_score(y_test_close, y_pred_close))
print(classification_report(y_test_close, y_pred_close))

# -----------------------------
# 13. Train Open Model
# -----------------------------
model_open = create_model()
model_open.fit(X_train, y_train_open)

y_pred_open = model_open.predict(X_test)

print("\n===== CLOSE-TO-OPEN TARGET =====")
print("Accuracy:", accuracy_score(y_test_open, y_pred_open))
print(classification_report(y_test_open, y_pred_open))

def get_feature_importance(model, preprocessor, numeric_features):
    # Get feature names
    tfidf = preprocessor.named_transformers_["text"]
    text_features = tfidf.get_feature_names_out()

    all_features = list(text_features) + numeric_features

    # Get coefficients
    clf = model.named_steps["classifier"]
    coefs = clf.coef_[0]

    importance_df = pd.DataFrame({
        "feature": all_features,
        "importance": coefs
    })

    importance_df["abs_importance"] = importance_df["importance"].abs()

    return importance_df.sort_values(by="abs_importance", ascending=False)

# Close model importance
importance_close = get_feature_importance(model_close, preprocessor, numeric_features)

print("\n===== TOP 20 IMPORTANT FEATURES (CLOSE MODEL) =====")
print(importance_close.head(20))

# Open model importance
importance_open = get_feature_importance(model_open, preprocessor, numeric_features)

print("\n===== TOP 20 IMPORTANT FEATURES (OPEN MODEL) =====")
print(importance_open.head(20))