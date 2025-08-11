# main.py — PJME Forecast (EDA + FE + Linear + XGBoost + Importance + Forecast + Save)
# --- her zaman bu dosyadan çalıştır  ---
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
print(">> Çalışma klasörü:", BASE_DIR)

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from xgboost import XGBRegressor

# -------------------------------
# 0) Yol ve klasörler
# -------------------------------
DATA_PATH = Path("data/PJME_hourly.csv")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------------
# 1) Veri yükleme + EDA görselleri
# -------------------------------
df = pd.read_csv(DATA_PATH)
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime")

print("\nEksik veri sayısı:\n", df.isna().sum())

daily = df["PJME_MW"].resample("D").mean()
monthly = df["PJME_MW"].resample("ME").mean()
rolling_30d = daily.rolling(window=30).mean()

plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1); plt.plot(daily);   plt.title("Günlük Ortalama Enerji Tüketimi (MW)")
plt.subplot(3, 1, 2); plt.plot(monthly); plt.title("Aylık Ortalama Enerji Tüketimi (MW)")
plt.subplot(3, 1, 3); plt.plot(rolling_30d); plt.title("30 Günlük Hareketli Ortalama (MW)")
plt.tight_layout(); plt.savefig("outputs/all_in_one.png"); plt.show()
print("\n>> Tüm grafikler tek pencerede 'outputs/all_in_one.png' olarak kaydedildi.")

# -------------------------------
# 2) Feature Engineering
# -------------------------------
df_features = df.copy()
# Zaman özellikleri
df_features["year"]        = df_features.index.year
df_features["month"]       = df_features.index.month
df_features["day"]         = df_features.index.day
df_features["hour"]        = df_features.index.hour
df_features["day_of_week"] = df_features.index.dayofweek
df_features["is_weekend"]  = (df_features["day_of_week"] >= 5).astype(int)

# 🔹 Döngüsel (cyclical) kodlama
df_features["hour_sin"] = np.sin(2 * np.pi * df_features["hour"] / 24)
df_features["hour_cos"] = np.cos(2 * np.pi * df_features["hour"] / 24)
df_features["dow_sin"]  = np.sin(2 * np.pi * df_features["day_of_week"] / 7)
df_features["dow_cos"]  = np.cos(2 * np.pi * df_features["day_of_week"] / 7)

# Lag & Rolling
df_features["lag_1"]   = df_features["PJME_MW"].shift(1)
df_features["lag_24"]  = df_features["PJME_MW"].shift(24)
df_features["lag_168"] = df_features["PJME_MW"].shift(168)
df_features["rolling_24h"] = df_features["PJME_MW"].rolling(window=24).mean()
df_features["rolling_7d"]  = df_features["PJME_MW"].rolling(window=24*7).mean()

df_features = df_features.dropna()
print("\nYeni veri seti boyutu:", df_features.shape)
print(df_features.head())

# -------------------------------
# 3) Train/Test split
# -------------------------------
train = df_features.loc[df_features.index < "2017-01-01"]
test  = df_features.loc[df_features.index >= "2017-01-01"]

feature_cols = [
    "year","month","day","hour","day_of_week","is_weekend",
    "hour_sin","hour_cos","dow_sin","dow_cos",        # <-- yeni
    "lag_1","lag_24","lag_168","rolling_24h","rolling_7d"
]
X_train, y_train = train[feature_cols], train["PJME_MW"]
X_test,  y_test  = test[feature_cols],  test["PJME_MW"]

# -------------------------------
# 4) Baseline: Linear Regression
# -------------------------------
lin = LinearRegression().fit(X_train, y_train)
y_pred_lin = lin.predict(X_test)
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
print(f"\n📊 Linear Regression RMSE: {rmse_lin:.2f} MW")

plt.figure(figsize=(12,4))
plt.plot(y_test.index, y_test, label="Gerçek", alpha=0.7)
plt.plot(y_test.index, y_pred_lin, label="Tahmin (Linear)", alpha=0.7)
plt.legend(); plt.title("Gerçek vs Tahmin (Linear Regression)")
plt.tight_layout(); plt.show()

# -------------------------------
# 5) XGBoost Modeli
# -------------------------------
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"\n📊 XGBoost RMSE: {rmse_xgb:.2f} MW")

plt.figure(figsize=(12,4))
plt.plot(y_test.index, y_test, label="Gerçek", alpha=0.7)
plt.plot(y_test.index, y_pred_xgb, label="Tahmin (XGBoost)", alpha=0.7)
plt.legend(); plt.title("Gerçek vs Tahmin (XGBoost)")
plt.tight_layout(); plt.show()

# -------------------------------
# 6) Feature Importance
# -------------------------------
fi = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(8,6))
plt.barh(fi.index, fi.values)
plt.title("XGBoost Feature Importance")
plt.tight_layout(); plt.savefig("outputs/feature_importance.png"); plt.show()
print(">> Feature importance 'outputs/feature_importance.png' olarak kaydedildi.")

# -------------------------------
# 7) Gelecek 7 gün (iteratif) tahmini
# -------------------------------
horizon = 24 * 7  # 7 gün
last_ts = df.index.max()
future_index = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon, freq="h")

# son 14 gün tampon (lag/rolling için)
hist = deque(df["PJME_MW"].tail(24*14).tolist(), maxlen=24*14)
preds = []

for t in future_index:
    year, month, day, hour = t.year, t.month, t.day, t.hour
    dow = t.dayofweek
    is_weekend = int(dow >= 5)

    # cyclical
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin  = np.sin(2 * np.pi * dow / 7)
    dow_cos  = np.cos(2 * np.pi * dow / 7)

    # lag/rolling
    lag_1   = hist[-1]
    lag_24  = hist[-24]
    lag_168 = hist[-168]
    rolling_24h = float(np.mean(list(hist)[-24:]))
    rolling_7d  = float(np.mean(list(hist)[-168:]))

    X_new = pd.DataFrame([[
        year, month, day, hour, dow, is_weekend,
        hour_sin, hour_cos, dow_sin, dow_cos,
        lag_1, lag_24, lag_168, rolling_24h, rolling_7d
    ]], columns=feature_cols)

    y_hat = xgb.predict(X_new)[0]
    preds.append(y_hat)
    hist.append(y_hat)

future_forecast = pd.Series(preds, index=future_index, name="Forecast")

# grafiği daha okunur pencere ile çiz
start_win = last_ts - pd.Timedelta(days=14)
fig = plt.figure(figsize=(12,4))
ax = plt.gca()
ax.plot(df["PJME_MW"].loc[start_win:last_ts].index,
        df["PJME_MW"].loc[start_win:last_ts].values,
        label="Gerçek (son 14 gün)", alpha=0.7)
ax.plot(future_forecast.index, future_forecast.values,
        label="Tahmin (gelecek 7 gün)", alpha=0.9)
ax.set_xlim(start_win, future_forecast.index[-1])
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
plt.title("XGBoost ile Gelecek 7 Gün Tahmin")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/forecast_7d.png"); plt.show()
print(">> Gelecek 7 gün tahmin grafiği 'outputs/forecast_7d.png' olarak kaydedildi.")

# -------------------------------
# 8) Modeli kaydet
# -------------------------------
joblib.dump(xgb, "models/xgb_pjme.pkl")
with open("models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)
print(">> Model 'models/xgb_pjme.pkl' ve feature listesi kaydedildi.")
