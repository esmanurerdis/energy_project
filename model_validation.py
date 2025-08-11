# --- her zaman bu dosyadan çalıştır  ---
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
print(">> Çalışma klasörü:", BASE_DIR)
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

DATA_PATH = Path("data/PJME_hourly.csv")

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    return df

def build_features(df: pd.DataFrame):
    df_fe = df.copy()
    # zaman özellikleri
    df_fe["year"]        = df_fe.index.year
    df_fe["month"]       = df_fe.index.month
    df_fe["day"]         = df_fe.index.day
    df_fe["hour"]        = df_fe.index.hour
    df_fe["day_of_week"] = df_fe.index.dayofweek
    df_fe["is_weekend"]  = (df_fe["day_of_week"] >= 5).astype(int)
    # lag & rolling
    df_fe["lag_1"]   = df_fe["PJME_MW"].shift(1)
    df_fe["lag_24"]  = df_fe["PJME_MW"].shift(24)
    df_fe["lag_168"] = df_fe["PJME_MW"].shift(168)
    df_fe["rolling_24h"] = df_fe["PJME_MW"].rolling(window=24).mean()
    df_fe["rolling_7d"]  = df_fe["PJME_MW"].rolling(window=24*7).mean()
    df_fe = df_fe.dropna()
    feature_cols = [
        "year","month","day","hour","day_of_week","is_weekend",
        "lag_1","lag_24","lag_168","rolling_24h","rolling_7d"
    ]
    X = df_fe[feature_cols].values
    y = df_fe["PJME_MW"].values
    return X, y, feature_cols

def main():
    df = load_data()
    X, y, feature_cols = build_features(df)

    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []

    for i, (tr, val) in enumerate(tscv.split(X), start=1):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]

        model = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        rmses.append(rmse)
        print(f"Fold {i} RMSE: {rmse:.2f} MW")

    print("-"*40)
    print("CV RMSE ortalama:", f"{np.mean(rmses):.2f} MW")
    print("CV RMSE std     :", f"{np.std(rmses):.2f} MW")
    print("Foldlar:", [round(x, 1) for x in rmses])

if __name__ == "__main__":
    main()
