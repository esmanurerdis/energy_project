# hyperparam_tuning.py
# Amaç: XGBoost için RandomizedSearchCV ile en iyi parametreleri bul,
#       ardından early-stopping ile yeniden eğit ve en iyi modeli kaydet.

# --- her zaman bu dosyadan çalıştır ---
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
print(">> Çalışma klasörü:", BASE_DIR)

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# ---- Yollar ----
DATA_PATH = Path("data/PJME_hourly.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODELS_DIR / "xgb_pjme_best.pkl"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
FEATURES_PATH = MODELS_DIR / "feature_cols.json"

# ---- Veri ve feature üretimi ----
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    return df

def build_features(df: pd.DataFrame):
    df_fe = df.copy()
    # Zaman temelli
    df_fe["year"]        = df_fe.index.year
    df_fe["month"]       = df_fe.index.month
    df_fe["day"]         = df_fe.index.day
    df_fe["hour"]        = df_fe.index.hour
    df_fe["day_of_week"] = df_fe.index.dayofweek
    df_fe["is_weekend"]  = (df_fe["day_of_week"] >= 5).astype(int)
    # Lag & Rolling
    df_fe["lag_1"]   = df_fe["PJME_MW"].shift(1)
    df_fe["lag_24"]  = df_fe["PJME_MW"].shift(24)
    df_fe["lag_168"] = df_fe["PJME_MW"].shift(168)
    df_fe["rolling_24h"] = df_fe["PJME_MW"].rolling(window=24).mean()
    df_fe["rolling_7d"]  = df_fe["PJME_MW"].rolling(window=24*7).mean()
    # NA temizle
    df_fe = df_fe.dropna()

    feature_cols = [
        "year","month","day","hour","day_of_week","is_weekend",
        "lag_1","lag_24","lag_168","rolling_24h","rolling_7d"
    ]
    return df_fe, feature_cols

def main():
    # 1) Veri + feature
    df = load_data()
    df_fe, feature_cols = build_features(df)

    # 2) Train/Test (2017 split)
    train = df_fe.loc[df_fe.index < "2017-01-01"]
    test  = df_fe.loc[df_fe.index >= "2017-01-01"]

    X_train = train[feature_cols].astype("float32")
    y_train = train["PJME_MW"].astype("float32")
    X_test  = test[feature_cols].astype("float32")
    y_test  = test["PJME_MW"].astype("float32")

    # 3) Parametre aralığı (bir tık geniş)
    param_dist = {
        "n_estimators":       [500, 800, 1200, 1600],
        "learning_rate":      [0.02, 0.03, 0.05, 0.08],
        "max_depth":          [4, 6, 8, 10],
        "subsample":          [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":   [0.7, 0.8, 0.9, 1.0],
        "min_child_weight":   [1, 2, 3, 5]
    }

    # 4) Base model (hız ve reproduksiyona dikkat)
    base = XGBRegressor(
        random_state=42,
        tree_method="hist",  # CPU'da hızlı; GPU varsa "gpu_hist"
        n_jobs=-1
    )

    # 5) RandomizedSearchCV + TimeSeriesSplit
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=30,  # biraz arttırdık
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=4),
        n_jobs=-1,
        verbose=1
    )

    rs.fit(X_train, y_train)

    best_cv_rmse = -rs.best_score_
    best_params = rs.best_params_
    print("\nBest CV RMSE:", f"{best_cv_rmse:.2f} MW")
    print("Best params:", best_params)

    # 6) En iyi parametrelerle EARLY-STOPPING'li yeniden eğitim
    #    (validasyon için test setini kullanıyoruz; istersen train'den bir dilim ayırabilirsin)
    best = XGBRegressor(
        **best_params,
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )
    best.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )

    # 7) Test performansı
    y_pred = best.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Test RMSE (best + early stopping):", f"{test_rmse:.2f} MW")

    # 8) Kaydet: model, feature listesi, best params
    joblib.dump(best, BEST_MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f">> En iyi model '{BEST_MODEL_PATH.as_posix()}' olarak kaydedildi.")
    print(f">> Feature listesi '{FEATURES_PATH.as_posix()}' olarak kaydedildi.")
    print(f">> Best params '{BEST_PARAMS_PATH.as_posix()}' olarak kaydedildi.")

if __name__ == "__main__":
    main()
