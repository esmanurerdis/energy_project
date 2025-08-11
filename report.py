# report.py — Türkçe raporlar (metrikler + görseller + CSV) — cyclical uyumlu & %100 hizalama

# --- her zaman bu dosyadan çalıştır ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
print(">> Çalışma klasörü:", BASE_DIR)

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Türkçe karakterler için font
plt.rcParams["font.family"] = "DejaVu Sans"

# Yollar
DATA_PATH   = Path("data/PJME_hourly.csv")
MODELS_DIR  = Path("models")
BEST_MODEL  = MODELS_DIR / "xgb_pjme_best.pkl"
FALLBACK    = MODELS_DIR / "xgb_pjme.pkl"
FEATS_PATH  = MODELS_DIR / "feature_cols.json"
OUT_DIR     = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# ---------- yardımcılar ----------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    return df

def build_features(df: pd.DataFrame):
    """Eğitim tarafıyla birebir aynı feature setini üretir (cyclical dahil)."""
    df_fe = df.copy()

    # Zaman temelli
    df_fe["year"]        = df_fe.index.year
    df_fe["month"]       = df_fe.index.month
    df_fe["day"]         = df_fe.index.day
    df_fe["hour"]        = df_fe.index.hour
    df_fe["day_of_week"] = df_fe.index.dayofweek
    df_fe["is_weekend"]  = (df_fe["day_of_week"] >= 5).astype(int)

    # Döngüsel özellikler (cyclical encoding)
    df_fe["hour_sin"] = np.sin(2 * np.pi * df_fe["hour"] / 24)
    df_fe["hour_cos"] = np.cos(2 * np.pi * df_fe["hour"] / 24)
    df_fe["dow_sin"]  = np.sin(2 * np.pi * df_fe["day_of_week"] / 7)
    df_fe["dow_cos"]  = np.cos(2 * np.pi * df_fe["day_of_week"] / 7)

    # Lag & Rolling
    df_fe["lag_1"]    = df_fe["PJME_MW"].shift(1)
    df_fe["lag_24"]   = df_fe["PJME_MW"].shift(24)
    df_fe["lag_168"]  = df_fe["PJME_MW"].shift(168)
    df_fe["rolling_24h"] = df_fe["PJME_MW"].rolling(window=24).mean()
    df_fe["rolling_7d"]  = df_fe["PJME_MW"].rolling(window=24*7).mean()

    # NA temizle
    df_fe = df_fe.dropna()
    return df_fe

def try_load_feature_list_from_json(feats_path=FEATS_PATH):
    if feats_path.exists():
        try:
            with open(feats_path, "r", encoding="utf-8") as f:
                cols = json.load(f)
            if isinstance(cols, list) and len(cols) > 0:
                print(">> feature_cols.json yüklendi.")
                return cols
        except Exception as e:
            print(">> Uyarı: feature_cols.json okunamadı:", e)
    return None

def try_load_feature_list_from_model(model):
    """XGBoost booster içinden eğitimde kullanılan feature adlarını yakala."""
    try:
        booster = model.get_booster()
        booster_cols = booster.feature_names
        if booster_cols and len(booster_cols) > 0:
            print(">> Model içindeki feature isimleri kullanılacak.")
            return booster_cols
    except Exception as e:
        print(">> Uyarı: Booster feature isimleri alınamadı:", e)
    return None

def load_model_and_features():
    # Model: önce best, yoksa fallback
    model_path = BEST_MODEL if BEST_MODEL.exists() else FALLBACK
    if not model_path.exists():
        raise FileNotFoundError("Model dosyası bulunamadı. 'models/xgb_pjme_best.pkl' veya 'models/xgb_pjme.pkl' gerekli.")
    model = joblib.load(model_path)
    print(">> Yüklenen model:", model_path.as_posix())

    # Feature list kaynağı öncelik sırası:
    # 1) model içindeki booster feature_names
    # 2) feature_cols.json
    feature_cols = try_load_feature_list_from_model(model)
    if feature_cols is None:
        feature_cols = try_load_feature_list_from_json()
    if feature_cols is None:
        raise FileNotFoundError("Feature listesi bulunamadı. Lütfen eğitim sırasında feature kolonlarını kaydedin.")
    return model, feature_cols

def strict_align_features(frame: pd.DataFrame, feature_list):
    """Test çerçevesini modelin beklediği feature sırasına %100 uydurur.
       Eksik kolonları 0 ile oluşturur, fazla kolonları atar, sırayı sabitler."""
    for col in feature_list:
        if col not in frame.columns:
            frame[col] = 0.0  # güvenli default
    aligned = frame[feature_list].copy()
    aligned = aligned.astype("float32").fillna(0.0)
    return aligned

# ---------- ana akış ----------
df = load_data()
df_fe = build_features(df)  # <-- feature'ları üret

# Train/Test (2017 kesimi)
train = df_fe.loc[df_fe.index < "2017-01-01"]
test  = df_fe.loc[df_fe.index >= "2017-01-01"]

model, feature_cols = load_model_and_features()

# Bilgilendirici log
print(">> Toplam feature sayısı (modelin beklediği):", len(feature_cols))

# X, y
X_test = strict_align_features(test, feature_cols)
y_test = test["PJME_MW"].astype("float32").copy()

# Tahmin
y_pred = model.predict(X_test)

# ---------- metrikler ----------
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae  = float(mean_absolute_error(y_test, y_pred))
r2   = float(r2_score(y_test, y_pred))
mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

print(f"\n📊 Test Metrikleri | RMSE: {rmse:.2f}  MAE: {mae:.2f}  MAPE: {mape:.2f}%  R²: {r2:.3f}")

# ---------- çıktı dosya adlarına timestamp ----------
ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

# CSV: tahmin tablosu
pred_df = pd.DataFrame({
    "Datetime": y_test.index,
    "Gerçek": y_test.values,
    "Tahmin": y_pred,
    "Hata": y_test.values - y_pred
}).set_index("Datetime")

pred_path = OUT_DIR / f"test_tahminleri_{ts}.csv"
pred_df.to_csv(pred_path, encoding="utf-8")
print(">> Tahmin CSV kaydedildi:", pred_path.as_posix())

# ---------- görseller ----------
date_span = f"{y_test.index.min().date()} → {y_test.index.max().date()}"

# 1) Gerçek vs Tahmin (zaman serisi)
plt.figure(figsize=(12,4))
plt.plot(y_test.index, y_test, label="Gerçek", alpha=0.7)
plt.plot(y_test.index, y_pred, label="Tahmin", alpha=0.8)
plt.title(f"Gerçek vs Tahmin (XGBoost) | {date_span} | RMSE={rmse:.0f}, MAE={mae:.0f}, MAPE={mape:.1f}%")
plt.xlabel("Tarih"); plt.ylabel("Enerji Tüketimi (MW)")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / f"rapor_gercek_vs_tahmin_{ts}.png"); plt.show()

# 2) Hatalar zaman içinde
plt.figure(figsize=(12,3))
plt.plot(pred_df.index, pred_df["Hata"], alpha=0.7)
plt.axhline(0, color="black", linewidth=1)
plt.title("Hatalar (Zaman Boyunca)")
plt.xlabel("Tarih"); plt.ylabel("Hata (MW)")
plt.tight_layout()
plt.savefig(OUT_DIR / f"rapor_hata_zaman_{ts}.png"); plt.show()

# 3) Hata dağılımı (histogram)
plt.figure(figsize=(6,4))
plt.hist(pred_df["Hata"], bins=50, alpha=0.85)
plt.title("Hata Dağılımı (Histogram)")
plt.xlabel("Hata (MW)"); plt.ylabel("Frekans")
plt.tight_layout()
plt.savefig(OUT_DIR / f"rapor_hata_hist_{ts}.png"); plt.show()

# 4) Scatter: Gerçek vs Tahmin (45°)
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, s=6, alpha=0.5)
lims_min = float(min(y_test.min(), y_pred.min()))
lims_max = float(max(y_test.max(), y_pred.max()))
plt.plot([lims_min, lims_max], [lims_min, lims_max], "r--", linewidth=1)  # ideal çizgi
plt.xlim(lims_min, lims_max); plt.ylim(lims_min, lims_max)
plt.xlabel("Gerçek (MW)"); plt.ylabel("Tahmin (MW)")
plt.title("Gerçek vs Tahmin (45° Çizgisi)")
plt.tight_layout()
plt.savefig(OUT_DIR / f"rapor_scatter_{ts}.png"); plt.show()

# 5) Saat bazında MAE
hour_mae = pred_df.assign(Saat=pred_df.index.hour)["Hata"].abs().groupby(pred_df.index.hour).mean()
plt.figure(figsize=(8,4))
plt.bar(hour_mae.index, hour_mae.values)
plt.title("Saat Bazında Ortalama Mutlak Hata (MAE)")
plt.xlabel("Saat"); plt.ylabel("MAE (MW)")
plt.tight_layout()
plt.savefig(OUT_DIR / f"rapor_mae_saat_{ts}.png"); plt.show()

# 6) Haftanın günü bazında MAE
dow_map = {0:"Pzt",1:"Sal",2:"Çar",3:"Per",4:"Cum",5:"Cts",6:"Paz"}
dow_mae = pred_df.assign(Gun=pred_df.index.dayofweek)["Hata"].abs().groupby(pred_df.index.dayofweek).mean()
dow_mae.index = dow_mae.index.map(dow_map)
plt.figure(figsize=(8,4))
plt.bar(dow_mae.index, dow_mae.values)
plt.title("Haftanın Günü Bazında Ortalama Mutlak Hata (MAE)")
plt.xlabel("Gün"); plt.ylabel("MAE (MW)")
plt.tight_layout()
plt.savefig(OUT_DIR / f"rapor_mae_gun_{ts}.png"); plt.show()

print("\n>> Rapor hazır! outputs/ klasörüne bak:")
print(f"   - {pred_path.name}")
print(f"   - rapor_gercek_vs_tahmin_{ts}.png")
print(f"   - rapor_hata_zaman_{ts}.png")
print(f"   - rapor_hata_hist_{ts}.png")
print(f"   - rapor_scatter_{ts}.png")
print(f"   - rapor_mae_saat_{ts}.png")
print(f"   - rapor_mae_gun_{ts}.png")


