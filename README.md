## 📊 PJME Hourly Energy Forecasting Project

•This project is a machine learning application to predict future energy demand using hourly energy consumption data from the PJM region in the USA.
•It covers data analysis, feature engineering, model training, hyperparameter optimization, and reporting (in Turkish).

---

## 🚀 Project Features

Data Preprocessing & EDA: Analysis of missing data, time series statistics, and visualization

Feature Engineering: Time-based, lag, and rolling features

Modeling: Linear Regression and XGBoost

Model Validation: K-Fold cross-validation

Hyperparameter Optimization: GridSearchCV for XGBoost parameter tuning

Prediction: Hourly energy demand forecast for the next 7 days

Reporting: Report generation with visualizations

---

# 📂 Project Structure
'''python
energy_project/
├── data/
│   └── PJME_hourly.csv           # Raw dataset
├── models/
│   └── xgb_pjme_best.pkl         # Trained XGBoost model
├── outputs/
│   ├── predictions.csv           # Forecast results
│   ├── report_actual_vs_pred_*.png
│   ├── report_error_time_*.png
│   ├── report_error_hist_*.png
├── main.py                       # Model training & prediction
├── model_validation.py           # Cross-validation
├── hyperparam_tuning.py          # Hyperparameter tuning
├── report.py                     # Report generation
├── requirements.txt              # Dependencies
└── README.md                     # Project description
'''

---


## 📊 Libraries Used

• Python 3.9+
• pandas
• numpy
• matplotlib
• scikit-learn
• xgboost
• joblib

---

## 🔧 Setup

# Install dependencies
'''python
pip install -r requirements.txt
'''

# Train the model and generate 7-day forecast
'''python
python main.py
'''

# Validate model with cross-validation
'''python
python model_validation.py
'''

# Hyperparameter tuning (optional)
'''python
python hyperparam_tuning.py
'''

# Generate report
'''python
python report.py
'''
---

## 📊 Dataset

• Source: Kaggle - PJME Hourly Energy Consumption

• Coverage: Hourly energy consumption (MW) in PJM from 2002 to 2018

---

## 💡 Notes

• The project is suitable for both Data Science and Machine Learning Engineer portfolios.

• Different models (LightGBM, Prophet, etc.) can be tried for higher performance.

• Forecasts can be useful for energy planning and demand management.

---

## 🏷 License

This project is licensed under the MIT License.
Feel free to use and improve it as you wish.

---


## 🇹🇷 Türkçe Summary

Bu proje, ABD PJM bölgesine ait saatlik enerji tüketimi verilerini kullanarak gelecek enerji talebini tahmin eden bir makine öğrenmesi uygulamasıdır.

Veri analizi, özellik mühendisliği, model eğitimi, hiperparametre optimizasyonu ve raporlama adımlarını kapsamaktadır.

Modelleme Linear Regression ve XGBoost ile yapılmıştır.

Cross-validation ve GridSearchCV kullanılarak model doğrulaması ve parametre optimizasyonu gerçekleştirilmiştir.

Tahmin sonuçları ve raporlar görselleştirilmiştir.
