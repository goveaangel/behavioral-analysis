# xgb_markov_model.py
"""
Entrena modelos XGBoost para predecir:
- engagement_estacionario
- score_neto (prob_mejorar - prob_empeorar)

Usa el mismo CSV agregado de combinaciones de Markov.
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from xgboost import XGBRegressor  # pip install xgboost


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV_DIR = os.path.join(BASE_DIR, "..", "outputs", "csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "outputs", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

SUMMARY_FILE = os.path.join(
    OUTPUT_CSV_DIR, "summary_ProjectTag_Seniority_Position_Location.csv"
)


def main():
    print("Leyendo:", SUMMARY_FILE)
    df = pd.read_csv(SUMMARY_FILE)

    required_cols = [
        "Project Tag",
        "Seniority",
        "Position",
        "Location",
        "prob_mejorar",
        "prob_empeorar",
        "engagement_estacionario",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV agregado: {missing}")

    df["score_neto"] = df["prob_mejorar"] - df["prob_empeorar"]

    # Features
    X_cat = df[["Project Tag", "Seniority", "Position", "Location"]].astype(str)
    X = pd.get_dummies(X_cat, drop_first=True)

    # Guardar columnas de features
    feature_cols_path = os.path.join(MODEL_DIR, "xgb_feature_columns.json")
    with open(feature_cols_path, "w") as f:
        json.dump(list(X.columns), f)
    print("Columnas XGBoost guardadas en:", feature_cols_path)

    # Targets
    y_eng = df["engagement_estacionario"].values
    y_net = df["score_neto"].values

    # Train/test split
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        X, y_eng, test_size=0.3, random_state=42
    )
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
        X, y_net, test_size=0.3, random_state=42
    )

    # ------------------------
    # XGB para engagement_estacionario
    # ------------------------
    xgb_eng = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    xgb_eng.fit(X_train_e, y_train_e)
    r2_train_e = xgb_eng.score(X_train_e, y_train_e)
    r2_test_e = xgb_eng.score(X_test_e, y_test_e)

    print("\nXGBoost – engagement_estacionario")
    print("R^2 train:", r2_train_e)
    print("R^2 test :", r2_test_e)

    # Importancias
    importances_e = pd.DataFrame(
        {"feature": X.columns, "importance": xgb_eng.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances_e_path = os.path.join(
        OUTPUT_CSV_DIR, "xgb_feature_importances_engagement_estacionario.csv"
    )
    importances_e.to_csv(importances_e_path, index=False)
    print("Importancias XGB engagement guardadas en:", importances_e_path)

    xgb_eng_path = os.path.join(MODEL_DIR, "xgb_engagement_estacionario.pkl")
    joblib.dump(xgb_eng, xgb_eng_path)
    print("Modelo XGB engagement guardado en:", xgb_eng_path)

    # ------------------------
    # XGB para score_neto
    # ------------------------
    xgb_net = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    xgb_net.fit(X_train_n, y_train_n)
    r2_train_n = xgb_net.score(X_train_n, y_train_n)
    r2_test_n = xgb_net.score(X_test_n, y_test_n)

    print("\nXGBoost – score_neto")
    print("R^2 train:", r2_train_n)
    print("R^2 test :", r2_test_n)

    importances_n = pd.DataFrame(
        {"feature": X.columns, "importance": xgb_net.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances_n_path = os.path.join(
        OUTPUT_CSV_DIR, "xgb_feature_importances_score_neto.csv"
    )
    importances_n.to_csv(importances_n_path, index=False)
    print("Importancias XGB score_neto guardadas en:", importances_n_path)

    xgb_net_path = os.path.join(MODEL_DIR, "xgb_score_neto.pkl")
    joblib.dump(xgb_net, xgb_net_path)
    print("Modelo XGB score_neto guardado en:", xgb_net_path)


if __name__ == "__main__":
    main()