# rf_markov_model.py
"""
Entrena modelos Random Forest sobre las métricas agregadas de cadenas de Markov.

- Lee summary_ProjectTag_Seniority_Position_Location.csv
- Define features categóricas (Project Tag, Seniority, Position, Location)
- Entrena:
    * rf_eng: predice engagement_estacionario
    * rf_net: predice score_neto = prob_mejorar - prob_empeorar
- Guarda:
    * modelos .pkl
    * importancias de features como CSV
    * lista de columnas de features (para usar en Streamlit u otros scripts)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

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

    # 🔴 IMPORTANTE: normalizar texto IGUAL que en la app
    for col in ["Project Tag", "Seniority", "Position", "Location"]:
        df[col] = df[col].astype(str).str.strip()

    df["score_neto"] = df["prob_mejorar"] - df["prob_empeorar"]

    # Features categóricas
    X_cat = df[["Project Tag", "Seniority", "Position", "Location"]].astype(str)
    X = pd.get_dummies(X_cat, drop_first=True)

    # Guardar columnas de features
    feature_cols_path = os.path.join(MODEL_DIR, "rf_feature_columns.json")
    with open(feature_cols_path, "w") as f:
        json.dump(list(X.columns), f)
    print("Columnas de features guardadas en:", feature_cols_path)

    # Targets
    y_eng = df["engagement_estacionario"].values
    y_net = df["score_neto"].values

    # ======================
    # 3. Train/test split
    # ======================
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        X, y_eng, test_size=0.3, random_state=42
    )

    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
        X, y_net, test_size=0.3, random_state=42
    )

    # ======================
    # 4. Random Forest para engagement_estacionario
    # ======================
    rf_eng = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    rf_eng.fit(X_train_e, y_train_e)

    r2_train_e = rf_eng.score(X_train_e, y_train_e)
    r2_test_e = rf_eng.score(X_test_e, y_test_e)

    print("\nRandomForest – engagement_estacionario")
    print("R^2 train:", r2_train_e)
    print("R^2 test :", r2_test_e)

    # Importancias de features
    importances_e = pd.DataFrame(
        {"feature": X.columns, "importance": rf_eng.feature_importances_}
    ).sort_values("importance", ascending=False)

    importances_e_path = os.path.join(
        OUTPUT_CSV_DIR, "rf_feature_importances_engagement_estacionario.csv"
    )
    importances_e.to_csv(importances_e_path, index=False)
    print("Importancias guardadas en:", importances_e_path)

    # Guardar modelo
    rf_eng_path = os.path.join(MODEL_DIR, "rf_engagement_estacionario.pkl")
    joblib.dump(rf_eng, rf_eng_path)
    print("Modelo RF (engagement_estacionario) guardado en:", rf_eng_path)

    # ======================
    # 5. Random Forest para score_neto
    # ======================
    rf_net = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    rf_net.fit(X_train_n, y_train_n)

    r2_train_n = rf_net.score(X_train_n, y_train_n)
    r2_test_n = rf_net.score(X_test_n, y_test_n)

    print("\nRandomForest – score_neto (prob_mejorar - prob_empeorar)")
    print("R^2 train:", r2_train_n)
    print("R^2 test :", r2_test_n)

    importances_n = pd.DataFrame(
        {"feature": X.columns, "importance": rf_net.feature_importances_}
    ).sort_values("importance", ascending=False)

    importances_n_path = os.path.join(
        OUTPUT_CSV_DIR, "rf_feature_importances_score_neto.csv"
    )
    importances_n.to_csv(importances_n_path, index=False)
    print("Importancias guardadas en:", importances_n_path)

    rf_net_path = os.path.join(MODEL_DIR, "rf_score_neto.pkl")
    joblib.dump(rf_net, rf_net_path)
    print("Modelo RF (score_neto) guardado en:", rf_net_path)


if __name__ == "__main__":
    main()