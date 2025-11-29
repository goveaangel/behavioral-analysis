import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")

OUTPUT_CSV_DIR = os.path.join(PROJECT_DIR, "outputs", "csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "outputs", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SUMMARY_FILE = os.path.join(
    OUTPUT_CSV_DIR, "summary_ProjectTag_Seniority_Position_Location.csv"
)

CAT_COLS = ["Project Tag", "Seniority", "Position", "Location"]


def main():
    print("Leyendo:", SUMMARY_FILE)
    df = pd.read_csv(SUMMARY_FILE)

    required_cols = CAT_COLS + [
        "prob_mejorar",
        "prob_empeorar",
        "engagement_estacionario",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV agregado: {missing}")

    # Normalizar texto
    for col in CAT_COLS:
        df[col] = df[col].astype(str).str.strip()

    df["score_neto"] = df["prob_mejorar"] - df["prob_empeorar"]

    X = df[CAT_COLS].copy()
    y_eng = df["engagement_estacionario"].values
    y_net = df["score_neto"].values

    # Preprocesador: one-hot para categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ],
        remainder="drop",
    )

    # ========= Modelo 1: engagement_estacionario =========
    pipe_eng = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("rf", RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_eng, test_size=0.3, random_state=42
    )

    pipe_eng.fit(X_train, y_train)
    r2_train = pipe_eng.score(X_train, y_train)
    r2_test = pipe_eng.score(X_test, y_test)

    print("\nRandomForest PIPELINE – engagement_estacionario")
    print("R^2 train:", r2_train)
    print("R^2 test :", r2_test)

    eng_path = os.path.join(MODEL_DIR, "rf_engagement_pipeline.pkl")
    joblib.dump(pipe_eng, eng_path)
    print("Pipeline RF engagement guardado en:", eng_path)

    # ========= Modelo 2: score_neto =========
    pipe_net = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("rf", RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )),
        ]
    )

    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
        X, df["score_neto"].values, test_size=0.3, random_state=42
    )

    pipe_net.fit(X_train_n, y_train_n)
    r2_train_n = pipe_net.score(X_train_n, y_train_n)
    r2_test_n = pipe_net.score(X_test_n, y_test_n)

    print("\nRandomForest PIPELINE – score_neto")
    print("R^2 train:", r2_train_n)
    print("R^2 test :", r2_test_n)

    net_path = os.path.join(MODEL_DIR, "rf_score_neto_pipeline.pkl")
    joblib.dump(pipe_net, net_path)
    print("Pipeline RF score_neto guardado en:", net_path)


if __name__ == "__main__":
    main()