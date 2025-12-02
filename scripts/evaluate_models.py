# evaluate_models.py

import os
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(BASE, "..")
OUT = os.path.join(PROJECT, "outputs")

SUMMARY = os.path.join(
    OUT, "csv", "summary_ProjectTag_Seniority_Position_Location.csv"
)
MODEL_DIR = os.path.join(OUT, "models")
EVAL_DIR = os.path.join(OUT, "figs", "evaluation_plots")

os.makedirs(EVAL_DIR, exist_ok=True)


def load_models_and_features():
    """Carga modelos RF/XGB y columnas de features usadas en entrenamiento."""
    rf_eng = joblib.load(
        os.path.join(MODEL_DIR, "rf_engagement_estacionario.pkl")
    )
    xgb_eng = joblib.load(
        os.path.join(MODEL_DIR, "xgb_engagement_estacionario.pkl")
    )

    rf_net = joblib.load(
        os.path.join(MODEL_DIR, "rf_score_neto.pkl")
    )
    xgb_net = joblib.load(
        os.path.join(MODEL_DIR, "xgb_score_neto.pkl")
    )

    with open(os.path.join(MODEL_DIR, "rf_feature_columns.json")) as f:
        feature_cols = json.load(f)

    return rf_eng, xgb_eng, rf_net, xgb_net, feature_cols


def evaluate_model(name, model, X, y_true):
    """Calcula RMSE, MAE y R² para un modelo dado."""
    y_pred = model.predict(X)

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "model": name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def main():
    print("Leyendo summary:", SUMMARY)
    df = pd.read_csv(SUMMARY)

    # Asegurar que score_neto exista (si no, lo creamos)
    if "score_neto" not in df.columns:
        if {"prob_mejorar", "prob_empeorar"}.issubset(df.columns):
            df["score_neto"] = df["prob_mejorar"] - df["prob_empeorar"]
            print("Columna 'score_neto' creada a partir de prob_mejorar - prob_empeorar.")
        else:
            raise ValueError(
                "No existe 'score_neto' ni (prob_mejorar, prob_empeorar) en el summary."
            )

    rf_eng, xgb_eng, rf_net, xgb_net, feature_cols = load_models_and_features()

    # One-hot encoding igual que en el entrenamiento
    X_cat = df[["Project Tag", "Seniority", "Position", "Location"]].astype(str)
    X = pd.get_dummies(X_cat, drop_first=True)
    X = X.reindex(columns=feature_cols, fill_value=0)

    y_eng = df["engagement_estacionario"].values
    y_net = df["score_neto"].values

    results = []

    results.append(evaluate_model("RF_engagement", rf_eng, X, y_eng))
    results.append(evaluate_model("XGB_engagement", xgb_eng, X, y_eng))
    results.append(evaluate_model("RF_score_neto", rf_net, X, y_net))
    results.append(evaluate_model("XGB_score_neto", xgb_net, X, y_net))

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(OUT, "csv", "model_evaluation_report.csv")
    res_df.to_csv(out_csv, index=False)

    print("\n✅ Evaluación terminada. Resultados guardados en:")
    print("   ", out_csv)
    print(res_df)


if __name__ == "__main__":
    main()