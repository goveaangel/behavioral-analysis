# run_markov_analysis.py
"""
Script principal para:
- Cargar datos de Globant.
- Calcular métricas de cadenas de Markov globales y por combinaciones.
- Entrenar modelos sencillos de regresión.
- Guardar resultados en CSV y figuras PNG para usar en reportes / siguientes pasos.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------
# Config matplotlib / seaborn
# -------------------
sns.set(style="whitegrid")

# -------------------
# Rutas
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # scripts/
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data_globant_cleaned.csv")
OUTPUT_CSV_DIR = os.path.join(BASE_DIR, "..", "outputs", "csv")
OUTPUT_FIG_DIR = os.path.join(BASE_DIR, "..", "outputs", "figs")

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# -------------------
# Importar módulo de funciones
# -------------------
sys.path.append(BASE_DIR)  # para que encuentre markov_engagement.py en scripts/

from markov_engagement import (  # noqa: E402
    build_transition_matrix,
    estacionaria,
    prob_mejorar,
    prob_empeorar,
    engagement_estacionario,
    markov_summary_by_group,
    DEFAULT_STATES,
)


def main():
    # =========================
    # 1. Cargar y preparar datos
    # =========================
    print("Cargando datos desde:", DATA_PATH)
    data = pd.read_csv(DATA_PATH)

    # Fecha
    data["Date"] = pd.to_datetime(data["Date"])

    # Filtrar estados válidos
    data = data[data["Engagement Group"] > 0].copy()

    # Ordenar por persona y tiempo
    data = data.sort_values(["Name", "Date"])

    print("Registros después de filtrar Engagement Group > 0:", data.shape[0])

    states = DEFAULT_STATES

    # =========================
    # 2. Análisis global
    # =========================
    print("\nCalculando matriz de transición global...")
    P_global = build_transition_matrix(data, col_state="Engagement Group", states=states)
    pi_global = estacionaria(P_global)
    p_up_global = prob_mejorar(P_global, states=states)
    p_down_global = prob_empeorar(P_global, states=states)
    E_inf_global = engagement_estacionario(pi_global, states=states)

    print("Prob mejorar (global):", p_up_global)
    print("Prob empeorar (global):", p_down_global)
    print("Engagement estacionario (global):", E_inf_global)

    # Guardar matriz global
    df_P_global = pd.DataFrame(
        P_global,
        index=[f"state_{s}" for s in states],
        columns=[f"state_{s}" for s in states],
    )
    df_P_global.to_csv(os.path.join(OUTPUT_CSV_DIR, "P_global.csv"), index=True)

    # Guardar métricas globales
    df_global_metrics = pd.DataFrame(
        {
            "prob_mejorar": [p_up_global],
            "prob_empeorar": [p_down_global],
            "engagement_estacionario": [E_inf_global],
        }
    )
    df_global_metrics.to_csv(
        os.path.join(OUTPUT_CSV_DIR, "global_metrics.csv"), index=False
    )

    # Figura: heatmap global
    plt.figure(figsize=(6, 5))
    sns.heatmap(P_global, annot=True, fmt=".2f", xticklabels=states, yticklabels=states)
    plt.title("Matriz de transición global")
    plt.xlabel("Estado siguiente")
    plt.ylabel("Estado actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIG_DIR, "P_global_heatmap.png"))
    plt.close()

    # =========================
    # 3. Combinación simple: Project Tag + Seniority
    # =========================
    print("\nCalculando combinaciones Project Tag + Seniority...")
    df_proj_sen = markov_summary_by_group(
        data,
        group_cols=["Project Tag", "Seniority"],
        col_state="Engagement Group",
        states=states,
        min_rows=30,
    )

    df_proj_sen.to_csv(
        os.path.join(OUTPUT_CSV_DIR, "summary_ProjectTag_Seniority.csv"), index=False
    )

    # Figura: barras top 15 combinaciones
    if not df_proj_sen.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        top_n = min(15, df_proj_sen.shape[0])
        vals = df_proj_sen["engagement_estacionario"].head(top_n)
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        colors = cm.plasma(norm(vals))

        labels = df_proj_sen.iloc[:top_n].apply(
            lambda r: f"{r['Project Tag']} | {r['Seniority']}", axis=1
        )

        ax.bar(labels, vals, color=colors)
        ax.set_title(
            "Top combinaciones Project Tag + Seniority por engagement estacionario",
            fontsize=14,
        )
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_FIG_DIR, "ProjectTag_Seniority_top_combinations_barplot.png"
            )
        )
        plt.close()

        # Scatter prob_mejorar vs prob_empeorar
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            df_proj_sen["prob_mejorar"],
            df_proj_sen["prob_empeorar"],
            c=df_proj_sen["engagement_estacionario"],
            cmap="viridis",
            s=80,
        )
        plt.colorbar(sc, label="Engagement estacionario")
        plt.xlabel("Probabilidad de mejorar")
        plt.ylabel("Probabilidad de empeorar")
        plt.title("Dinámica de engagement (Project Tag + Seniority)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_FIG_DIR, "ProjectTag_Seniority_mejorar_vs_empeorar_scatter.png"
            )
        )
        plt.close()

    # =========================
    # 4. Combinación rica: Project Tag + Seniority + Position + Location
    # =========================
    print("\nCalculando combinaciones Project Tag + Seniority + Position + Location...")
    df_combo = markov_summary_by_group(
        data,
        group_cols=["Project Tag", "Seniority", "Position", "Location"],
        col_state="Engagement Group",
        states=states,
        min_rows=40,
    )

    df_combo.to_csv(
        os.path.join(
            OUTPUT_CSV_DIR, "summary_ProjectTag_Seniority_Position_Location.csv"
        ),
        index=False,
    )

    if not df_combo.empty:
        # Violinplot por Seniority
        plt.figure(figsize=(10, 5))
        sns.violinplot(
            data=df_combo,
            x="Seniority",
            y="engagement_estacionario",
            inner="quart",
        )
        plt.title("Distribución de engagement estacionario por seniority")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_FIG_DIR, "EngagementEstacionario_violin_Seniority.png"
            )
        )
        plt.close()

        # Correlación entre métricas
        corr = df_combo[["prob_mejorar", "prob_empeorar", "engagement_estacionario"]].corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlación entre métricas de Markov")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIG_DIR, "Markov_metrics_correlation_heatmap.png"))
        plt.close()

    # =========================
    # 5. Modelo sobre engagement estacionario
    # =========================
    if df_combo.empty:
        print("\nNo hay suficientes combinaciones para modelar. Terminando.")
        return

    print("\nEntrenando modelo de regresión para engagement estacionario...")

    df_model = df_combo.copy()
    y = df_model["engagement_estacionario"].values

    X_cat = df_model[["Project Tag", "Seniority", "Position", "Location"]].astype(str)
    X = pd.get_dummies(X_cat, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    linreg = LinearRegression().fit(X_train, y_train)

    r2_train = linreg.score(X_train, y_train)
    r2_test = linreg.score(X_test, y_test)

    print("R² Train (engagement estacionario):", r2_train)
    print("R² Test (engagement estacionario):", r2_test)

    coef_df = pd.DataFrame({"feature": X.columns, "coef": linreg.coef_}).sort_values(
        "coef"
    )
    coef_df.to_csv(
        os.path.join(OUTPUT_CSV_DIR, "regression_coeffs_engagement_estacionario.csv"),
        index=False,
    )

    # Top positivos
    plt.figure(figsize=(8, 12))
    plt.barh(
        coef_df["feature"].tail(20),
        coef_df["coef"].tail(20),
        color="seagreen",
    )
    plt.title(
        "Top 20 variables con influencia positiva en engagement estacionario", fontsize=12
    )
    plt.xlabel("Coeficiente")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_FIG_DIR, "Regression_top_positive_features_engagement.png"
        )
    )
    plt.close()

    # Top negativos
    plt.figure(figsize=(8, 12))
    plt.barh(coef_df["feature"].head(20), coef_df["coef"].head(20), color="firebrick")
    plt.title(
        "Top 20 variables con influencia negativa en engagement estacionario", fontsize=12
    )
    plt.xlabel("Coeficiente")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_FIG_DIR, "Regression_top_negative_features_engagement.png"
        )
    )
    plt.close()

    # =========================
    # 6. Modelo sobre score_neto = prob_mejorar - prob_empeorar
    # =========================
    print("\nEntrenando modelo de regresión para score_neto (mejorar - empeorar)...")

    df_model["score_neto"] = df_model["prob_mejorar"] - df_model["prob_empeorar"]
    y_net = df_model["score_neto"].values

    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
        X, y_net, test_size=0.3, random_state=42
    )
    linreg_net = LinearRegression().fit(X_train_n, y_train_n)

    r2_train_net = linreg_net.score(X_train_n, y_train_n)
    r2_test_net = linreg_net.score(X_test_n, y_test_n)

    print("R² Train (score_neto):", r2_train_net)
    print("R² Test (score_neto):", r2_test_net)

    coef_net = pd.DataFrame(
        {"feature": X.columns, "coef": linreg_net.coef_}
    ).sort_values("coef")
    coef_net.to_csv(
        os.path.join(OUTPUT_CSV_DIR, "regression_coeffs_score_neto.csv"), index=False
    )

    # Top positivos score_neto
    plt.figure(figsize=(8, 12))
    plt.barh(
        coef_net["feature"].tail(15),
        coef_net["coef"].tail(15),
        color="dodgerblue",
    )
    plt.title(
        "Top 15 variables con mejor dinámica (mejorar > empeorar)", fontsize=12
    )
    plt.xlabel("Coeficiente")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_FIG_DIR, "Regression_top_positive_features_score_neto.png"
        )
    )
    plt.close()

    # Top negativos score_neto
    plt.figure(figsize=(8, 12))
    plt.barh(
        coef_net["feature"].head(15),
        coef_net["coef"].head(15),
        color="crimson",
    )
    plt.title(
        "Top 15 variables con peor dinámica (empeorar > mejorar)", fontsize=12
    )
    plt.xlabel("Coeficiente")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_FIG_DIR, "Regression_top_negative_features_score_neto.png"
        )
    )
    plt.close()

    # =========================
    # 7. Scatter engagement_estacionario vs score_neto
    # =========================
    print("\nGenerando scatter engagement_estacionario vs score_neto...")

    plt.figure(figsize=(8, 6))
    sc2 = plt.scatter(
        df_model["engagement_estacionario"],
        df_model["score_neto"],
        c=df_model["prob_mejorar"],
        cmap="viridis",
        s=90,
        alpha=0.8,
    )
    plt.colorbar(sc2, label="Probabilidad de mejorar")
    plt.xlabel("Engagement estacionario")
    plt.ylabel("Score neto (mejorar - empeorar)")
    plt.title("Relación entre engagement estacionario y dinámica del engagement")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_FIG_DIR, "Scatter_engagementEstacionario_vs_scoreNeto.png"
        )
    )
    plt.close()

    print("\nAnálisis completado. CSVs en:", OUTPUT_CSV_DIR)
    print("Figuras en:", OUTPUT_FIG_DIR)


if __name__ == "__main__":
    main()