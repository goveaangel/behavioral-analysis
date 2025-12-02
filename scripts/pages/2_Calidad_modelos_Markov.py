import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

# ============= RUTAS =============

# BASE_DIR ahora es scripts/pages
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Subimos dos niveles: pages -> scripts -> proyecto
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

OUTPUT_CSV_DIR = os.path.join(PROJECT_DIR, "outputs", "csv")
MODEL_DIR      = os.path.join(PROJECT_DIR, "outputs", "models")

SUMMARY_FILE = os.path.join(
    OUTPUT_CSV_DIR, "summary_ProjectTag_Seniority_Position_Location.csv"
)
EVAL_FILE    = os.path.join(
    OUTPUT_CSV_DIR, "model_evaluation_report.csv"
)

# Usamos los mismos pipelines que la app principal
RF_ENG_PIPE = os.path.join(MODEL_DIR, "rf_engagement_pipeline.pkl")
RF_NET_PIPE = os.path.join(MODEL_DIR, "rf_score_neto_pipeline.pkl")


# ============= LOADERS CACHEADOS =============

@st.cache_data
def load_summary():
    return pd.read_csv(SUMMARY_FILE)


@st.cache_data
def load_eval_metrics():
    if not os.path.exists(EVAL_FILE):
        return None
    return pd.read_csv(EVAL_FILE)


@st.cache_resource
def load_pipelines():
    rf_eng_pipe = joblib.load(RF_ENG_PIPE)
    rf_net_pipe = joblib.load(RF_NET_PIPE)
    return rf_eng_pipe, rf_net_pipe


# ============= PÁGINA PRINCIPAL =============

def main():
    st.title("📊 Calidad de modelos vs Cadenas de Markov")

    st.write(
        """
        Esta página resume **qué tan bien** los modelos de Machine Learning
        (Random Forest / XGBoost) logran **reproducir las métricas derivadas de
        las Cadenas de Markov** a nivel segmento.

        - Tomamos como “verdad” las métricas de Markov:
          `engagement_estacionario` y `score_neto = prob_mejorar - prob_empeorar`.
        - Entrenamos modelos que predicen esas métricas a partir de:
          `Project Tag`, `Seniority`, `Position`, `Location`.
        - Aquí medimos la **calidad global** de esa aproximación.
        """
    )

    # Cargar datos
    df_summary = load_summary()
    df_eval    = load_eval_metrics()
    rf_eng_pipe, rf_net_pipe = load_pipelines()

    # ---------- 1. Dashboard de métricas globales ----------
    st.subheader("📋 Métricas globales de evaluación")

    if df_eval is None:
        st.info(
            "No se encontraron métricas de evaluación "
            f"en `{os.path.basename(EVAL_FILE)}`.\n\n"
            "Ejecuta primero el pipeline de entrenamiento:\n\n"
            "`python scripts/run_all.py`"
        )
    else:
        st.write(
            """
            Usamos:
            - **RMSE**: error cuadrático medio (en unidades de la métrica).
            - **MAE**: error absoluto medio.
            - **R²**: proporción de variabilidad explicada (lo usamos como un
              equivalente de “overall accuracy” respecto a Markov).
            """
        )

        st.dataframe(
            df_eval.style.format(
                {"rmse": "{:.3f}", "mae": "{:.3f}", "r2": "{:.3f}"}
            ),
            use_container_width=True,
        )

        # Resumen rápido: mejor modelo para engagement
        df_eng = df_eval[df_eval["model"].str.contains("eng", case=False)]
        if not df_eng.empty:
            best = df_eng.sort_values("r2", ascending=False).iloc[0]
            st.markdown(
                f"""
                **Mejor modelo para *engagement_estacionario***  
                - Modelo: `{best['model']}`  
                - R² ≈ **{best['r2']:.2f}** → explica aproximadamente
                  **{best['r2']*100:.1f}%** de la variabilidad de la métrica
                  estacionaria de Markov entre segmentos.
                """
            )

        # Gráfica de barras de R²
        fig_r2 = px.bar(
            df_eval,
            x="model",
            y="r2",
            title="R² por modelo (qué tanto explican las métricas de Markov)",
            labels={"r2": "R² (proporción de variabilidad explicada)"},
            text=df_eval["r2"].round(2),
        )
        fig_r2.update_traces(textposition="outside")
        fig_r2.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")

    # ---------- 2. Comparación directa Markov vs Random Forest ----------
    st.subheader("📈 Markov vs Random Forest por segmento")

    st.write(
        """
        En este scatter cada punto es una combinación
        `(Project Tag, Seniority, Position, Location)`:

        - El eje **X** muestra el engagement estacionario obtenido con **Markov**.
        - El eje **Y** muestra la predicción del modelo **Random Forest** (pipeline)
          para esa misma combinación.

        Cuanto más cerca estén los puntos de la línea diagonal, mejor está
        reproduciendo el modelo las métricas de Markov.
        """
    )

    # Construir X para el pipeline
    X_cat = df_summary[["Project Tag", "Seniority", "Position", "Location"]].astype(str)
    y_markov = df_summary["engagement_estacionario"].values
    y_rf_pred = rf_eng_pipe.predict(X_cat)

    comp_df = df_summary.copy()
    comp_df["eng_markov"] = y_markov
    comp_df["eng_pred_rf"] = y_rf_pred

    fig_scatter = px.scatter(
        comp_df,
        x="eng_markov",
        y="eng_pred_rf",
        hover_data=["Project Tag", "Seniority", "Position", "Location"],
        labels={
            "eng_markov": "Engagement estacionario (Markov)",
            "eng_pred_rf": "Engagement predicho (Random Forest)",
        },
        title="Comparación Markov vs Random Forest por segmento",
    )

    # Línea diagonal referencia y=x
    min_val = float(min(comp_df["eng_markov"].min(), comp_df["eng_pred_rf"].min()))
    max_val = float(max(comp_df["eng_markov"].max(), comp_df["eng_pred_rf"].max()))
    fig_scatter.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="gray", dash="dash"),
        name="y = x",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown(
        """
        **Lectura rápida del gráfico:**

        - Puntos **sobre** la línea → el modelo y Markov coinciden casi perfecto.
        - Puntos **por encima** → el modelo tiende a sobreestimar el engagement.
        - Puntos **por debajo** → el modelo tiende a subestimar el engagement.

        La idea no es reemplazar las Cadenas de Markov, sino usar los modelos
        de ML como una **aproximación rápida** de las métricas estacionarias para
        nuevos segmentos o configuraciones.
        """
    )


if __name__ == "__main__":
    main()