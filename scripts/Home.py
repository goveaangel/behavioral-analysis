import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

# BASE_DIR es la carpeta scripts/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Subimos un nivel para llegar a la raíz del proyecto
PROJECT_DIR = os.path.join(BASE_DIR, "..")

OUTPUT_CSV_DIR = os.path.join(PROJECT_DIR, "outputs", "csv")
MODEL_DIR      = os.path.join(PROJECT_DIR, "outputs", "models")

SUMMARY_FILE   = os.path.join(
    OUTPUT_CSV_DIR, "summary_ProjectTag_Seniority_Position_Location.csv"
)
DATA_FILE = os.path.join(PROJECT_DIR, "data", "data_globant_cleaned.csv")
RF_ENG_PIPE = os.path.join(MODEL_DIR, "rf_engagement_pipeline.pkl")
RF_NET_PIPE = os.path.join(MODEL_DIR, "rf_score_neto_pipeline.pkl")


@st.cache_data
def load_summary():
    return pd.read_csv(SUMMARY_FILE)

@st.cache_data
def load_full_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    # Normalizar texto en las columnas clave
    for col in ["Project Tag", "Seniority", "Position", "Location"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


@st.cache_resource
def load_pipelines():
    rf_eng_pipe = joblib.load(RF_ENG_PIPE)
    rf_net_pipe = joblib.load(RF_NET_PIPE)
    return rf_eng_pipe, rf_net_pipe


def main():
    st.title("🔮 Predictor de Engagement – Globant (Markov + Random Forest)")
    st.write(
        """
        Este demo usa:
        - Cadenas de Markov para estimar métricas por segmento,
        - Random Forest para predecir:
            * Engagement estacionario esperado
            * Score neto = prob_mejorar - prob_empeorar
        """
    )

    df_summary = load_summary()
    
    df_full = load_full_data()
    rf_eng_pipe, rf_net_pipe = load_pipelines()

    # Sidebar de selección
    st.sidebar.header("Características del segmento / empleado")

    proj_options = sorted(df_summary["Project Tag"].dropna().unique())
    sen_options  = sorted(df_summary["Seniority"].dropna().unique())
    pos_options  = sorted(df_summary["Position"].dropna().unique())
    loc_options  = sorted(df_summary["Location"].dropna().unique())

    project   = st.sidebar.selectbox("Project Tag", proj_options)
    seniority = st.sidebar.selectbox("Seniority",  sen_options)
    position  = st.sidebar.selectbox("Position",   pos_options)
    location  = st.sidebar.selectbox("Location",   loc_options)

    if st.sidebar.button("Predecir engagement"):
        # ---- Guardar selección en session_state ----
        st.session_state["selected_project"]   = str(project).strip()
        st.session_state["selected_seniority"] = str(seniority).strip()
        st.session_state["selected_position"]  = str(position).strip()
        st.session_state["selected_location"]  = str(location).strip()

        # Normalizar valores seleccionados
        project_norm   = st.session_state["selected_project"]
        seniority_norm = st.session_state["selected_seniority"]
        position_norm  = st.session_state["selected_position"]
        location_norm  = st.session_state["selected_location"]

        # DataFrame crudo con 1 fila
        row = pd.DataFrame(
            {
                "Project Tag": [project_norm],
                "Seniority": [seniority_norm],
                "Position": [position_norm],
                "Location": [location_norm],
            }
        )

        # DEBUG opcional
        # st.write("Row de entrada al pipeline:", row)

        # Predicciones usando los pipelines
        pred_eng = float(rf_eng_pipe.predict(row)[0])
        pred_net = float(rf_net_pipe.predict(row)[0])

        st.subheader("📈 Resultados de la predicción")
        st.metric("Engagement estacionario esperado", f"{pred_eng:.3f}")
        st.metric("Score neto (mejorar - empeorar)", f"{pred_net:.3f}")

        # Interpretación simple
        if pred_net > 0.05:
            msg = "Tendencia positiva: es más probable mejorar el engagement que empeorarlo."
        elif pred_net < -0.05:
            msg = "Tendencia negativa: hay riesgo de deterioro del engagement."
        else:
            msg = "Tendencia neutra: probabilidades de mejora y empeoramiento están balanceadas."
        st.write("**Interpretación:**", msg)

        # -------- Contexto vs otros segmentos (lo dejamos igual) --------
        st.markdown("---")
        st.markdown("### Contexto vs otros segmentos")

        mask_sim_proj = (df_summary["Project Tag"] == project)
        df_sim = df_summary[mask_sim_proj].sort_values(
            "engagement_estacionario", ascending=False
        ).head(5)
        st.write("Top 5 combinaciones en este mismo proyecto:", df_sim)

                # -------- NUEVA GRÁFICA: Trayectoria inicial → estacionario --------
        st.markdown("---")
        st.markdown("### Trayectoria esperada del engagement para este segmento")

        # 1) Intento 1: Project Tag + Seniority
        mask_seg = (
            (df_full["Project Tag"] == project_norm) &
            (df_full["Seniority"] == seniority_norm)
        )
        sub = df_full[mask_seg].copy()
        nivel_agg = "Project Tag + Seniority"

        # 2) Si no hay datos, relajar a solo Project Tag
        if sub.empty:
            mask_seg = (df_full["Project Tag"] == project_norm)
            sub = df_full[mask_seg].copy()
            nivel_agg = "Project Tag (todas las seniorities)"

        # 3) Si sigue vacío (raro), usar global
        if sub.empty:
            sub = df_full.copy()
            nivel_agg = "Global (todos los empleados)"

        if sub.empty:
            st.info("No hay suficientes datos históricos para mostrar la trayectoria en este segmento.")
        else:
            st.caption(f"Nivel de agregación usado para el histórico: **{nivel_agg}**")

            sub = sub.sort_values(["Name", "Date"])

            # Engagement inicial por empleado (primer registro)
            first = sub.groupby("Name").head(1)
            start_mean = first["Engagement Group"].mean()

            # Engagement estacionario Markov:
            #   1) Proj+Sen
            #   2) Proj
            #   3) Global
            mask_summary = (
                (df_summary["Project Tag"] == project) &
                (df_summary["Seniority"] == seniority)
            )
            if mask_summary.any():
                markov_est = float(
                    df_summary.loc[mask_summary, "engagement_estacionario"].mean()
                )
                nivel_markov = "Project Tag + Seniority"
            else:
                mask_summary = (df_summary["Project Tag"] == project)
                if mask_summary.any():
                    markov_est = float(
                        df_summary.loc[mask_summary, "engagement_estacionario"].mean()
                    )
                    nivel_markov = "Project Tag"
                else:
                    markov_est = float(df_summary["engagement_estacionario"].mean())
                    nivel_markov = "Global"

            # Trayectoria: de t=0 (inicial) a t=T (estacionario)
            n_steps = 6
            periods = list(range(n_steps + 1))

            target = markov_est
            eng_path = np.linspace(start_mean, target, n_steps + 1)

            traj_df = pd.DataFrame({
                "Periodo": periods,
                "Engagement esperado (Markov)": eng_path
            })

            fig = px.line(
                traj_df,
                x="Periodo",
                y="Engagement esperado (Markov)",
                markers=True,
                title="Evolución esperada del engagement\n(desde el inicial hacia el estacionario)",
            )
            fig.update_layout(
                xaxis_title="Periodo (pasos de tiempo)",
                yaxis_title="Nivel de engagement",
                yaxis=dict(range=[1, 5]),
            )

            # Predicción del modelo como referencia en el último periodo
            fig.add_scatter(
                x=[periods[-1]],
                y=[pred_eng],
                mode="markers+text",
                name="Predicción modelo (RF)",
                text=[f"RF: {pred_eng:.2f}"],
                textposition="top center"
            )

            # Marcar el estacionario Markov en el último periodo
            fig.add_scatter(
                x=[periods[-1]],
                y=[markov_est],
                mode="markers+text",
                name=f"Estacionario Markov ({nivel_markov})",
                text=[f"π*: {markov_est:.2f}"],
                textposition="bottom center"
            )

            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()