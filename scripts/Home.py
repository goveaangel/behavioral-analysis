import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from markov_engagement import (
    build_transition_matrix,
    estacionaria,
    engagement_estacionario,
    DEFAULT_STATES,
)

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


# ================== LOADERS ==================

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

@st.cache_data
def load_state_means(df):
    """
    Calcula el promedio real continuo de Engagement por cada estado discreto.
    """
    tmp = (
        df[df["Engagement"] > 0]
        .groupby("Engagement Group")["Engagement"]
        .agg(mean_eng_real="mean")
    )
    return tmp

@st.cache_resource
def load_pipelines():
    rf_eng_pipe = joblib.load(RF_ENG_PIPE)
    rf_net_pipe = joblib.load(RF_NET_PIPE)
    return rf_eng_pipe, rf_net_pipe


# ================== HELPERS MARKOV ==================

def select_history_segment(df_full, project, seniority, position, location, min_rows=40):
    """
    Elige el subconjunto de datos históricos que se usará para construir la
    dinámica Markov (matriz P), con fallback de granularidad:

    1) Project + Seniority + Position + Location
    2) Project + Seniority
    3) Project
    4) Global
    """
    project   = str(project).strip()
    seniority = str(seniority).strip()
    position  = str(position).strip()
    location  = str(location).strip()

    # 1) Combo completo
    mask_full = (
        (df_full["Project Tag"] == project) &
        (df_full["Seniority"]   == seniority) &
        (df_full["Position"]    == position) &
        (df_full["Location"]    == location)
    )
    sub_full = df_full[mask_full].copy()
    if sub_full.shape[0] >= min_rows:
        return sub_full, "Project Tag + Seniority + Position + Location"

    # 2) Project + Seniority
    mask_ps = (
        (df_full["Project Tag"] == project) &
        (df_full["Seniority"]   == seniority)
    )
    sub_ps = df_full[mask_ps].copy()
    if sub_ps.shape[0] >= min_rows:
        return sub_ps, "Project Tag + Seniority"

    # 3) Solo Project
    mask_p = (df_full["Project Tag"] == project)
    sub_p = df_full[mask_p].copy()
    if sub_p.shape[0] >= min_rows:
        return sub_p, "Project Tag (todas las seniorities / posiciones / ubicaciones)"

    # 4) Global
    return df_full.copy(), "Global (todos los empleados)"


def compute_markov_from_segment(sub, states=None):
    """
    Dado un subconjunto histórico `sub`, construye:

    - Matriz de transición P
    - Distribución estacionaria pi*
    - Nivel estacionario esperado E[X] (escala 1–5)
    """
    if states is None:
        states = DEFAULT_STATES

    if sub.empty:
        return None, None, np.nan

    sub_sorted = sub.sort_values(["Name", "Date"])
    P = build_transition_matrix(sub_sorted, col_state="Engagement Group", states=states)
    if P.sum() == 0:
        return None, None, np.nan

    pi_star = estacionaria(P)
    markov_level = engagement_estacionario(pi_star, states=states)
    return sub_sorted, P, markov_level


# ================== APP ==================

def main():
    st.title("🔮 Predictor de Engagement – Globant (Markov + Random Forest)")
    st.write(
        """
        Este demo combina:
        - Cadenas de Markov para estimar la dinámica de engagement por segmento.
        - Random Forest para aproximar el nivel de equilibrio esperado a partir
          de características del segmento.
        """
    )

    df_summary = load_summary()
    df_full    = load_full_data()
    state_means = load_state_means(df_full)
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

    # --- Botón: solo prende la bandera en session_state ---
    predict_clicked = st.sidebar.button("Predecir engagement")

    if predict_clicked:
        # guardamos la selección actual
        st.session_state["selected_project"]   = str(project).strip()
        st.session_state["selected_seniority"] = str(seniority).strip()
        st.session_state["selected_position"]  = str(position).strip()
        st.session_state["selected_location"]  = str(location).strip()

        # para que Insights tome los mismos filtros
        st.session_state["from_home_for_insight"] = True

        # 🔑 muy importante: decirle a la app que ya hay una predicción activa
        st.session_state["do_prediction"] = True

    # --- Si ya hicimos al menos una predicción, mostramos resultados y gráfica ---
    if st.session_state.get("do_prediction", False):

        # Usamos SIEMPRE lo guardado en session_state para ser consistentes
        project_norm   = st.session_state["selected_project"]
        seniority_norm = st.session_state["selected_seniority"]
        position_norm  = st.session_state["selected_position"]
        location_norm  = st.session_state["selected_location"]

        # DataFrame crudo con 1 fila para el modelo ML
        row = pd.DataFrame(
            {
                "Project Tag": [project_norm],
                "Seniority": [seniority_norm],
                "Position": [position_norm],
                "Location": [location_norm],
            }
        )

        # ===== 1) Predicciones usando los pipelines (ML) =====
        pred_eng = float(rf_eng_pipe.predict(row)[0])   # escala 1–5 (nivel esperado)
        pred_net = float(rf_net_pipe.predict(row)[0])   # score neto

        # ---------- Reconstrucción de engagement continuo ----------
        eng_continuo = None
        try:
            floor_state = int(np.floor(pred_eng))
            ceil_state  = int(np.ceil(pred_eng))

            if floor_state == ceil_state:
                eng_continuo = state_means.loc[floor_state]["mean_eng_real"]
            else:
                w = pred_eng - floor_state
                mu_floor = state_means.loc[floor_state]["mean_eng_real"]
                mu_ceil  = state_means.loc[ceil_state]["mean_eng_real"]
                eng_continuo = (1 - w) * mu_floor + w * mu_ceil
        except Exception:
            eng_continuo = None

        # ===== 2) Dinámica Markov para este segmento (histórico) =====
        states = [1, 2, 3, 4, 5]
        sub_hist, nivel_agg = select_history_segment(
            df_full,
            project_norm,
            seniority_norm,
            position_norm,
            location_norm,
            min_rows=40,
        )

        sub_sorted = None
        P = None
        markov_est = np.nan

        if not sub_hist.empty:
            sub_sorted, P, markov_est = compute_markov_from_segment(
                sub_hist,
                states=states,
            )

        # ===== 3) Métricas en pantalla =====
        st.subheader("📈 Resultados para el segmento seleccionado")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Nivel esperado ML (escala 1–5)",
                f"{pred_eng:.3f}",
                help="Predicción del modelo Random Forest sobre el nivel de equilibrio esperado."
            )

        with col2:
            if not np.isnan(markov_est):
                st.metric(
                    "Nivel de equilibrio Markov (escala 1–5)",
                    f"{markov_est:.3f}",
                    help="Equilibrio esperado de la cadena de Markov estimada con datos históricos del segmento."
                )
            else:
                st.metric(
                    "Nivel de equilibrio Markov (escala 1–5)",
                    "N/D",
                )

        with col3:
            if not np.isnan(markov_est):
                diff_ml_markov = pred_eng - markov_est
                st.metric(
                    "Diferencia ML – Markov",
                    f"{diff_ml_markov:+.3f}",
                    help="Diferencia entre la aproximación ML y el equilibrio Markov."
                )

        if eng_continuo is not None:
            st.metric(
                "Engagement esperado (escala original continua)",
                f"{eng_continuo:.2f}",
                help=(
                    "Traducción del nivel esperado en la escala 1–5 a la escala continua "
                    "original de la encuesta, usando la media observada por estado."
                ),
            )
            st.caption(
                "Reconstruido a partir de la media real de engagement continuo en cada estado discreto (1–5)."
            )
        else:
            st.caption("No se pudo reconstruir el engagement en la escala continua original.")

        st.metric("Score neto (mejorar - empeorar)", f"{pred_net:.3f}")

        # Interpretación simple del score neto
        if pred_net > 0.05:
            msg = "Tendencia positiva: es más probable mejorar el engagement que empeorarlo."
        elif pred_net < -0.05:
            msg = "Tendencia negativa: hay riesgo de deterioro del engagement."
        else:
            msg = "Tendencia neutra: probabilidades de mejora y empeoramiento están balanceadas."
        st.write("**Interpretación del score neto:**", msg)

        # -------- Contexto vs otros segmentos --------
        st.markdown("---")
        st.markdown("### Contexto vs otros segmentos del mismo Project Tag")

        mask_sim_proj = (df_summary["Project Tag"] == project_norm)
        df_sim = df_summary[mask_sim_proj].sort_values(
            "engagement_estacionario", ascending=False
        ).head(5)
        st.write("Top 5 combinaciones históricas en este Project Tag:", df_sim)

                # -------- TRAYECTORIA MARKOV REAL --------
        st.markdown("---")
        st.markdown("### Evolución esperada del engagement (dinámica Markov)")

        if P is None:
            st.info(
                "No hay suficientes transiciones históricas en este segmento "
                "para construir una dinámica de Markov confiable."
            )
        else:
            st.caption(
                f"Dinámica estimada usando datos históricos a nivel: **{nivel_agg}**. "
                "Cada paso se interpreta como ~1 ciclo mensual de medición de engagement. "
                "Si no hay suficientes datos en la combinación completa de filtros, "
                "agregamos el historial (por ejemplo, a nivel Project Tag + Seniority) "
                "para evitar conclusiones basadas en muy pocos registros."
            )

            # --- π0: SIEMPRE histórico inicial del segmento ---
            snap = sub_sorted.groupby("Name").head(1)
            fuente_pi0 = "primer engagement observado por persona en el segmento"

            states = [1, 2, 3, 4, 5]
            counts = (
                snap["Engagement Group"]
                .astype(int)
                .value_counts()
                .reindex(states, fill_value=0)
            )
            total = counts.sum()
            if total == 0:
                # Fallback extremo: si por alguna razón no hay datos,
                # usamos un delta en el estado más cercano a la media
                start_mean = snap["Engagement Group"].mean()
                k = int(round(start_mean))
                k = max(1, min(5, k))
                pi0 = np.zeros(len(states))
                pi0[states.index(k)] = 1.0
                fuente_pi0 += " (aprox. por media de estados)"
            else:
                pi0 = counts.to_numpy() / total

            st.caption(f"Distribución inicial π₀ basada en: **{fuente_pi0}**")

            # --- Evolución π_t y E[X_t] ---
            n_steps = 12  # 12 meses aprox.
            periods = list(range(n_steps + 1))

            eng_path = []
            pi_t = pi0.copy()
            for t in periods:
                eng_t = float(np.dot(states, pi_t))
                eng_path.append(eng_t)
                pi_t = pi_t @ P

            # Equilibrio Markov π*
            pi_star = estacionaria(P)
            markov_eq = engagement_estacionario(pi_star, states=states)

            traj_df = pd.DataFrame({
                "Mes": periods,
                "Nivel esperado Markov": eng_path,
            })

            fig = px.line(
                traj_df,
                x="Mes",
                y="Nivel esperado Markov",
                markers=True,
                title="Trayectoria promedio de engagement\n"
                      "desde el arranque histórico hacia el equilibrio",
            )
            fig.update_layout(
                xaxis_title="Meses (aprox.) desde el inicio",
                yaxis_title="Nivel de engagement (escala 1–5)",
                yaxis=dict(range=[1, 5]),
            )

            # Punto de equilibrio Markov (π*)
            fig.add_scatter(
                x=[periods[-1]],
                y=[markov_eq],
                mode="markers+text",
                name="Equilibrio Markov π*",
                text=[f"π*: {markov_eq:.2f}"],
                textposition="bottom center",
            )

            # Predicción del modelo ML como referencia
            fig.add_scatter(
                x=[periods[-1]],
                y=[pred_eng],
                mode="markers+text",
                name="Predicción modelo (ML, escala 1–5)",
                text=[f"ML: {pred_eng:.2f}"],
                textposition="top center",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "La curva muestra cómo evoluciona el nivel medio de engagement del segmento "
                "si aplicamos repetidamente las probabilidades de cambio observadas históricamente. "
                "El punto π* es el nivel de equilibrio de la cadena de Markov; el punto ML "
                "es la aproximación del modelo de Machine Learning a ese equilibrio, dado el segmento elegido."
            )


if __name__ == "__main__":
    main()