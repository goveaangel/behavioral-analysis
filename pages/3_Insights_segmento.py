# scripts/pages/3_Insights_segmento.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ===================== RUTAS =====================

# Directorio actual del archivo (pages/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Subir un nivel: proyecto raíz (globant/)
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

OUTPUT_CSV_DIR = os.path.join(PROJECT_DIR, "outputs", "csv")
DATA_DIR       = os.path.join(PROJECT_DIR, "data")

SUMMARY_FILE = os.path.join(
    OUTPUT_CSV_DIR, "summary_ProjectTag_Seniority_Position_Location.csv"
)
DATA_FILE = os.path.join(
    DATA_DIR, "data_globant_cleaned.csv"
)

# ===================== LOADERS =====================

@st.cache_data
def load_summary():
    df = pd.read_csv(SUMMARY_FILE)

    # Asegurar columna score_neto
    if "score_neto" not in df.columns and \
       {"prob_mejorar", "prob_empeorar"}.issubset(df.columns):
        df["score_neto"] = df["prob_mejorar"] - df["prob_empeorar"]

    # Crear una etiqueta legible de segmento
    for col in ["Project Tag", "Seniority", "Position", "Location"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["segmento"] = (
        df["Project Tag"] + " | " +
        df["Seniority"] + " | " +
        df["Position"] + " | " +
        df["Location"]
    )

    return df


@st.cache_data
def load_full_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Project Tag", "Seniority", "Position", "Location"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


# ===================== UI PRINCIPAL =====================

def main():
    st.title("📊 Insights globales de engagement a largo plazo")

    st.write(
        """
        Esta pestaña resume **cómo se comportan los segmentos a largo plazo**, usando:

        - `engagement_estacionario`: nivel de equilibrio esperado del engagement (escala 1–5).
        - `score_neto = prob_mejorar - prob_empeorar`: tendencia neta a subir o bajar de nivel.

        Aquí no elegimos un solo segmento, sino que vemos **cuáles grupos salen mejor o peor parados**.
        """
    )

    df_summary = load_summary()
    df_full    = load_full_data()

    if df_summary.empty:
        st.error("No se encontraron datos en el resumen de Markov.")
        return

    # ================== SIDEBAR: FILTROS ==================
    st.sidebar.header("Filtros")

    project_options = ["(Todos los proyectos)"] + sorted(
        df_summary["Project Tag"].dropna().unique().tolist()
    )
    selected_project = st.sidebar.selectbox(
        "Filtrar por Project Tag",
        project_options,
    )

    # (Opcional) podrías añadir más filtros, pero para la presentación con Project basta.
    df_filt = df_summary.copy()
    if selected_project != "(Todos los proyectos)":
        df_filt = df_filt[df_filt["Project Tag"] == selected_project]

    if df_filt.empty:
        st.warning("No hay segmentos para el filtro seleccionado.")
        return

    # ================== RESUMEN GLOBAL ==================
    st.markdown("---")
    st.subheader("🌍 Resumen global de niveles de equilibrio")

    eng_mean = float(df_filt["engagement_estacionario"].mean())
    eng_q25  = float(df_filt["engagement_estacionario"].quantile(0.25))
    eng_q75  = float(df_filt["engagement_estacionario"].quantile(0.75))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Promedio de engagement estacionario (escala 1–5)",
            f"{eng_mean:.2f}",
        )
    with col2:
        st.metric(
            "Percentil 25 (segmentos más bajos)",
            f"{eng_q25:.2f}",
        )
    with col3:
        st.metric(
            "Percentil 75 (segmentos más altos)",
            f"{eng_q75:.2f}",
        )

    st.caption(
        "Estas métricas resumen en qué rango se mueven los segmentos a largo plazo, "
        "según la dinámica estimada por las Cadenas de Markov."
    )

    # ================== TOP / BOTTOM POR ENGAGEMENT ESTACIONARIO ==================
    st.markdown("---")
    st.subheader("🏆 Top y bottom segmentos por nivel de equilibrio")

    top_n = st.slider(
        "¿Cuántos segmentos quieres ver en cada ranking?",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    # Top
    df_top_eng = (
        df_filt.sort_values("engagement_estacionario", ascending=False)
        .head(top_n)
        .copy()
    )
    fig_top_eng = px.bar(
        df_top_eng,
        x="segmento",
        y="engagement_estacionario",
        title=f"Top {top_n} segmentos por engagement a largo plazo",
        labels={"engagement_estacionario": "Engagement estacionario (escala 1–5)"},
    )
    fig_top_eng.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_top_eng, use_container_width=True)

    st.caption(
        "Estos segmentos son los que, según el modelo de Markov, tienden a estabilizarse "
        "en un nivel de engagement más alto a largo plazo."
    )

    # Bottom
    df_bottom_eng = (
        df_filt.sort_values("engagement_estacionario", ascending=True)
        .head(top_n)
        .copy()
    )
    fig_bottom_eng = px.bar(
        df_bottom_eng,
        x="segmento",
        y="engagement_estacionario",
        title=f"Bottom {top_n} segmentos por engagement a largo plazo",
        labels={"engagement_estacionario": "Engagement estacionario (escala 1–5)"},
    )
    fig_bottom_eng.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bottom_eng, use_container_width=True)

    st.caption(
        "Estos segmentos son los que, de no intervenir, tenderían a permanecer en niveles "
        "de engagement más bajos en el largo plazo."
    )

    # ================== TOP / BOTTOM POR SCORE NETO ==================
    st.markdown("---")
    st.subheader("📈 Tendencias netas de mejora / deterioro")

    if "score_neto" in df_filt.columns:
        df_tend_up = (
            df_filt.sort_values("score_neto", ascending=False)
            .head(top_n)
            .copy()
        )
        df_tend_down = (
            df_filt.sort_values("score_neto", ascending=True)
            .head(top_n)
            .copy()
        )

        # Top score_neto (mejor dinámica)
        fig_tend_up = px.bar(
            df_tend_up,
            x="segmento",
            y="score_neto",
            title=f"Top {top_n} segmentos con mejor dinámica (score neto)",
            labels={"score_neto": "Score neto = prob(mejorar) - prob(empeorar)"},
        )
        fig_tend_up.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_tend_up, use_container_width=True)

        st.caption(
            "Score neto positivo significa que, en promedio, es más probable subir de nivel "
            "de engagement que bajar. Estos segmentos tienen una dinámica relativamente saludable."
        )

        # Bottom score_neto (peor dinámica)
        fig_tend_down = px.bar(
            df_tend_down,
            x="segmento",
            y="score_neto",
            title=f"Bottom {top_n} segmentos con peor dinámica (score neto)",
            labels={"score_neto": "Score neto = prob(mejorar) - prob(empeorar)"},
        )
        fig_tend_down.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_tend_down, use_container_width=True)

        st.caption(
            "Score neto negativo indica que es más probable bajar de nivel que subir. "
            "Estos segmentos son candidatos claros a priorizar en planes de acción."
        )
    else:
        st.info(
            "El resumen no contiene la columna `score_neto`. "
            "Asegúrate de recalcular el CSV agregado con esa métrica."
        )

    # ================== SCATTER: EQUILIBRIO vs DINÁMICA ==================
    st.markdown("---")
    st.subheader("🔍 Cuadrantes de segmentos: nivel vs dinámica")

    if "score_neto" in df_filt.columns:
        fig_scatter = px.scatter(
            df_filt,
            x="engagement_estacionario",
            y="score_neto",
            color="Project Tag",
            hover_data=["segmento"],
            labels={
                "engagement_estacionario": "Engagement estacionario (escala 1–5)",
                "score_neto": "Score neto (mejorar - empeorar)",
            },
            title="Relación entre nivel de equilibrio y dinámica del engagement por segmento",
        )

        # Líneas de referencia (promedios)
        x_mean = df_filt["engagement_estacionario"].mean()
        y_mean = df_filt["score_neto"].mean()

        fig_scatter.add_vline(
            x=x_mean,
            line_dash="dash",
            annotation_text="Promedio engagement",
            annotation_position="top left",
        )
        fig_scatter.add_hline(
            y=y_mean,
            line_dash="dash",
            annotation_text="Promedio score neto",
            annotation_position="top right",
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        st.caption(
            "Cada punto es un segmento. El eje X indica el nivel de equilibrio esperado del engagement, "
            "y el eje Y la tendencia neta a mejorar o empeorar. "
            "El cuadrante superior derecho (alto engagement + score neto positivo) "
            "muestra los grupos más saludables a largo plazo."
        )


if __name__ == "__main__":
    main()