# scripts/pages/3_Insights_segmento.py
from utils.ui import set_background
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ===================== RUTAS =====================

# BASE_DIR ahora es scripts/pages
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Subimos dos niveles: pages -> scripts -> proyecto
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

OUTPUT_CSV_DIR = os.path.join(PROJECT_DIR, "outputs", "csv")
MODEL_DIR      = os.path.join(PROJECT_DIR, "outputs", "models")
DATA_DIR       = os.path.join(PROJECT_DIR, "data")

SUMMARY_FILE = os.path.join(
    OUTPUT_CSV_DIR, "summary_ProjectTag_Seniority_Position_Location.csv"
)
DATA_FILE    = os.path.join(
    DATA_DIR, "data_globant_cleaned.csv"
)

RF_ENG_PIPE = os.path.join(MODEL_DIR, "rf_engagement_pipeline.pkl")
RF_NET_PIPE = os.path.join(MODEL_DIR, "rf_score_neto_pipeline.pkl")


# ===================== LOADERS =====================

@st.cache_data
def load_summary():
    df = pd.read_csv(SUMMARY_FILE)
    # Asegurar columna score_neto
    if "score_neto" not in df.columns and \
       {"prob_mejorar", "prob_empeorar"}.issubset(df.columns):
        df["score_neto"] = df["prob_mejorar"] - df["prob_empeorar"]
    return df


@st.cache_data
def load_full_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Project Tag", "Seniority", "Position", "Location"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


@st.cache_resource
def load_pipelines():
    rf_eng_pipe = joblib.load(RF_ENG_PIPE)
    rf_net_pipe = joblib.load(RF_NET_PIPE)
    return rf_eng_pipe, rf_net_pipe


# ===================== LÓGICA DE INSIGHT =====================

def compute_risk_level(eng_est, score_neto, eng_q25, eng_q75):
    if np.isnan(eng_est):
        return "⚪ Indefinido (sin datos suficientes)"

    if score_neto is not None and score_neto < -0.05:
        return "🔴 Alto riesgo (tendencia a deterioro)"

    if eng_est < eng_q25:
        return "🟠 Riesgo moderado (engagement bajo vs otros segmentos)"

    if eng_est > eng_q75 and (score_neto is None or score_neto >= 0):
        return "🟢 Segmento saludable (buena dinámica de engagement)"

    return "🟡 Situación intermedia / mixta"


def generate_text_insight(
    project, seniority, position, location,
    nivel_agg,
    eng_inicial, eng_est, eng_global,
    prob_up, prob_down, score_neto,
    risk_label
):
    linea_base = (
        f"Para el segmento seleccionado (**Project Tag = {project}**, "
        f"**Seniority = {seniority}**, **Position = {position}**, "
        f"**Location = {location}**), el análisis se realizó a nivel "
        f"**{nivel_agg}** para asegurar suficiente información histórica."
    )

    linea_dinamica = ""
    if not np.isnan(eng_inicial) and not np.isnan(eng_est):
        if eng_est > eng_inicial + 0.05:
            tendencia = "ligeramente superior a"
        elif eng_est < eng_inicial - 0.05:
            tendencia = "ligeramente inferior a"
        else:
            tendencia = "similar a"

        linea_dinamica = (
            f"\n\nEn promedio, los empleados de este segmento **inician** en un "
            f"nivel de engagement de aproximadamente **{eng_inicial:.2f}**, "
            f"y el modelo de cadenas de Markov indica que tienden a "
            f"estabilizarse en un nivel **{tendencia}** ese valor "
            f"(**{eng_est:.2f}**)."
        )

    linea_prob = ""
    if (prob_up is not None) and (prob_down is not None):
        linea_prob = (
            f"\n\nLa probabilidad promedio de **mejorar** de nivel es "
            f"~**{prob_up:.2f}**, mientras que la probabilidad de **empeorar** "
            f"es ~**{prob_down:.2f}**. "
        )
        if prob_up > prob_down:
            linea_prob += "Esto sugiere una dinámica relativamente positiva."
        elif prob_up < prob_down:
            linea_prob += (
                "Esto sugiere una dinámica de engagement delicada, con mayor "
                "tendencia a deterioro."
            )
        else:
            linea_prob += (
                "Las probabilidades de mejora y deterioro están equilibradas."
            )

    linea_neto = ""
    if score_neto is not None:
        if score_neto > 0.05:
            linea_neto = (
                f"\n\nEl **score neto** (mejorar - empeorar) es **{score_neto:.3f}**, "
                "lo que indica una **tendencia neta al alza** en el engagement."
            )
        elif score_neto < -0.05:
            linea_neto = (
                f"\n\nEl **score neto** (mejorar - empeorar) es **{score_neto:.3f}**, "
                "lo que indica una **tendencia neta a la baja** en el engagement."
            )
        else:
            linea_neto = (
                f"\n\nEl **score neto** (mejorar - empeorar) es **{score_neto:.3f}**, "
                "mostrando una dinámica prácticamente neutral."
            )

    linea_global = ""
    if not np.isnan(eng_global) and not np.isnan(eng_est):
        if eng_est > eng_global + 0.1:
            comp = "por encima del promedio global"
        elif eng_est < eng_global - 0.1:
            comp = "por debajo del promedio global"
        else:
            comp = "en línea con el promedio global"

        linea_global = (
            f"\n\nComparado con todos los segmentos, el engagement estacionario "
            f"de este grupo (**{eng_est:.2f}**) está **{comp}** "
            f"(promedio global ≈ **{eng_global:.2f}**)."
        )

    linea_riesgo = f"\n\n**Evaluación global del segmento:** {risk_label}"

    return (
        linea_base
        + linea_dinamica
        + linea_prob
        + linea_neto
        + linea_global
        + linea_riesgo
    )


def classify_segment_type(eng_est, score_neto, prob_up, prob_down, eng_global):
    if np.isnan(eng_est):
        return "Sin datos suficientes"

    delta_global = eng_est - eng_global
    momentum = score_neto if score_neto is not None else 0.0
    vol = 0.0
    if (prob_up is not None) and (prob_down is not None):
        vol = prob_up + prob_down

    if eng_est < 2.8 and momentum < -0.03:
        return "Segmento en deterioro estructural"
    if eng_est < 3.0 and momentum >= 0.0:
        return "Segmento frágil pero recuperable"
    if 2.8 <= eng_est <= 3.2 and abs(momentum) <= 0.03:
        return "Segmento estable pero estancado"
    if eng_est > 3.2 and momentum > 0.03:
        return "Segmento saludable con potencial"
    if vol > 0.6:
        return "Segmento volátil (altas subidas y bajadas)"

    if delta_global < -0.1:
        return "Segmento por debajo del promedio"
    if delta_global > 0.1:
        return "Segmento por encima del promedio"
    return "Segmento en situación intermedia"


def generate_action_bullets(seg_type, project, seniority, position, location):
    bullets = []

    if seg_type == "Segmento en deterioro estructural":
        bullets.append(
            "Revisar carga de trabajo y expectativas del cliente en "
            f"**{project}**; patrones similares suelen asociarse a sobrecarga "
            "o poca claridad de prioridades."
        )
        bullets.append(
            "Programar conversaciones 1:1 con el liderazgo para entender "
            "fricciones específicas (procesos, comunicación, decisiones)."
        )
        bullets.append(
            "Revisar rotación reciente en este segmento y compararla con "
            "otros equipos del mismo Project Tag."
        )

    elif seg_type == "Segmento frágil pero recuperable":
        bullets.append(
            "El nivel de engagement es relativamente bajo pero aún sin "
            "tendencia fuerte a la baja: es un buen momento para intervenir."
        )
        bullets.append(
            "Explorar acciones rápidas: feedback más frecuente, "
            "reconocimiento visible y ajuste fino de carga de trabajo."
        )
        bullets.append(
            "Comparar prácticas de onboarding, coaching y comunicación con "
            "segmentos del mismo Project Tag que tengan mejor dinámica."
        )

    elif seg_type == "Segmento estable pero estancado":
        bullets.append(
            "El segmento es estable, pero el engagement no crece. El riesgo es "
            "la salida silenciosa por falta de reto."
        )
        bullets.append(
            "Revisar oportunidades de crecimiento y movilidad interna para "
            f"roles como **{position}**, especialmente en niveles **{seniority}**."
        )
        bullets.append(
            "Explorar iniciativas de innovación interna o participación en "
            "proyectos estratégicos para reactivar la motivación."
        )

    elif seg_type == "Segmento saludable con potencial":
        bullets.append(
            "Este segmento combina buen engagement con tendencia positiva: "
            "es ideal para documentar buenas prácticas."
        )
        bullets.append(
            "Analizar qué hace distinto a este grupo (liderazgo, rituales, "
            "dinámica con el cliente) y usarlo como benchmark."
        )
        bullets.append(
            "Evitar sobrecargar al equipo: es común concentrar cada vez más "
            "responsabilidades en los segmentos que mejor funcionan."
        )

    elif seg_type == "Segmento volátil (altas subidas y bajadas)":
        bullets.append(
            "La probabilidad de subir y bajar de engagement es alta; esto sugiere "
            "un contexto inestable (cambios frecuentes o comunicación inconsistente)."
        )
        bullets.append(
            "Revisar frecuencia de cambios de alcance, stakeholders o liderazgo "
            f"en el proyecto **{project}**."
        )
        bullets.append(
            "Refinar claridad de objetivos de corto plazo y dar visibilidad sobre "
            "decisiones clave para reducir la incertidumbre."
        )

    elif seg_type == "Segmento por debajo del promedio":
        bullets.append(
            "El engagement esperado está por debajo del promedio de Globant, "
            "aunque la dinámica no es necesariamente crítica."
        )
        bullets.append(
            "Identificar 1–2 segmentos comparables (mismo Project Tag o Position) "
            "que tengan mejor desempeño y analizar qué prácticas se pueden replicar."
        )

    elif seg_type == "Segmento por encima del promedio":
        bullets.append(
            "Este segmento presenta un engagement esperado por encima del promedio; "
            "es importante entender y cuidar los factores que lo explican."
        )
        bullets.append(
            "Conversar con el liderazgo sobre autonomía, tipo de desafíos y "
            "relación con el cliente para no perder esos elementos."
        )

    else:  # intermedio o sin datos
        bullets.append(
            "El segmento no muestra señales extremas de riesgo ni de excelencia; "
            "es recomendable monitorearlo regularmente."
        )
        bullets.append(
            "Revisar al menos una vez por trimestre las métricas de engagement "
            "y compararlas con segmentos similares por Project Tag y Seniority."
        )

    return bullets


def find_better_neighbors(df_summary, project, seniority, position, location, k=3):
    base = df_summary[df_summary["Project Tag"] == project].copy()
    if base.empty:
        return []

    if "score_neto" not in base.columns and \
       {"prob_mejorar", "prob_empeorar"}.issubset(base.columns):
        base["score_neto"] = base["prob_mejorar"] - base["prob_empeorar"]

    base = base.sort_values(
        ["engagement_estacionario", "score_neto"],
        ascending=[False, False]
    )

    neighbors = []
    for _, row in base.iterrows():
        if (row["Seniority"] == seniority and
            row["Position"] == position and
            row["Location"] == location):
            continue
        neighbors.append(row)
        if len(neighbors) >= k:
            break

    resultados = []
    for row in neighbors:
        resultados.append(
            f"- **Project Tag:** {row['Project Tag']}, "
            f"**Seniority:** {row['Seniority']}, "
            f"**Position:** {row['Position']}, "
            f"**Location:** {row['Location']} "
            f"(engagement estacionario ≈ {row['engagement_estacionario']:.2f})"
        )
    return resultados


# ===================== UI PRINCIPAL =====================

def main():
    set_background()
    st.title("🧠 Insights automáticos por segmento")

    df_summary = load_summary()
    df_full    = load_full_data()
    rf_eng_pipe, rf_net_pipe = load_pipelines()

    # ---- Sidebar selección ----
    st.sidebar.header("Selecciona el segmento")

    proj_options = sorted(df_summary["Project Tag"].dropna().unique())
    sen_options  = sorted(df_summary["Seniority"].dropna().unique())
    pos_options  = sorted(df_summary["Position"].dropna().unique())
    loc_options  = sorted(df_summary["Location"].dropna().unique())

    # Valores por defecto tomados de session_state (si existen)
    default_project   = st.session_state.get("selected_project", proj_options[0])
    default_seniority = st.session_state.get("selected_seniority", sen_options[0])
    default_position  = st.session_state.get("selected_position", pos_options[0])
    default_location  = st.session_state.get("selected_location", loc_options[0])

    # Calcular índice seguro para cada selectbox
    proj_idx = proj_options.index(default_project)   if default_project in proj_options   else 0
    sen_idx  = sen_options.index(default_seniority)  if default_seniority in sen_options else 0
    pos_idx  = pos_options.index(default_position)   if default_position in pos_options   else 0
    loc_idx  = loc_options.index(default_location)   if default_location in loc_options   else 0

    project   = st.sidebar.selectbox("Project Tag", proj_options, index=proj_idx)
    seniority = st.sidebar.selectbox("Seniority",   sen_options,  index=sen_idx)
    position  = st.sidebar.selectbox("Position",    pos_options,  index=pos_idx)
    location  = st.sidebar.selectbox("Location",    loc_options,  index=loc_idx)

    
    clicked = st.sidebar.button("Generar insight")

    # Si no hizo clic pero venimos de Home con una selección, auto-disparar una vez
    if not clicked and st.session_state.get("from_home_for_insight", False):
        clicked = True
        # opcional: reseteamos la bandera para no auto-disparar siempre
        st.session_state["from_home_for_insight"] = False

    if clicked:
        project_norm   = str(project).strip()
        seniority_norm = str(seniority).strip()
        position_norm  = str(position).strip()
        location_norm  = str(location).strip()

        # ----- 1) Métricas de Markov (resumen) -----
        mask = (
            (df_summary["Project Tag"] == project_norm) &
            (df_summary["Seniority"]  == seniority_norm) &
            (df_summary["Position"]   == position_norm) &
            (df_summary["Location"]   == location_norm)
        )
        nivel_agg = "Project Tag + Seniority + Position + Location"
        df_seg = df_summary[mask].copy()

        if df_seg.empty:
            mask = (
                (df_summary["Project Tag"] == project_norm) &
                (df_summary["Seniority"]  == seniority_norm)
            )
            df_seg = df_summary[mask].copy()
            nivel_agg = "Project Tag + Seniority"

        if df_seg.empty:
            mask = (df_summary["Project Tag"] == project_norm)
            df_seg = df_summary[mask].copy()
            nivel_agg = "Project Tag"

        if df_seg.empty:
            df_seg = df_summary.copy()
            nivel_agg = "Global (todos los segmentos)"

        eng_est   = float(df_seg["engagement_estacionario"].mean())
        prob_up   = float(df_seg["prob_mejorar"].mean())   if "prob_mejorar" in df_seg.columns else None
        prob_down = float(df_seg["prob_empeorar"].mean())  if "prob_empeorar" in df_seg.columns else None
        score_neto = float(df_seg["score_neto"].mean())    if "score_neto" in df_seg.columns else None

        eng_global = float(df_summary["engagement_estacionario"].mean())
        eng_q25    = float(df_summary["engagement_estacionario"].quantile(0.25))
        eng_q75    = float(df_summary["engagement_estacionario"].quantile(0.75))

        # ----- 2) Engagement inicial histórico -----
        mask_hist = (
            (df_full["Project Tag"] == project_norm) &
            (df_full["Seniority"]   == seniority_norm)
        )
        sub = df_full[mask_hist].copy()
        if sub.empty:
            sub = df_full.copy()

        if not sub.empty:
            sub = sub.sort_values(["Name", "Date"])
            first = sub.groupby("Name").head(1)
            eng_inicial = float(first["Engagement Group"].mean())
        else:
            eng_inicial = np.nan

        # ----- 3) Predicciones ML (se usan sólo como contexto) -----
        row = pd.DataFrame(
            {
                "Project Tag": [project_norm],
                "Seniority": [seniority_norm],
                "Position": [position_norm],
                "Location": [location_norm],
            }
        )
        pred_eng = float(rf_eng_pipe.predict(row)[0])
        pred_net = float(rf_net_pipe.predict(row)[0])
        # (No los mostramos explícitamente, pero forman parte del contexto de análisis si lo necesitan después.)

        # ----- 4) Cálculo de riesgo + tipo de segmento -----
        risk_label = compute_risk_level(eng_est, score_neto, eng_q25, eng_q75)
        seg_type = classify_segment_type(
            eng_est=eng_est,
            score_neto=score_neto,
            prob_up=prob_up,
            prob_down=prob_down,
            eng_global=eng_global,
        )

        # ----- 5) INSIGHT EN TEXTO -----
        st.markdown("---")
        st.subheader("📝 Insight automático del segmento")

        insight_text = generate_text_insight(
            project_norm, seniority_norm, position_norm, location_norm,
            nivel_agg,
            eng_inicial, eng_est, eng_global,
            prob_up, prob_down, score_neto,
            risk_label
        )
        st.markdown(insight_text)

        # ----- 6) Diagnóstico ejecutivo + acciones -----
        st.markdown("---")
        st.subheader("🎯 Diagnóstico ejecutivo y focos de acción")

        st.markdown(f"**Tipo de segmento detectado:** {seg_type}")

        bullets = generate_action_bullets(
            seg_type,
            project=project_norm,
            seniority=seniority_norm,
            position=position_norm,
            location=location_norm,
        )

        st.markdown("**Sugerencias de enfoque para People / líderes:**")
        for b in bullets:
            st.markdown(f"- {b}")

        st.markdown("**Segmentos de referencia dentro del mismo Project Tag:**")
        neigh_lines = find_better_neighbors(
            df_summary, project_norm, seniority_norm, position_norm, location_norm, k=3
        )
        if neigh_lines:
            for line in neigh_lines:
                st.markdown(line)
        else:
            st.markdown(
                "_No se encontraron segmentos claramente mejores dentro de este mismo Project Tag._"
            )


if __name__ == "__main__":
    main()