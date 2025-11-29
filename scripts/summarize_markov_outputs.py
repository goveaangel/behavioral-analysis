# summarize_markov_outputs.py

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV_DIR = os.path.join(BASE_DIR, "..", "outputs", "csv")

def main():
    # --- Cargar CSVs principales ---
    proj_sen_path = os.path.join(OUTPUT_CSV_DIR, "summary_ProjectTag_Seniority.csv")
    combo_path    = os.path.join(OUTPUT_CSV_DIR, "summary_ProjectTag_Seniority_Position_Location.csv")
    global_path   = os.path.join(OUTPUT_CSV_DIR, "global_metrics.csv")
    coeff_eng_path = os.path.join(OUTPUT_CSV_DIR, "regression_coeffs_engagement_estacionario.csv")
    coeff_net_path = os.path.join(OUTPUT_CSV_DIR, "regression_coeffs_score_neto.csv")

    df_global = pd.read_csv(global_path)
    df_proj_sen = pd.read_csv(proj_sen_path)
    df_combo = pd.read_csv(combo_path)
    df_coef_eng = pd.read_csv(coeff_eng_path)
    df_coef_net = pd.read_csv(coeff_net_path)

    print("\n=== MÉTRICAS GLOBALES ===")
    print(df_global)

    # --- Top / Bottom por Project Tag + Seniority ---
    print("\n=== Top 5 combinaciones Project Tag + Seniority por engagement_estacionario ===")
    top5_proj_sen = df_proj_sen.head(5)
    print(top5_proj_sen[["Project Tag", "Seniority", "engagement_estacionario",
                         "prob_mejorar", "prob_empeorar"]])

    print("\n=== Bottom 5 combinaciones Project Tag + Seniority ===")
    bottom5_proj_sen = df_proj_sen.tail(5)
    print(bottom5_proj_sen[["Project Tag", "Seniority", "engagement_estacionario",
                            "prob_mejorar", "prob_empeorar"]])

    # --- Top / Bottom por combinación rica ---
    print("\n=== Top 5 combos completos (Project Tag + Seniority + Position + Location) ===")
    top5_combo = df_combo.head(5)
    print(top5_combo[["Project Tag", "Seniority", "Position", "Location",
                      "engagement_estacionario", "prob_mejorar", "prob_empeorar"]])

    print("\n=== Bottom 5 combos completos ===")
    bottom5_combo = df_combo.tail(5)
    print(bottom5_combo[["Project Tag", "Seniority", "Position", "Location",
                         "engagement_estacionario", "prob_mejorar", "prob_empeorar"]])

    # --- Coeficientes del modelo sobre engagement_estacionario ---
    print("\n=== Top 10 variables con mayor coeficiente (engagement_estacionario) ===")
    print(df_coef_eng.sort_values("coef", ascending=False).head(10))

    print("\n=== Top 10 variables con menor coeficiente (engagement_estacionario) ===")
    print(df_coef_eng.sort_values("coef", ascending=True).head(10))

    # --- Coeficientes del modelo sobre score_neto ---
    print("\n=== Top 10 variables con mejor dinámica (score_neto) ===")
    print(df_coef_net.sort_values("coef", ascending=False).head(10))

    print("\n=== Top 10 variables con peor dinámica (score_neto) ===")
    print(df_coef_net.sort_values("coef", ascending=True).head(10))

    # --- Guardar un CSV “final” con top y bottom para el reporte ---
    final_summary = {
        "top5_proj_sen": top5_proj_sen,
        "bottom5_proj_sen": bottom5_proj_sen,
        "top5_combo": top5_combo,
        "bottom5_combo": bottom5_combo,
    }

    # Concatenar con una columna que indique de qué lista viene
    pieces = []
    for name, df_part in final_summary.items():
        temp = df_part.copy()
        temp["group"] = name
        pieces.append(temp)

    df_final = pd.concat(pieces, ignore_index=True)
    final_path = os.path.join(OUTPUT_CSV_DIR, "final_top_bottom_segments.csv")
    df_final.to_csv(final_path, index=False)
    print(f"\nCSV final de segmentos clave guardado en: {final_path}")

if __name__ == "__main__":
    main()