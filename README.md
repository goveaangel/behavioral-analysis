

# 📘 README — Behavioral Engagement Analysis (Globant)

⸻

## 🧠 Descripción general del proyecto

Este proyecto construye una solución integral para entender, modelar y predecir el engagement de empleados, usando una combinación de:

🔹 1. Cadenas de Markov

Para modelar dinámicas reales de engagement:
	•	Probabilidad de mejorar
	•	Probabilidad de empeorar
	•	Matrices de transición por segmento
	•	Distribución estacionaria (hacia dónde tiende el engagement)

🔹 2. Machine Learning (RF, XGB, LR)

Para predecir:
	•	Engagement estacionario estimado
	•	Score neto = prob_mejorar − prob_empeorar

Modelos entrenados:
	•	Random Forest (RF)
	•	Gradient Boosting (XGBoost)
	•	Linear Regression (baseline)

🔹 3. Visualizaciones y análisis

Incluye:
	•	Heatmaps
	•	Feature importances
	•	Scatter plots
	•	Transiciones entre niveles
	•	Distribución inicial
	•	Trayectoria esperada del engagement

🔹 4. Aplicación interactiva (Streamlit)

Permite predecir engagement por:
	•	Project Tag
	•	Seniority
	•	Position
	•	Location

E incluye gráficas dinámicas con Plotly.

🔹 5. Pipeline 100% automatizado

Con un archivo que corre TODO:

python scripts/run_all.py


⸻

## 📂 Estructura del repositorio

behavioral-analysis/
│
├── data/
│   ├── data_globant.csv
│   ├── data_globant_cleaned.csv
│   └── sample_data.csv
│
├── notebooks/
│   ├── EDA_*.ipynb
│   ├── limpieza_datos.ipynb
│   ├── Modelo_Markov.ipynb
│   ├── engagement_phase.ipynb
│   └── …
│
├── outputs/
│   ├── csv/
│   │   ├── summary_ProjectTag_Seniority_Position_Location.csv
│   │   ├── global_metrics.csv
│   │   ├── final_top_bottom_segments.csv
│   │   ├── rf_feature_importances_*.csv
│   │   ├── xgb_feature_importances_*.csv
│   │   └── model_evaluation_report.csv   ← Evaluación completa
│   │
│   ├── figs/
│   │   ├── Scatter_engagement_vs_scoreNeto.png
│   │   ├── P_global_heatmap.png
│   │   ├── Markov_metrics_correlation_heatmap.png
│   │   └── evaluation_plots/             ← Gráficas de evaluación
│   │
│   └── models/
│       ├── rf_engagement_estacionario.pkl
│       ├── rf_score_neto.pkl
│       ├── rf_feature_columns.json
│       ├── xgb_engagement_estacionario.pkl
│       ├── xgb_score_neto.pkl
│       ├── xgb_feature_columns.json
│       └── …
│
├── scripts/
│   ├── markov_engagement.py
│   ├── run_markov_analysis.py
│   ├── summarize_markov_outputs.py
│   ├── rf_markov_model.py
│   ├── xgb_markov_model.py
│   ├── evaluate_models.py         ← Nuevo
│   ├── run_all.py                 ← Nuevo
│   └── streamlit_app.py
│
└── README.md


⸻

## 🔁 Flujo del pipeline

 RAW DATA
   ↓
 Limpieza de datos
   ↓
 Cálculo de Cadenas de Markov
 |→ Matrices de transición
 |→ Probabilidad de mejorar / empeorar
 |→ Distribución estacionaria
   ↓
 Resumen por segmento (summary.csv)
   ↓
 Entrenamiento ML (RF / XGB / LR)
 |→ Predicción estacionaria
 |→ Pred. score neto
 |→ Feature importances
   ↓
 Evaluación de modelos (métricas + gráficas)
   ↓
 Streamlit App para interacción del usuario


⸻

## 🧱 Scripts principales

📌 markov_engagement.py
	•	Construye matrices de transición
	•	Calcula estacionaria
	•	Probabilidades de cambio
	•	Engagement estacionario esperado

⸻

📌 run_markov_analysis.py

Corre Markov para:
	•	Project Tag
	•	Seniority
	•	Position
	•	Location

Genera CSVs intermedios.

⸻

📌 summarize_markov_outputs.py

Fusiona:
	•	prob_mejorar
	•	prob_empeorar
	•	engagement estacionario
	•	counts

Y genera el archivo clave:

summary_ProjectTag_Seniority_Position_Location.csv


⸻

📌 rf_markov_model.py / xgb_markov_model.py

Entrenan y guardan:
	•	Modelos .pkl
	•	Columnas one-hot
	•	Feature importances
	•	Predicciones

⸻

 📌 evaluate_models.py

Reporta:
	•	RMSE
	•	MAE
	•	R²
	•	Comparación RF vs XGB vs LR
	•	Gráficas de residuales y pred vs real

Output:

model_evaluation_report.csv


⸻

📌 run_all.py

Ejecuta todo el pipeline automáticamente:

python scripts/run_all.py

Incluye:
	1.	Markov
	2.	Resumen
	3.	RF
	4.	XGB
	5.	Evaluación

⸻

## 📌 streamlit_app.py

Aplicación interactiva con:
	•	Predicción por combinación
	•	Gráficos inicial + estacionario + modelo RF
	•	Trayectoria esperada del engagement (Plotly)
	•	Segmentos similares
	•	Debug de features
	•	Validación de inputs
	•	Manejo de segmentos sin datos

⸻

## 🧪 Evaluación de modelos

El archivo:

outputs/csv/model_evaluation_report.csv

Incluye:

Modelo	RMSE	MAE	R²	Target
RF engagement	…	…	…	engagement_estacionario
XGB engagement	…	…	…	engagement_estacionario
RF score neto	…	…	…	score_neto
XGB score neto	…	…	…	score_neto

Gráficas generadas en:

outputs/figs/evaluation_plots/


⸻

## 🚀 Cómo correr el proyecto

1. Instalar dependencias

pip install -r requirements.txt


⸻

## 2. Ejecutar el pipeline completo

Este comando genera:
	•	Cadenas de Markov
	•	Resumen por segmento
	•	Entrenamiento RF
	•	Entrenamiento XGB
	•	Evaluación completa

python scripts/run_all.py


⸻

## 3. Abrir la aplicación


streamlit run scripts/streamlit_app.py

Esto abre la app interactiva donde puedes:
	•	Seleccionar Project Tag / Seniority / Position / Location
	•	Ver engagement inicial, esperado y predicho
	•	Visualizar trayectoria del engagement
	•	Comparar segmentos similares
	•	Explorar distribuciones y dinámicas
