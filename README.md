# 🧠 Behavioral analysis of the teams of a software company

---

## 📂 Repository Structure

```bash
README.md
```

---

## ⚙️ Project Overview

---

## 📊 Methodology

---

## 📈 Results Summary

---

## 🧠 Key Insights

---

## 🧩 Technologies Used

---

## 📘 Reports

---

## 👥 Authors

- **Diego Vértiz Padilla**  
- **José Ángel Govea García**  
- **Daniel Alberto Sánchez Fortiz**  
- **Augusto Ley Rodríguez**  
- **Ángel Esparza Enríquez**

Tecnológico de Monterrey, School of Engineering and Sciences  
Guadalajara, Jalisco — México  

---

## 🔒 Confidentiality

---

## 🧾 License



📌 Resumen del proyecto

Este proyecto construye un sistema completo para:
	1.	Analizar el engagement de empleados usando:
	•	Cadenas de Markov (transiciones históricas entre niveles de engagement)
	•	Métricas derivadas (probabilidad de mejorar, empeorar, estado estacionario)
	2.	Generar reportes agregados por segmentos como:
	•	Project Tag
	•	Seniority
	•	Position
	•	Location
	•	Combinaciones de ellos
	3.	Entrenar modelos predictivos (Random Forest y XGBoost) usando esos datos agregados.
	4.	Proveer una interfaz interactiva (Streamlit) que:
	•	Predice engagement esperado para un empleado/segmento.
	•	Muestra gráficas de trayectoria esperada del engagement.
	•	Compara el segmento con otros similares.
	•	Ofrece insights de dinámica de Markov.

El resultado final es una herramienta completa de análisis, simulación y predicción.

⸻

🏛 Estructura del proyecto

behavioral-analysis/
│
├── data/
│   └── data_globant_cleaned.csv
│
├── scripts/
│   ├── markov_engagement.py
│   ├── run_markov_analysis.py
│   ├── summarize_markov_outputs.py
│   ├── rf_markov_model.py
│   ├── xgb_markov_model.py
│   ├── rf_markov_pipeline.py
│   └── streamlit_app.py
│
└── outputs/
    ├── csv/
    └── models/


⸻

🧩 1. Cálculo de Cadenas de Markov (script modular)

📄 Archivo: markov_engagement.py

Este archivo contiene funciones independientes y reutilizables para construir y analizar cadenas de Markov basadas en engagement de empleados.

Incluye:

✔ Construcción de matriz de transición

A partir del dataset ordenado por:

Name → Date → Engagement Group

El módulo:
	•	Detecta transiciones (ej. 3→4, 4→5…)
	•	Construye la matriz de conteos
	•	Normaliza para obtener la matriz de transición P

✔ Probabilidad de mejorar

Promedio de probabilidades de pasar a un estado superior.

✔ Probabilidad de empeorar

Promedio de probabilidades de caer a un estado inferior.

✔ Distribución estacionaria

Calculada usando el eigenvector asociado a λ=1.

✔ Engagement estacionario

Valor esperado del engagement en el largo plazo.

Este módulo es reutilizable en cualquier script del proyecto.

⸻

🛠 2. Script de análisis general de Markov

📄 Archivo: run_markov_analysis.py

Este script:
	•	Lee el dataset crudo.
	•	Limpia fechas y estados inválidos.
	•	Genera análisis global:
	•	Matriz de transición global
	•	Probabilidades de mejorar/empeorar
	•	Distribución estacionaria
	•	Engagement estacionario promedio
	•	Además genera análisis por grupos:
	•	Project Tag
	•	Seniority
	•	Position
	•	Location
	•	Combinaciones (Project + Seniority + Position + Location)

Los resultados se guardan en:

outputs/csv/summary_<combination>.csv
outputs/png/*.png   (gráficas)

Es la base para todo lo que viene después.

⸻

📊 3. Script para resumir outputs agregados

📄 Archivo: summarize_markov_outputs.py

Este script:
	•	Lee cada CSV generado por el análisis de Markov.
	•	Limpia y ordena por engagement estacionario.
	•	Genera tablas más breves para usar en reportes y modelos.
	•	Compara segmentos (top/bottom).

Funciona como una capa de estandarización para que los modelos trabajen siempre con formatos consistentes.

⸻

🤖 4. Modelos predictivos

Tenemos dos enfoques:

⸻

🔹 A) Modelos clásicos (construíamos X_row manualmente)

Archivos:
	•	rf_markov_model.py
	•	xgb_markov_model.py

Antes, el modelo recibía un vector de features manual.
Esto funcionaba, pero era frágil a inconsistencias entre entrenamiento y predicción.

⸻

🔹 B) Modelos modernos con Pipeline (recomendado)

📄 Archivo: rf_markov_pipeline.py

Este fue el fix definitivo:
	•	Construye un Pipeline con:
	•	OneHotEncoder (categóricas)
	•	Random Forest
	•	Entrena 2 modelos:
	•	Engagement estacionario esperado
	•	Score neto (prob_mejorar – prob_empeorar)
	•	Guarda ambos modelos completos en:

outputs/models/rf_engagement_pipeline.pkl
outputs/models/rf_score_neto_pipeline.pkl



Ventajas:
	•	Se evita construir X_row a mano.
	•	El pipeline se encarga del one-hot correcto.
	•	El modelo recibe datos crudos y funciona siempre igual.
	•	Zero riesgo de “todas las columnas en cero”.

⸻

🖥 5. Interfaz interactiva: Streamlit

📄 Archivo: streamlit_app.py

Es la capa de producto:
una app que permite interactuar con los modelos y los datos.

Incluye:

⸻

✔ Selección de características

El usuario elige:
	•	Proyecto
	•	Seniority
	•	Posición
	•	Ubicación

La app arma un DataFrame crudo y lo envía al pipeline de RF.

⸻

✔ Predicción del modelo

Muestra:

✨ Engagement estacionario esperado (modelo)

✨ Score neto (mejora – empeora)

✨ Interpretación automática (positiva / negativa / neutra)

⸻

✔ Comparación con otros segmentos
	•	Si el empleado está en ATLINT → muestra los 5 mejores segmentos dentro de ATLINT.
	•	Ayuda a entender “qué tan bueno es este contexto”.

⸻

✔ Gráfica profesional de Plotly: Trayectoria esperada

Esta gráfica muestra:
	•	Dónde suelen iniciar empleados similares (histórico)
	•	Cómo evolucionarían hacia el engagement estacionario
	•	Qué predice el modelo de ML
	•	Línea de tiempo simulada

Incluye fallback:
	•	Si no hay suficientes datos exactos → usa Project+Seniority
	•	Si tampoco → usa Project Tag
	•	Si tampoco → global
(esto evita que la gráfica desaparezca)

⸻

📦 6. Outputs generados

El proyecto produce automáticamente:

✔ CSVs con resúmenes de Markov

En outputs/csv/summary_*.csv

✔ Gráficas PNG

En outputs/png/

✔ Modelos entrenados

En outputs/models/:
	•	rf_engagement_pipeline.pkl
	•	rf_score_neto_pipeline.pkl
	•	(opcional) XGB equivalents

⸻

🧪 7. Cómo correr el proyecto

🟦 1. Preprocesar Markov

python scripts/run_markov_analysis.py

🟦 2. Resumir outputs

python scripts/summarize_markov_outputs.py

🟦 3. Entrenar modelos (pipeline recomendado)

python scripts/rf_markov_pipeline.py

🟦 4. Ejecutar la aplicación

streamlit run scripts/streamlit_app.py


⸻

🧠 8. Qué modelos se están prediciendo

🔍 Engagement estacionario esperado

Qué nivel de engagement se espera a largo plazo para un segmento con esas características.

🔍 Score neto

prob_mejorar – prob_empeorar
Un indicador de dinámica:
	•	0 → tendencia a mejorar
	•	< 0 → tendencia a deteriorarse
	•	≈ 0 → neutro

⸻

🚀 9. Tecnología utilizada
	•	Python 3
	•	Pandas (manejo de datos)
	•	NumPy (matrices)
	•	scikit-learn (Random Forest, OneHotEncoder, Pipeline)
	•	XGBoost (opcional)
	•	Plotly (visualizaciones interactivas)
	•	Streamlit (interfaz web)
	•	Cadenas de Markov (modelos estocásticos)

⸻

💡 10. Beneficios del sistema
	•	Combina análisis probabilístico (Markov) con ML predictivo.
	•	Permite comparar segmentos y entender dinámicas internas.
	•	Ofrece una forma interactiva y clara para comunicar resultados.
	•	Es extensible a optimización (“a qué proyecto debería moverse un empleado para maximizar su engagement”).
	•	Está modularizado y listo para producción.

⸻

🏁 11. Próximos pasos (opcional)
	•	Explicabilidad SHAP
	•	Optimización de reasignación (estocástica)
	•	Añadir XGBoost Pipeline
	•	Guardar histórico de predicciones
	•	Dashboard adicional estilo BI

