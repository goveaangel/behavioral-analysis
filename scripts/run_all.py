# run_all.py

import subprocess
import os

scripts = [
    "run_markov_analysis.py",
    "summarize_markov_outputs.py",
    "rf_markov_model.py",
    "xgb_markov_model.py",
    "evaluate_models.py"
]

print("\n🚀 Running full pipeline...\n")

BASE = os.path.join(os.path.dirname(__file__))

for s in scripts:
    path = os.path.join(BASE, s)
    print(f"\n▶ Running {s}\n")
    subprocess.run(["python", path], check=True)

print("\n✅ Pipeline completo ejecutado exitosamente.\n")