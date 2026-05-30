"""
Etapa 2 – Paso 7: Gaps pendientes
1. Distribución final de clases post-limpieza
2. FN=0 documentado formalmente
3. Media y std de CV del mejor modelo
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score

# ─── Configuración ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "etapa2")
RANDOM_STATE = 42

print("=" * 60)
print("ETAPA 2 – Gaps pendientes")
print("=" * 60)

# ─── Carga ────────────────────────────────────────────────────────────────────
df       = pd.read_csv(os.path.join(OUTPUT_DIR, "datos_limpios_base.csv"))
split    = joblib.load(os.path.join(OUTPUT_DIR, "split_data.joblib"))
X_train  = split["X_train"]
y_train  = split["y_train"]
X_test   = split["X_test"]
y_test   = split["y_test"]

model_files = [f for f in os.listdir(OUTPUT_DIR)
               if f.startswith("mejor_modelo_") and f.endswith(".joblib")]
best_model  = joblib.load(os.path.join(OUTPUT_DIR, model_files[0]))
best_name   = model_files[0].replace("mejor_modelo_", "").replace("_base.joblib", "")

with open(os.path.join(OUTPUT_DIR, "selected_threshold.json")) as f:
    threshold = json.load(f)["threshold"]

print(f"\nModelo: {best_name}  |  Umbral: {threshold}")

# ─── Gap 1: Distribución final de clases post-limpieza ───────────────────────
print("\n[1] Distribución final de clases post-limpieza")

y = df["Label"]
counts = y.value_counts().sort_index()
labels = ["BENIGN", "ATTACK"]
values = [counts[0], counts[1]]
pcts   = [v / len(y) * 100 for v in values]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Escala lineal
colors = ["#3498db", "#e74c3c"]
bars = axes[0].bar(labels, values, color=colors, edgecolor="white", width=0.5)
for bar, pct in zip(bars, pcts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f"{pct:.2f}%", ha="center", va="bottom", fontsize=11)
axes[0].set_title("Distribución de clases — Dataset limpio\n(escala lineal)")
axes[0].set_ylabel("Número de instancias")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Escala logarítmica
bars2 = axes[1].bar(labels, values, color=colors, edgecolor="white", width=0.5)
axes[1].set_yscale("log")
for bar, val in zip(bars2, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, val * 1.3,
                 f"{val:,}", ha="center", va="bottom", fontsize=11)
axes[1].set_title("Distribución de clases — Dataset limpio\n(escala logarítmica)")
axes[1].set_ylabel("Número de instancias (log)")

fig.suptitle(f"Dataset: {len(df):,} filas  |  BENIGN={values[0]:,} ({pcts[0]:.2f}%)  "
             f"ATTACK={values[1]:,} ({pcts[1]:.2f}%)", fontsize=11)
fig.tight_layout()
path_dist = os.path.join(OUTPUT_DIR, "distribucion_clases_final.png")
fig.savefig(path_dist, dpi=150)
plt.close(fig)
print(f"  BENIGN: {values[0]:,} ({pcts[0]:.2f}%)")
print(f"  ATTACK: {values[1]:,} ({pcts[1]:.2f}%)")
print(f"  Guardado: distribucion_clases_final.png")

# ─── Gap 2: FN=0 documentado formalmente ─────────────────────────────────────
print("\n[2] Documentación formal de FN=0")

y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

tp = int(((y_test == 1) & (y_pred == 1)).sum())
fp = int(((y_test == 0) & (y_pred == 1)).sum())
fn = int(((y_test == 1) & (y_pred == 0)).sum())
tn = int(((y_test == 0) & (y_pred == 0)).sum())

fn_report = {
    "fn_count": fn,
    "tp_count": tp,
    "fp_count": fp,
    "tn_count": tn,
    "total_attacks_in_test": int(y_test.sum()),
    "recall_attack": float(recall_score(y_test, y_pred, zero_division=0)),
    "interpretacion": (
        "El modelo no produjo ningún falso negativo en el conjunto de prueba. "
        "Esto significa que todos los flujos de ataque fueron correctamente identificados. "
        "Sin embargo, este resultado debe interpretarse con cautela: el dataset contiene "
        "únicamente tres tipos de ataque web (Brute Force, XSS, SQL Injection) capturados "
        "en un entorno controlado. En producción real con variantes de ataques no vistas "
        "o nuevos vectores, el modelo podría generar falsos negativos. "
        "Adicionalmente, la clase SQL Injection solo contaba con 21 instancias originales "
        "(reducidas tras limpieza), lo que limita la capacidad de generalización del modelo "
        "para ese subtipo específico de ataque."
    ),
    "limitaciones": [
        "Dataset de entorno controlado: no refleja variabilidad real de producción",
        "Solo 3 tipos de ataque web: generalización limitada a otros vectores",
        "SQL Injection con muy pocas instancias: aprendizaje estadísticamente frágil",
        "FN=0 puede no mantenerse ante ataques evasivos o variantes no vistas",
    ],
    "shap_fn_analysis": "No aplica: no existen casos FN en el conjunto de prueba con el umbral seleccionado (0.05). El análisis SHAP local cubre TP×2, FP×2 y TN×1."
}

fn_path = os.path.join(OUTPUT_DIR, "fn_analysis_formal.json")
with open(fn_path, "w", encoding="utf-8") as f:
    json.dump(fn_report, f, indent=2, ensure_ascii=False)
print(f"  FN={fn} — documentado formalmente")
print(f"  Interpretación guardada: fn_analysis_formal.json")

# ─── Gap 3: Media y std de CV del mejor modelo ───────────────────────────────
print(f"\n[3] Validación cruzada del mejor modelo ({best_name})")
print("  Ejecutando cross_validate (puede tardar ~1 minuto)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_results = cross_validate(
    best_model,
    X_train, y_train,
    cv=cv,
    scoring={
        "recall":            "recall",
        "precision":         "precision",
        "f1_macro":          "f1_macro",
        "average_precision": "average_precision",
        "roc_auc":           "roc_auc",
    },
    return_train_score=True,
    n_jobs=-1,
)

cv_summary = {}
metricas_cv = ["recall", "precision", "f1_macro", "average_precision", "roc_auc"]
print(f"\n  {'Métrica':<25} {'Val mean':>10} {'Val std':>10} {'Train mean':>12}")
print("  " + "-" * 60)
for m in metricas_cv:
    val_mean  = cv_results[f"test_{m}"].mean()
    val_std   = cv_results[f"test_{m}"].std()
    train_mean = cv_results[f"train_{m}"].mean()
    print(f"  {m:<25} {val_mean:>10.4f} {val_std:>10.4f} {train_mean:>12.4f}")
    cv_summary[m] = {
        "val_mean":   round(float(val_mean), 4),
        "val_std":    round(float(val_std), 4),
        "train_mean": round(float(train_mean), 4),
        "gap":        round(float(train_mean - val_mean), 4),
        "fold_scores": [round(float(s), 4) for s in cv_results[f"test_{m}"]],
    }

# Diagnóstico de variabilidad
recall_std = cv_summary["recall"]["val_std"]
recall_gap = cv_summary["recall"]["gap"]
print(f"\n  Gap train-val (recall): {recall_gap:.4f}")
print(f"  Std val (recall):       {recall_std:.4f}")
if recall_gap > 0.15:
    diag = "OVERFITTING: gap train-val elevado"
elif recall_std > 0.05:
    diag = "ALTA VARIABILIDAD: desviación std elevada entre folds"
elif cv_summary["recall"]["val_mean"] < 0.5:
    diag = "UNDERFITTING: recall bajo en validación cruzada"
else:
    diag = "GENERALIZACIÓN ADECUADA: gap y variabilidad dentro de rangos normales"
print(f"  Diagnóstico: {diag}")

cv_summary["diagnostico"] = diag
cv_summary["modelo"] = best_name
cv_summary["n_folds"] = 5
cv_summary["conjunto"] = "train (98,349 filas)"

cv_path = os.path.join(OUTPUT_DIR, "cv_results_summary.json")
with open(cv_path, "w", encoding="utf-8") as f:
    json.dump(cv_summary, f, indent=2)
print(f"\n  Resultados CV guardados: cv_results_summary.json")

# Gráfico de scores por fold
fig, ax = plt.subplots(figsize=(10, 5))
folds = [f"Fold {i+1}" for i in range(5)]
x = np.arange(5)
width = 0.15
plot_metrics = ["recall", "precision", "f1_macro", "average_precision", "roc_auc"]
plot_colors  = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
plot_labels  = ["Recall", "Precision", "F1 macro", "PR-AUC", "ROC-AUC"]

for i, (m, color, label) in enumerate(zip(plot_metrics, plot_colors, plot_labels)):
    scores = cv_results[f"test_{m}"]
    ax.bar(x + i * width, scores, width, color=color, alpha=0.85, label=label)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(folds)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title(f"Scores por fold — Cross Validation ({best_name}, 5-fold StratifiedKFold)")
ax.legend(loc="lower right", fontsize=8)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
fig.tight_layout()
cv_fig_path = os.path.join(OUTPUT_DIR, "cv_scores_por_fold.png")
fig.savefig(cv_fig_path, dpi=150)
plt.close(fig)
print(f"  Gráfico por fold guardado: cv_scores_por_fold.png")
