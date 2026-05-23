"""
Etapa 2 – Paso 3: Baselines completos
Clasificación binaria: BENIGN (0) vs ATTACK (1)

Baselines implementados:
  1. DummyClassifier most_frequent  → siempre predice BENIGN
  2. DummyClassifier stratified     → predice aleatoriamente respetando distribución
  3. LogisticRegression balanceada  → baseline interpretable lineal

Todos los baselines usan el mismo split estratificado train/val/test (60/20/20).
Ninguna transformación se fitea fuera del conjunto de entrenamiento.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

# ─── Configuración ────────────────────────────────────────────────────────────
VARIANT      = "base"
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH    = os.path.join(BASE_DIR, "outputs", "etapa2", f"datos_limpios_{VARIANT}.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs", "etapa2")
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Carga ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ETAPA 2 – Baselines completos")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=["Label"])
y  = df["Label"]
feature_names = X.columns.tolist()

print(f"\nDataset: {X.shape[0]:,} filas × {X.shape[1]} features")
print(f"ATTACK: {y.sum():,} ({y.mean()*100:.2f}%)")

# ─── Partición train / val / test (60 / 20 / 20) ─────────────────────────────
# Split estratificado para preservar proporción de ATTACK en todos los conjuntos
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE
)

print(f"\nPartición estratificada:")
print(f"  Train: {len(X_train):,}  ATTACK={y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"  Val:   {len(X_val):,}   ATTACK={y_val.sum():,} ({y_val.mean()*100:.2f}%)")
print(f"  Test:  {len(X_test):,}  ATTACK={y_test.sum():,} ({y_test.mean()*100:.2f}%)")

# ─── Función de evaluación ────────────────────────────────────────────────────
def evaluar(nombre, modelo, X_te, y_te, umbral=0.5):
    y_pred = modelo.predict(X_te)
    # Para DummyClassifier stratified predict_proba puede no estar disponible
    try:
        y_prob = modelo.predict_proba(X_te)[:, 1]
        roc = round(roc_auc_score(y_te, y_prob), 4)
        pr  = round(average_precision_score(y_te, y_prob), 4)
    except Exception:
        roc = None
        pr  = None

    return {
        "Modelo":            nombre,
        "Recall_ATTACK":     round(recall_score(y_te, y_pred, zero_division=0), 4),
        "Precision_ATTACK":  round(precision_score(y_te, y_pred, zero_division=0), 4),
        "F1_macro":          round(f1_score(y_te, y_pred, average="macro", zero_division=0), 4),
        "Accuracy":          round(accuracy_score(y_te, y_pred), 4),
        "ROC_AUC":           roc,
        "PR_AUC":            pr,
        "Umbral":            umbral,
        "Observacion":       "",
    }

# ─── Definición de baselines ──────────────────────────────────────────────────
# Baseline 1: siempre predice la clase mayoritaria (BENIGN)
# Recall ATTACK = 0 porque nunca predice ATTACK
# Accuracy alta (~98%) pero engañosa — demuestra por qué accuracy no sirve aquí
baseline_dummy_mf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)

# Baseline 2: predice aleatoriamente respetando la distribución de clases
# Recall ATTACK muy bajo (~1.3%) porque casi nunca elige ATTACK
baseline_dummy_st = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)

# Baseline 3: regresión logística con pesos balanceados
# Este es el baseline interpretable — formalización del experimento preliminar de Etapa 1
# El StandardScaler se fitea solo sobre train (dentro del Pipeline)
baseline_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )),
])

baselines = {
    "Dummy_MostFrequent": (baseline_dummy_mf, "Baseline ingenuo: siempre predice BENIGN"),
    "Dummy_Stratified":   (baseline_dummy_st, "Baseline aleatorio: respeta distribución"),
    "LogReg_Balanced":    (baseline_lr,        "Baseline interpretable: frontera lineal"),
}

# ─── Entrenamiento y evaluación ───────────────────────────────────────────────
print("\n=== Entrenando Baselines ===")
resultados = []
modelos_fit = {}

for nombre, (modelo, descripcion) in baselines.items():
    print(f"\n  [{nombre}]")
    print(f"    Descripción: {descripcion}")

    modelo.fit(X_train, y_train)
    metricas = evaluar(nombre, modelo, X_test, y_test)
    metricas["Observacion"] = descripcion
    resultados.append(metricas)
    modelos_fit[nombre] = modelo

    print(f"    Recall ATTACK:    {metricas['Recall_ATTACK']}")
    print(f"    Precision ATTACK: {metricas['Precision_ATTACK']}")
    print(f"    F1 macro:         {metricas['F1_macro']}")
    print(f"    Accuracy:         {metricas['Accuracy']}  ← no usar como criterio principal")
    if metricas["PR_AUC"]:
        print(f"    PR-AUC:           {metricas['PR_AUC']}")
        print(f"    ROC-AUC:          {metricas['ROC_AUC']}")

# ─── Tabla de resultados ──────────────────────────────────────────────────────
print("\n=== Tabla Comparativa de Baselines ===")
df_res = pd.DataFrame(resultados)
print(df_res[["Modelo", "Recall_ATTACK", "Precision_ATTACK",
              "F1_macro", "PR_AUC", "ROC_AUC", "Accuracy"]].to_string(index=False))

csv_path = os.path.join(OUTPUT_DIR, "metricas_baselines.csv")
df_res.to_csv(csv_path, index=False)
print(f"\nMétricas guardadas: {csv_path}")

# ─── Matrices de confusión ────────────────────────────────────────────────────
print("\n=== Generando matrices de confusión ===")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (nombre, (modelo, _)) in zip(axes, baselines.items()):
    y_pred = modelo.predict(X_test)
    r = recall_score(y_test, y_pred, zero_division=0)
    p = precision_score(y_test, y_pred, zero_division=0)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        ax=ax,
        display_labels=["BENIGN", "ATTACK"],
        colorbar=False,
        cmap="Blues",
    )
    ax.set_title(f"{nombre}\nRecall={r:.3f}  Prec={p:.3f}", fontsize=9)

fig.suptitle("Matrices de Confusión — Baselines (Test set)", fontsize=12)
fig.tight_layout()
path_cm = os.path.join(OUTPUT_DIR, "confusion_matrices_baselines.png")
fig.savefig(path_cm, dpi=150)
plt.close(fig)
print(f"  Guardado: confusion_matrices_baselines.png")

# ─── Curvas PR y ROC para LogReg (único baseline con probabilidades útiles) ───
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
y_prob_lr = baseline_lr.predict_proba(X_test)[:, 1]

RocCurveDisplay.from_predictions(y_test, y_prob_lr, ax=axes[0], name="LogReg_Balanced")
axes[0].set_title("Curva ROC — LogReg Baseline")

PrecisionRecallDisplay.from_predictions(y_test, y_prob_lr, ax=axes[1], name="LogReg_Balanced")
axes[1].set_title("Curva Precision-Recall — LogReg Baseline")

fig.suptitle("Curvas de evaluación — Baseline LogReg", fontsize=12)
fig.tight_layout()
path_curves = os.path.join(OUTPUT_DIR, "curvas_baseline_logreg.png")
fig.savefig(path_curves, dpi=150)
plt.close(fig)
print(f"  Guardado: curvas_baseline_logreg.png")

# ─── Gráfico comparativo de métricas ─────────────────────────────────────────
metricas_plot = ["Recall_ATTACK", "Precision_ATTACK", "F1_macro"]
df_plot = df_res[["Modelo"] + metricas_plot].copy()
df_plot = df_plot.fillna(0)

x = np.arange(len(df_plot))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#e74c3c", "#3498db", "#2ecc71"]
for i, m in enumerate(metricas_plot):
    ax.bar(x + i * width, df_plot[m], width, label=m, color=colors[i], alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(df_plot["Modelo"], rotation=10, ha="right")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Valor")
ax.set_title("Comparación de métricas — Baselines")
ax.legend()
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
fig.tight_layout()
path_comp = os.path.join(OUTPUT_DIR, "comparacion_baselines.png")
fig.savefig(path_comp, dpi=150)
plt.close(fig)
print(f"  Guardado: comparacion_baselines.png")

# ─── Classification reports ───────────────────────────────────────────────────
report_path = os.path.join(OUTPUT_DIR, "classification_reports_baselines.txt")
with open(report_path, "w", encoding="utf-8") as f:
    for nombre, (modelo, descripcion) in baselines.items():
        yp = modelo.predict(X_test)
        cr = classification_report(y_test, yp,
                                   target_names=["BENIGN", "ATTACK"],
                                   zero_division=0)
        f.write(f"{'='*55}\n{nombre} — {descripcion}\n{'='*55}\n{cr}\n\n")
print(f"  Classification reports guardados: {report_path}")

# ─── Guardar split para reutilizar en scripts siguientes ─────────────────────
import joblib
split_data = {
    "X_train": X_train, "X_val": X_val, "X_test": X_test,
    "y_train": y_train, "y_val": y_val, "y_test": y_test,
    "feature_names": feature_names,
}
split_path = os.path.join(OUTPUT_DIR, "split_data.joblib")
joblib.dump(split_data, split_path)
print(f"\nSplit guardado para reutilización: {split_path}")

print("\n=== Interpretación de resultados ===")
print("""
  Dummy_MostFrequent:
    Recall ATTACK = 0.0  → nunca detecta ataques
    Accuracy alta (~98%) → engañosa por desbalance
    Confirma que accuracy NO es métrica válida aquí

  Dummy_Stratified:
    Recall ATTACK muy bajo (~1%) → detección casi nula
    Establece el piso aleatorio para comparación

  LogReg_Balanced:
    Recall ATTACK alto → detecta la mayoría de ataques
    Precision baja → genera falsas alarmas
    Es el punto de comparación real para modelos no lineales
""")

print("Baselines completados exitosamente.")
