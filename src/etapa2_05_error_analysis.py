"""
Etapa 2 – Paso 5: Análisis de errores, diagnóstico y artefactos para Etapa 3
Clasificación binaria: BENIGN (0) vs ATTACK (1)

Este script:
  1. Analiza FP y FN del mejor modelo
  2. Identifica casos límite (cerca del umbral)
  3. Diagnostica overfitting comparando train / val / test
  4. Genera curva de aprendizaje
  5. Guarda todos los artefactos necesarios para Etapa 3
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

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

# ─── Configuración ────────────────────────────────────────────────────────────
VARIANT      = "base"
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH    = os.path.join(BASE_DIR, "outputs", "etapa2", f"datos_limpios_{VARIANT}.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs", "etapa2")
ERROR_DIR    = os.path.join(OUTPUT_DIR, "error_analysis")
RANDOM_STATE = 42

os.makedirs(ERROR_DIR, exist_ok=True)

# ─── Carga ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ETAPA 2 – Análisis de errores + Artefactos Etapa 3")
print("=" * 60)

# Cargar split guardado
split_path = os.path.join(OUTPUT_DIR, "split_data.joblib")
if os.path.exists(split_path):
    split = joblib.load(split_path)
    X_train = split["X_train"]
    X_val   = split["X_val"]
    X_test  = split["X_test"]
    y_train = split["y_train"]
    y_val   = split["y_val"]
    y_test  = split["y_test"]
    feature_names = split["feature_names"]
else:
    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=["Label"])
    y  = df["Label"]
    feature_names = X.columns.tolist()
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE
    )

# Cargar modelo y umbral
model_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("mejor_modelo_") and f.endswith(".joblib")]
if model_files:
    best_model = joblib.load(os.path.join(OUTPUT_DIR, model_files[0]))
    best_name  = model_files[0].replace("mejor_modelo_", "").replace(f"_{VARIANT}.joblib", "")
else:
    print("Modelo no encontrado. Entrenando RF de respaldo...")
    best_name  = "RF"
    best_model = RandomForestClassifier(
        class_weight="balanced_subsample", n_estimators=200,
        max_depth=20, n_jobs=-1, random_state=RANDOM_STATE
    )
    best_model.fit(X_train, y_train)

thresh_path = os.path.join(OUTPUT_DIR, "selected_threshold.json")
if os.path.exists(thresh_path):
    with open(thresh_path) as f:
        thresh_data = json.load(f)
    selected_threshold = thresh_data["threshold"]
else:
    selected_threshold = 0.5

print(f"\nModelo: {best_name}  |  Umbral: {selected_threshold}")

# ─── Predicciones en test ─────────────────────────────────────────────────────
y_prob_test = best_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= selected_threshold).astype(int)
y_true_arr  = y_test.values

tp_mask = (y_true_arr == 1) & (y_pred_test == 1)
fp_mask = (y_true_arr == 0) & (y_pred_test == 1)
fn_mask = (y_true_arr == 1) & (y_pred_test == 0)
tn_mask = (y_true_arr == 0) & (y_pred_test == 0)

n_tp = tp_mask.sum()
n_fp = fp_mask.sum()
n_fn = fn_mask.sum()
n_tn = tn_mask.sum()

print(f"\nMatriz de confusión (test):")
print(f"  TP={n_tp:,}  FP={n_fp:,}  FN={n_fn:,}  TN={n_tn:,}")

# ─── Análisis de Falsos Negativos (FN) ───────────────────────────────────────
# FN = ataques clasificados como BENIGN → los más peligrosos
print("\n=== Análisis de Falsos Negativos (ataques no detectados) ===")

X_fn = X_test.iloc[fn_mask].copy()
X_fn["prob_ATTACK"]  = y_prob_test[fn_mask]
X_fn["true_label"]   = 1
X_fn["pred_label"]   = 0
X_fn["margin"]       = selected_threshold - y_prob_test[fn_mask]  # qué tan lejos del umbral

print(f"  Total FN: {len(X_fn):,}")
print(f"  Prob ATTACK promedio en FN: {X_fn['prob_ATTACK'].mean():.4f}")
print(f"  Prob ATTACK máxima en FN:   {X_fn['prob_ATTACK'].max():.4f}")
print(f"  FN cerca del umbral (margin <= 0.05): {(X_fn['margin'] <= 0.05).sum():,}")

# Estadísticas de variables clave en FN vs TP
print("\n  Comparación de medias FN vs TP en variables clave:")
key_vars = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s",
            "Total Fwd Packets", "Total Backward Packets"]
key_vars_present = [v for v in key_vars if v in X_test.columns]

X_tp = X_test.iloc[tp_mask].copy()
comparison_rows = []
for var in key_vars_present:
    fn_mean = X_fn[var].mean()
    tp_mean = X_tp[var].mean()
    comparison_rows.append({
        "Variable": var,
        "Media_FN": round(fn_mean, 2),
        "Media_TP": round(tp_mean, 2),
        "Ratio_FN_TP": round(fn_mean / tp_mean, 3) if tp_mean != 0 else None,
    })

df_fn_analysis = pd.DataFrame(comparison_rows)
print(df_fn_analysis.to_string(index=False))
df_fn_analysis.to_csv(os.path.join(ERROR_DIR, "fn_vs_tp_analysis.csv"), index=False)

# ─── Análisis de Falsos Positivos (FP) ───────────────────────────────────────
print("\n=== Análisis de Falsos Positivos (tráfico benigno clasificado como ataque) ===")

X_fp = X_test.iloc[fp_mask].copy()
X_fp["prob_ATTACK"] = y_prob_test[fp_mask]
X_fp["true_label"]  = 0
X_fp["pred_label"]  = 1
X_fp["margin"]      = y_prob_test[fp_mask] - selected_threshold

print(f"  Total FP: {len(X_fp):,}")
print(f"  Prob ATTACK promedio en FP: {X_fp['prob_ATTACK'].mean():.4f}")
print(f"  FP muy confiados (prob >= 0.8): {(X_fp['prob_ATTACK'] >= 0.8).sum():,}")
print(f"  FP cerca del umbral (margin <= 0.05): {(X_fp['margin'] <= 0.05).sum():,}")

# Distribución de FP por puerto de destino
if "Destination Port" in X_fp.columns:
    top_ports_fp = X_fp["Destination Port"].value_counts().head(10)
    print(f"\n  Top 10 puertos en FP:")
    print(top_ports_fp.to_string())

# ─── Casos límite ─────────────────────────────────────────────────────────────
print("\n=== Casos límite (P(ATTACK) cerca del umbral ±0.05) ===")
margin_total = np.abs(y_prob_test - selected_threshold)
borderline_mask = margin_total <= 0.05
n_borderline = borderline_mask.sum()
print(f"  Casos en zona límite: {n_borderline:,} ({n_borderline/len(y_test)*100:.2f}%)")

X_border = X_test.iloc[borderline_mask].copy()
X_border["prob_ATTACK"] = y_prob_test[borderline_mask]
X_border["true_label"]  = y_true_arr[borderline_mask]
X_border["pred_label"]  = y_pred_test[borderline_mask]
X_border.to_csv(os.path.join(ERROR_DIR, "casos_limite.csv"), index=False)
print(f"  Casos límite guardados: casos_limite.csv")

# ─── Diagnóstico overfitting: train vs val vs test ─────────────────────────────
print("\n=== Diagnóstico de Overfitting ===")

def metricas_conjunto(modelo, X, y, umbral, nombre):
    yp_prob = modelo.predict_proba(X)[:, 1]
    yp      = (yp_prob >= umbral).astype(int)
    return {
        "Conjunto":        nombre,
        "Recall_ATTACK":   round(recall_score(y, yp, zero_division=0), 4),
        "Precision_ATTACK":round(precision_score(y, yp, zero_division=0), 4),
        "F1_macro":        round(f1_score(y, yp, average="macro", zero_division=0), 4),
        "PR_AUC":          round(average_precision_score(y, yp_prob), 4),
        "ROC_AUC":         round(roc_auc_score(y, yp_prob), 4),
        "Accuracy":        round(accuracy_score(y, yp), 4),
    }

rows = [
    metricas_conjunto(best_model, X_train, y_train, selected_threshold, "Train"),
    metricas_conjunto(best_model, X_val,   y_val,   selected_threshold, "Validation"),
    metricas_conjunto(best_model, X_test,  y_test,  selected_threshold, "Test"),
]
df_overfitting = pd.DataFrame(rows)
print(df_overfitting.to_string(index=False))
df_overfitting.to_csv(os.path.join(ERROR_DIR, "diagnostico_overfitting.csv"), index=False)

# Interpretación automática
r_train = df_overfitting.loc[df_overfitting["Conjunto"]=="Train",  "Recall_ATTACK"].values[0]
r_val   = df_overfitting.loc[df_overfitting["Conjunto"]=="Validation","Recall_ATTACK"].values[0]
r_test  = df_overfitting.loc[df_overfitting["Conjunto"]=="Test",   "Recall_ATTACK"].values[0]
gap     = r_train - r_val

print(f"\n  Gap Train-Val (Recall ATTACK): {gap:.4f}")
if gap > 0.15:
    print("  → Posible OVERFITTING: rendimiento en train significativamente mayor que en val")
elif r_val < 0.50:
    print("  → Posible UNDERFITTING: recall en val menor a 0.50")
else:
    print("  → Generalización adecuada: gap train-val razonable")

# ─── Gráfico comparativo train / val / test ───────────────────────────────────
metrics_names = ["Recall_ATTACK", "Precision_ATTACK", "F1_macro", "PR_AUC", "ROC_AUC"]
x = np.arange(len(metrics_names))
width = 0.28
colors = {"Train": "#e74c3c", "Validation": "#f39c12", "Test": "#2ecc71"}

fig, ax = plt.subplots(figsize=(12, 6))
for i, row in df_overfitting.iterrows():
    conjunto = row["Conjunto"]
    vals = [row[m] for m in metrics_names]
    ax.bar(x + i * width, vals, width, label=conjunto, color=colors[conjunto], alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics_names, rotation=10, ha="right")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Valor")
ax.set_title(f"Diagnóstico: Train vs Validation vs Test\n{best_name}")
ax.legend()
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(ERROR_DIR, "diagnostico_overfitting.png"), dpi=150)
plt.close(fig)
print("\n  Gráfico guardado: diagnostico_overfitting.png")

# ─── Curva de aprendizaje ──────────────────────────────────────────────────────
print("\n=== Curva de aprendizaje ===")
print("  (puede tardar ~1-2 minutos)")

# Usar un subconjunto de features para velocidad si hay muchas
X_lc = pd.concat([X_train, X_val])
y_lc = pd.concat([y_train, y_val])

# Modelo simplificado para la curva (más rápido)
lc_model = RandomForestClassifier(
    class_weight="balanced_subsample",
    n_estimators=50,
    max_depth=10,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

train_sizes_abs, train_scores, val_scores = learning_curve(
    lc_model, X_lc, y_lc,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    scoring="average_precision",
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(train_sizes_abs, train_mean, "o-", color="#e74c3c", label="Train PR-AUC")
ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                alpha=0.15, color="#e74c3c")
ax.plot(train_sizes_abs, val_mean, "s-", color="#3498db", label="Val PR-AUC (CV)")
ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                alpha=0.15, color="#3498db")
ax.set_xlabel("Tamaño del conjunto de entrenamiento")
ax.set_ylabel("PR-AUC")
ax.set_title(f"Curva de Aprendizaje — {best_name}\n(métrica: PR-AUC, clase ATTACK)")
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(ERROR_DIR, "learning_curve.png"), dpi=150)
plt.close(fig)
print("  Guardado: learning_curve.png")

# ─── Guardar artefactos para Etapa 3 ─────────────────────────────────────────
print("\n=== Guardando artefactos para Etapa 3 ===")

# feature_names.json
fn_path = os.path.join(OUTPUT_DIR, "feature_names.json")
with open(fn_path, "w") as f:
    json.dump(feature_names, f, indent=2)
print(f"  feature_names.json guardado")

# label_mapping.json
lm_path = os.path.join(OUTPUT_DIR, "label_mapping.json")
with open(lm_path, "w") as f:
    json.dump({"0": "BENIGN", "1": "ATTACK"}, f)
print(f"  label_mapping.json guardado")

# model_metadata.json
meta = {
    "model_name": best_name,
    "variant": VARIANT,
    "threshold": selected_threshold,
    "n_features": len(feature_names),
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
    "attack_pct_train": float(y_train.mean()),
    "test_metrics": {
        "recall_attack": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "precision_attack": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred_test, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_test)),
        "pr_auc": float(average_precision_score(y_test, y_prob_test)),
    },
    "confusion_matrix": {
        "TP": int(n_tp), "FP": int(n_fp),
        "FN": int(n_fn), "TN": int(n_tn),
    },
}
meta_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  model_metadata.json guardado")

# Función de inferencia de ejemplo (para Etapa 3)
inference_example = os.path.join(OUTPUT_DIR, "inference_example.py")
with open(inference_example, "w", encoding="utf-8") as f:
    f.write(f'''"""
Función de inferencia para Etapa 3 — Agente inteligente
Uso: importar predict_flow y llamarla con un diccionario de features
"""
import json
import numpy as np
import pandas as pd
import joblib

MODEL_PATH     = "outputs/etapa2/mejor_modelo_{best_name}_{VARIANT}.joblib"
FEATURES_PATH  = "outputs/etapa2/feature_names.json"
THRESHOLD_PATH = "outputs/etapa2/selected_threshold.json"

_model     = joblib.load(MODEL_PATH)
_features  = json.load(open(FEATURES_PATH))
_threshold = json.load(open(THRESHOLD_PATH))["threshold"]

def predict_flow(flow_features: dict) -> dict:
    """
    Clasifica un flujo de red como BENIGN o ATTACK.

    Args:
        flow_features: diccionario con los valores de cada feature

    Returns:
        dict con:
            prediction:        "BENIGN" o "ATTACK"
            attack_probability: float entre 0 y 1
            threshold:         umbral usado
            confidence:        "HIGH" / "MEDIUM" / "LOW" segun distancia al umbral
    """
    df = pd.DataFrame([flow_features])[_features]
    prob = float(_model.predict_proba(df)[0, 1])
    pred = "ATTACK" if prob >= _threshold else "BENIGN"
    margin = abs(prob - _threshold)
    confidence = "HIGH" if margin > 0.2 else ("MEDIUM" if margin > 0.05 else "LOW")

    return {{
        "prediction":         pred,
        "attack_probability": round(prob, 4),
        "threshold":          _threshold,
        "confidence":         confidence,
    }}
''')
print(f"  inference_example.py guardado")

# ─── Resumen final ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN FINAL — ETAPA 2")
print("=" * 60)
print(f"\nModelo: {best_name}  |  Umbral: {selected_threshold}")
print(f"\nMétricas en TEST:")
for k, v in meta["test_metrics"].items():
    print(f"  {k}: {v:.4f}")
print(f"\nMatriz de confusión:")
print(f"  TP={n_tp}  FP={n_fp}  FN={n_fn}  TN={n_tn}")
print(f"\nArtefactos guardados en: {OUTPUT_DIR}")
print(f"\nEtapa 2 completada. Artefactos listos para Etapa 3.")
