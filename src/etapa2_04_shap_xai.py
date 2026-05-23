"""
Etapa 2 – Paso 4: XAI con SHAP + Selección de umbral
Clasificación binaria: BENIGN (0) vs ATTACK (1)

Este script:
  1. Carga el mejor modelo tuned del paso 2 (RF o HGB)
  2. Selecciona el umbral óptimo sobre el conjunto de validación
  3. Genera SHAP global: bar plot + beeswarm
  4. Genera SHAP local para casos TP, FP, FN, TN (waterfall plots)
  5. Guarda el umbral seleccionado para Etapa 3
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
import shap

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score,
    PrecisionRecallDisplay, RocCurveDisplay,
    confusion_matrix, ConfusionMatrixDisplay,
)

# ─── Configuración ────────────────────────────────────────────────────────────
VARIANT      = "base"
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH    = os.path.join(BASE_DIR, "outputs", "etapa2", f"datos_limpios_{VARIANT}.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs", "etapa2")
SHAP_DIR     = os.path.join(OUTPUT_DIR, "shap")
RANDOM_STATE = 42
SHAP_SAMPLE  = 500   # muestras para SHAP (balance entre velocidad y representatividad)

os.makedirs(SHAP_DIR, exist_ok=True)

# ─── Carga del dataset y split ────────────────────────────────────────────────
print("=" * 60)
print("ETAPA 2 – XAI con SHAP + Selección de umbral")
print("=" * 60)

# Intentar cargar el split guardado por el paso 3
split_path = os.path.join(OUTPUT_DIR, "split_data.joblib")
if os.path.exists(split_path):
    print("\nCargando split guardado por etapa2_03...")
    split = joblib.load(split_path)
    X_train = split["X_train"]
    X_val   = split["X_val"]
    X_test  = split["X_test"]
    y_train = split["y_train"]
    y_val   = split["y_val"]
    y_test  = split["y_test"]
    feature_names = split["feature_names"]
else:
    print("\nSplit no encontrado. Reconstruyendo desde CSV...")
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

print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# ─── Cargar o entrenar el modelo principal ────────────────────────────────────
# Buscar el mejor modelo serializado del paso 2
model_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("mejor_modelo_") and f.endswith(".joblib")]

if model_files:
    model_path = os.path.join(OUTPUT_DIR, model_files[0])
    print(f"\nCargando modelo: {model_files[0]}")
    best_model = joblib.load(model_path)
    best_name  = model_files[0].replace("mejor_modelo_", "").replace(f"_{VARIANT}.joblib", "")
else:
    # Si no existe el modelo del paso 2, entrenar Random Forest directamente
    print("\nModelo del paso 2 no encontrado. Entrenando Random Forest...")
    best_name  = "RF"
    best_model = RandomForestClassifier(
        class_weight="balanced_subsample",
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    best_model.fit(X_train, y_train)
    model_path = os.path.join(OUTPUT_DIR, f"mejor_modelo_{best_name}_{VARIANT}.joblib")
    joblib.dump(best_model, model_path)
    print(f"  Random Forest entrenado y guardado: {model_path}")

print(f"  Modelo activo: {best_name}")

# ─── Selección de umbral sobre VALIDACIÓN ────────────────────────────────────
# No se usa test aquí — el umbral se selecciona sobre val y se aplica una vez en test
print("\n=== Selección de umbral sobre conjunto de validación ===")
print("  (test permanece ciego hasta la evaluación final)")

y_prob_val = best_model.predict_proba(X_val)[:, 1]

# Evaluar umbrales de 0.05 a 0.95
thresholds = np.arange(0.05, 0.96, 0.01)
results_thresh = []
for t in thresholds:
    y_pred_t = (y_prob_val >= t).astype(int)
    r = recall_score(y_val, y_pred_t, zero_division=0)
    p = precision_score(y_val, y_pred_t, zero_division=0)
    f = f1_score(y_val, y_pred_t, average="macro", zero_division=0)
    results_thresh.append({"threshold": round(t, 2), "recall": r, "precision": p, "f1_macro": f})

df_thresh = pd.DataFrame(results_thresh)

# Criterio: maximizar recall manteniendo precision >= 0.20
# En detección de intrusiones es más costoso perder un ataque que tener falsas alarmas
# pero precision < 0.20 haría el sistema inútil operativamente
candidates = df_thresh[df_thresh["precision"] >= 0.20]
if len(candidates) == 0:
    candidates = df_thresh  # fallback si ninguno cumple el criterio

# Seleccionar el umbral que maximiza recall entre los candidatos
best_thresh_row = candidates.loc[candidates["recall"].idxmax()]
selected_threshold = float(best_thresh_row["threshold"])

print(f"\n  Umbral seleccionado: {selected_threshold}")
print(f"  Recall ATTACK (val):    {best_thresh_row['recall']:.4f}")
print(f"  Precision ATTACK (val): {best_thresh_row['precision']:.4f}")
print(f"  F1 macro (val):         {best_thresh_row['f1_macro']:.4f}")
print(f"  Criterio: máximo recall con precision >= 0.20")

# Guardar umbral para Etapa 3
threshold_path = os.path.join(OUTPUT_DIR, "selected_threshold.json")
with open(threshold_path, "w") as f:
    json.dump({
        "threshold": selected_threshold,
        "model": best_name,
        "variant": VARIANT,
        "criterion": "max_recall_with_precision_gte_0.20",
        "recall_val": float(best_thresh_row["recall"]),
        "precision_val": float(best_thresh_row["precision"]),
        "f1_macro_val": float(best_thresh_row["f1_macro"]),
    }, f, indent=2)
print(f"  Umbral guardado: {threshold_path}")

# Curva Precision-Recall con umbral marcado
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

PrecisionRecallDisplay.from_predictions(y_val, y_prob_val, ax=axes[0], name=best_name)
axes[0].axvline(x=best_thresh_row["recall"], color="red", linestyle="--", alpha=0.7,
                label=f"Umbral={selected_threshold}")
axes[0].set_title("Curva Precision-Recall — Validación\nUmbral seleccionado marcado en rojo")
axes[0].legend()

axes[1].plot(df_thresh["threshold"], df_thresh["recall"], label="Recall ATTACK", color="#e74c3c")
axes[1].plot(df_thresh["threshold"], df_thresh["precision"], label="Precision ATTACK", color="#3498db")
axes[1].plot(df_thresh["threshold"], df_thresh["f1_macro"], label="F1 macro", color="#2ecc71")
axes[1].axvline(x=selected_threshold, color="black", linestyle="--", alpha=0.7,
                label=f"Umbral={selected_threshold}")
axes[1].set_xlabel("Umbral")
axes[1].set_ylabel("Valor")
axes[1].set_title("Métricas vs Umbral — Validación")
axes[1].legend()
axes[1].set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "threshold_selection.png"), dpi=150)
plt.close(fig)
print(f"  Gráfico guardado: threshold_selection.png")

# ─── Evaluación final en TEST con umbral seleccionado ─────────────────────────
print("\n=== Evaluación final en TEST (primera y única vez) ===")
y_prob_test = best_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= selected_threshold).astype(int)

recall_test    = recall_score(y_test, y_pred_test, zero_division=0)
precision_test = precision_score(y_test, y_pred_test, zero_division=0)
f1_test        = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
roc_test       = roc_auc_score(y_test, y_prob_test)
pr_test        = average_precision_score(y_test, y_prob_test)

print(f"  Recall ATTACK:    {recall_test:.4f}")
print(f"  Precision ATTACK: {precision_test:.4f}")
print(f"  F1 macro:         {f1_test:.4f}")
print(f"  ROC-AUC:          {roc_test:.4f}")
print(f"  PR-AUC:           {pr_test:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
RocCurveDisplay.from_predictions(y_test, y_prob_test, ax=axes[0], name=best_name)
axes[0].set_title(f"Curva ROC — Test\n{best_name} (AUC={roc_test:.4f})")
PrecisionRecallDisplay.from_predictions(y_test, y_prob_test, ax=axes[1], name=best_name)
axes[1].set_title(f"Curva PR — Test\n{best_name} (AP={pr_test:.4f})")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "curvas_test_final.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_test, ax=ax,
    display_labels=["BENIGN", "ATTACK"], cmap="Blues", colorbar=False
)
ax.set_title(f"{best_name} — Test\nUmbral={selected_threshold}")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_test_final.png"), dpi=150)
plt.close(fig)
print("  Curvas y matriz de confusión guardadas.")

# ─── SHAP Global ──────────────────────────────────────────────────────────────
print(f"\n=== SHAP Global ({SHAP_SAMPLE} muestras de test) ===")

# Muestra representativa para SHAP
np.random.seed(RANDOM_STATE)
idx_sample = np.random.choice(len(X_test), size=min(SHAP_SAMPLE, len(X_test)), replace=False)
X_shap = X_test.iloc[idx_sample].reset_index(drop=True)

# Determinar tipo de explainer según el modelo
model_type = type(best_model).__name__
print(f"  Tipo de modelo: {model_type}")

if model_type == "Pipeline":
    # Si es Pipeline (ej. LogReg), extraer el clasificador
    clf_inner = best_model.named_steps.get("clf", best_model[-1])
    X_shap_transformed = best_model[:-1].transform(X_shap)
    explainer = shap.LinearExplainer(clf_inner, X_shap_transformed)
    shap_values = explainer.shap_values(X_shap_transformed)
    X_shap_display = pd.DataFrame(X_shap_transformed, columns=feature_names)
elif model_type in ("RandomForestClassifier", "DecisionTreeClassifier"):
    explainer = shap.TreeExplainer(best_model)
    shap_values_obj = explainer(X_shap)
    # Para clasificación binaria: tomar valores de clase ATTACK (índice 1)
    if hasattr(shap_values_obj, 'values') and shap_values_obj.values.ndim == 3:
        shap_values = shap_values_obj.values[:, :, 1]
        shap_base   = shap_values_obj.base_values[:, 1] if shap_values_obj.base_values.ndim > 1 else shap_values_obj.base_values
    else:
        shap_values = shap_values_obj.values
        shap_base   = shap_values_obj.base_values
    X_shap_display = X_shap
else:
    # HistGradientBoosting u otros: usar KernelExplainer con muestra pequeña
    print("  Usando KernelExplainer (más lento, modelo no tree-based directo)...")
    background = shap.sample(X_train, 100, random_state=RANDOM_STATE)
    explainer = shap.KernelExplainer(
        lambda x: best_model.predict_proba(pd.DataFrame(x, columns=feature_names))[:, 1],
        background
    )
    shap_values = explainer.shap_values(X_shap.values[:50])  # reducir para velocidad
    X_shap_display = X_shap.iloc[:50]
    shap_base = None

# SHAP Bar Plot — importancia global media
plt.figure(figsize=(10, 8))
mean_abs_shap = np.abs(shap_values).mean(axis=0)
idx_top = np.argsort(mean_abs_shap)[-20:]
fn = [feature_names[i] for i in idx_top]
vals = mean_abs_shap[idx_top]
plt.barh(fn, vals, color="steelblue")
plt.xlabel("mean |SHAP value|")
plt.title(f"SHAP — Top 20 Features más influyentes\n{best_name} (clase ATTACK)")
plt.tight_layout()
plt.savefig(os.path.join(SHAP_DIR, "shap_bar_global.png"), dpi=150)
plt.close()
print("  Guardado: shap_bar_global.png")

# SHAP Beeswarm (summary plot)
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_shap_display,
    feature_names=feature_names,
    show=False,
    max_display=20,
)
plt.title(f"SHAP Beeswarm — {best_name} (clase ATTACK)")
plt.tight_layout()
plt.savefig(os.path.join(SHAP_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Guardado: shap_beeswarm.png")

# ─── SHAP Local — casos TP, FP, FN, TN ───────────────────────────────────────
print("\n=== SHAP Local — análisis de casos individuales ===")

# Identificar casos en el conjunto de test completo
y_pred_full = (best_model.predict_proba(X_test)[:, 1] >= selected_threshold).astype(int)
y_true_arr  = y_test.values

tp_idx = np.where((y_true_arr == 1) & (y_pred_full == 1))[0]
fp_idx = np.where((y_true_arr == 0) & (y_pred_full == 1))[0]
fn_idx = np.where((y_true_arr == 1) & (y_pred_full == 0))[0]
tn_idx = np.where((y_true_arr == 0) & (y_pred_full == 0))[0]

print(f"  TP: {len(tp_idx):,}  FP: {len(fp_idx):,}  FN: {len(fn_idx):,}  TN: {len(tn_idx):,}")

# Selección de casos representativos
# TP: los más confiados (mayor probabilidad ATTACK)
# FN: los más peligrosos (menor probabilidad ATTACK — ataques casi no detectados)
# FP: los más "atacantes" en apariencia
# TN: un caso benigno bien clasificado
y_prob_full = best_model.predict_proba(X_test)[:, 1]

cases = {}
if len(tp_idx) >= 2:
    top2_tp = tp_idx[np.argsort(y_prob_full[tp_idx])[-2:]]
    cases["TP"] = list(top2_tp)
if len(fp_idx) >= 2:
    top2_fp = fp_idx[np.argsort(y_prob_full[fp_idx])[-2:]]
    cases["FP"] = list(top2_fp)
if len(fn_idx) >= 2:
    top2_fn = fn_idx[np.argsort(y_prob_full[fn_idx])[:2]]  # los de menor prob
    cases["FN"] = list(top2_fn)
if len(tn_idx) >= 1:
    cases["TN"] = [tn_idx[np.argmax(y_prob_full[tn_idx])]]  # más cercano al umbral

# Generar SHAP para todos los casos locales
all_local_idx = []
for idxs in cases.values():
    all_local_idx.extend(idxs)
all_local_idx = list(set(all_local_idx))

X_local = X_test.iloc[all_local_idx].reset_index(drop=True)

if model_type in ("RandomForestClassifier", "DecisionTreeClassifier"):
    shap_local_obj = explainer(X_local)
    if hasattr(shap_local_obj, 'values') and shap_local_obj.values.ndim == 3:
        sv_local  = shap_local_obj.values[:, :, 1]
        bv_local  = shap_local_obj.base_values[:, 1] if shap_local_obj.base_values.ndim > 1 else shap_local_obj.base_values
    else:
        sv_local = shap_local_obj.values
        bv_local = shap_local_obj.base_values
else:
    sv_local = np.array([explainer.shap_values(X_local.values[i:i+1])[0]
                         for i in range(len(X_local))])
    bv_local = np.array([explainer.expected_value] * len(X_local))

# Waterfall plots por caso
local_report = []
for case_type, idxs in cases.items():
    for rank, idx in enumerate(idxs):
        local_pos = all_local_idx.index(idx)
        sv_i = sv_local[local_pos]
        bv_i = float(bv_local[local_pos]) if hasattr(bv_local, '__len__') else float(bv_local)
        prob_i = float(y_prob_full[idx])
        true_i = int(y_true_arr[idx])
        pred_i = int(y_pred_full[idx])

        # Top features para este caso
        top_pos = np.argsort(sv_i)[-5:][::-1]
        top_neg = np.argsort(sv_i)[:5]

        local_report.append({
            "Caso": f"{case_type}_{rank+1}",
            "Tipo": case_type,
            "Etiqueta_real": "ATTACK" if true_i == 1 else "BENIGN",
            "Prediccion": "ATTACK" if pred_i == 1 else "BENIGN",
            "Prob_ATTACK": round(prob_i, 4),
            "Umbral": selected_threshold,
            "Top_features_hacia_ATTACK": [(feature_names[i], round(sv_i[i], 4)) for i in top_pos],
            "Top_features_hacia_BENIGN": [(feature_names[i], round(sv_i[i], 4)) for i in top_neg],
        })

        # Waterfall plot manual
        fig, ax = plt.subplots(figsize=(9, 6))
        top10_idx = np.argsort(np.abs(sv_i))[-10:]
        top10_names = [feature_names[i] for i in top10_idx]
        top10_vals  = sv_i[top10_idx]
        colors_bar  = ["#e74c3c" if v > 0 else "#3498db" for v in top10_vals]
        ax.barh(top10_names, top10_vals, color=colors_bar)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impacto en P(ATTACK))")
        ax.set_title(
            f"SHAP Local — {case_type} #{rank+1}\n"
            f"Real={true_i}  Pred={pred_i}  P(ATTACK)={prob_i:.3f}  Umbral={selected_threshold}"
        )
        fig.tight_layout()
        out_path = os.path.join(SHAP_DIR, f"shap_local_{case_type}_{rank+1}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Guardado: shap_local_{case_type}_{rank+1}.png")

# Guardar reporte local en JSON
local_report_path = os.path.join(SHAP_DIR, "shap_local_report.json")
with open(local_report_path, "w", encoding="utf-8") as f:
    json.dump(local_report, f, indent=2, ensure_ascii=False)
print(f"\n  Reporte SHAP local guardado: shap_local_report.json")

# Imprimir resumen del reporte local
print("\n=== Resumen de casos SHAP locales ===")
for caso in local_report:
    print(f"\n  [{caso['Caso']}] Real={caso['Etiqueta_real']}  Pred={caso['Prediccion']}  "
          f"P(ATTACK)={caso['Prob_ATTACK']}  Umbral={caso['Umbral']}")
    print(f"    → hacia ATTACK: {caso['Top_features_hacia_ATTACK'][:3]}")
    print(f"    → hacia BENIGN: {caso['Top_features_hacia_BENIGN'][:3]}")

print("\nXAI completado exitosamente.")
