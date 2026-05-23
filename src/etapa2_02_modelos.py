"""
Etapa 2 – Paso 2: Modelos supervisados, optimización y evaluación
Clasificación binaria: BENIGN (0) vs ATTACK (1)
Métrica principal: Recall de la clase ATTACK
"""

import os
import sys
import warnings
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,  # soporta class_weight nativo
)
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ─── Configuración ────────────────────────────────────────────────────────────
VARIANT     = "base"          # "base" o "ablacion"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH   = os.path.join(BASE_DIR, "outputs", "etapa2", f"datos_limpios_{VARIANT}.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "etapa2")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
N_ITER       = 15             # iteraciones RandomizedSearchCV por modelo

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Carga ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"ETAPA 2 – Modelos supervisados  [variante: {VARIANT}]")
print("=" * 60)
print(f"\nCargando {DATA_PATH} ...")

df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=["Label"])
y  = df["Label"]
feature_names = X.columns.tolist()

n_attack = y.sum()
print(f"  Shape: {X.shape}  |  ATTACK: {n_attack:,} ({n_attack/len(y)*100:.2f}%)")

# ─── Train / Val / Test split estratificado (60 / 20 / 20) ───────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE
)
total = len(X_train) + len(X_val) + len(X_test)
print(f"  Train: {len(X_train):,} ({len(X_train)/total*100:.0f}%)  ATTACK={y_train.sum():,}")
print(f"  Val:   {len(X_val):,}  ({len(X_val)/total*100:.0f}%)  ATTACK={y_val.sum():,}")
print(f"  Test:  {len(X_test):,}  ({len(X_test)/total*100:.0f}%)  ATTACK={y_test.sum():,}")
print(f"  Test permanece ciego hasta evaluación final")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ─── Función de evaluación ────────────────────────────────────────────────────
def evaluar(nombre, modelo, X_te, y_te):
    y_pred = modelo.predict(X_te)
    y_prob = modelo.predict_proba(X_te)[:, 1]
    return {
        "Modelo":           nombre,
        "Recall_ATTACK":    round(recall_score(y_te, y_pred),                      4),
        "Precision_ATTACK": round(precision_score(y_te, y_pred, zero_division=0),   4),
        "F1_macro":         round(f1_score(y_te, y_pred, average="macro"),          4),
        "ROC_AUC":          round(roc_auc_score(y_te, y_prob),                      4),
        "PR_AUC":           round(average_precision_score(y_te, y_prob),            4),
    }

# ─── Definición de modelos ────────────────────────────────────────────────────
modelos_def = {
    "LR": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        )),
    ]),
    "DT": DecisionTreeClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    ),
    "RF": RandomForestClassifier(
        class_weight="balanced", n_estimators=100,
        n_jobs=-1, random_state=RANDOM_STATE
    ),
    "HGB": HistGradientBoostingClassifier(
        class_weight="balanced", max_iter=100, random_state=RANDOM_STATE
    ),
}

# Grillas de hiperparámetros para RandomizedSearchCV
param_grids = {
    "LR": {
        "clf__C":       [0.001, 0.01, 0.1, 1, 10, 100],
        "clf__solver":  ["lbfgs", "saga"],
        "clf__penalty": ["l2"],
    },
    "DT": {
        "max_depth":         [None, 5, 10, 20, 30],
        "min_samples_leaf":  [1, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion":         ["gini", "entropy"],
    },
    "RF": {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_leaf":  [1, 5, 10],
        "max_features":      ["sqrt", "log2"],
    },
    "HGB": {
        "max_iter":          [100, 200, 300],
        "max_depth":         [None, 5, 10, 20],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "min_samples_leaf":  [10, 20, 30, 50],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
}

colores = {"LR": "#1f77b4", "DT": "#ff7f0e", "RF": "#2ca02c", "HGB": "#d62728"}

# ─── Entrenamiento baseline ───────────────────────────────────────────────────
print("\n=== Modelos Baseline ===")
resultados   = []
modelos_fit  = {}

for nombre, modelo in modelos_def.items():
    print(f"  [{nombre}] entrenando baseline ...")
    modelo.fit(X_train, y_train)
    metricas = evaluar(f"{nombre}_base", modelo, X_test, y_test)
    resultados.append(metricas)
    modelos_fit[f"{nombre}_base"] = modelo
    print(f"    Recall ATTACK: {metricas['Recall_ATTACK']}  |  "
          f"PR-AUC: {metricas['PR_AUC']}  |  ROC-AUC: {metricas['ROC_AUC']}")

# ─── Búsqueda de hiperparámetros ──────────────────────────────────────────────
print(f"\n=== Búsqueda de Hiperparámetros (RandomizedSearchCV, n_iter={N_ITER}, cv={CV_FOLDS}) ===")
mejores_modelos = {}

for nombre, modelo_base in modelos_def.items():
    print(f"  [{nombre}] buscando hiperparámetros ...")

    # Instancia fresca con configuración base
    if nombre == "LR":
        estimador = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
            )),
        ])
    elif nombre == "DT":
        estimador = DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        )
    elif nombre == "RF":
        estimador = RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        )
    else:  # HGB
        estimador = HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        )

    search = RandomizedSearchCV(
        estimador,
        param_grids[nombre],
        n_iter=N_ITER,
        scoring="recall",   # optimizar recall de ATTACK (clase positiva = 1)
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    mejores_modelos[nombre] = best
    metricas = evaluar(f"{nombre}_tuned", best, X_test, y_test)
    resultados.append(metricas)

    print(f"    Mejores params: {search.best_params_}")
    print(f"    Recall ATTACK: {metricas['Recall_ATTACK']}  |  "
          f"PR-AUC: {metricas['PR_AUC']}  |  ROC-AUC: {metricas['ROC_AUC']}")

# ─── Tabla de resultados ──────────────────────────────────────────────────────
print("\n=== Tabla de Resultados ===")
df_res = pd.DataFrame(resultados)
df_res = df_res.sort_values("Recall_ATTACK", ascending=False).reset_index(drop=True)
print(df_res.to_string(index=False))

csv_path = os.path.join(OUTPUT_DIR, f"metricas_modelos_{VARIANT}.csv")
df_res.to_csv(csv_path, index=False)
print(f"\nMétricas guardadas: {csv_path}")
# Guardar split para reutilización en scripts 03-05
split_data = {
    "X_train": X_train, "X_val": X_val, "X_test": X_test,
    "y_train": y_train, "y_val": y_val, "y_test": y_test,
    "feature_names": feature_names,
}
split_path_out = os.path.join(OUTPUT_DIR, "split_data.joblib")
joblib.dump(split_data, split_path_out)
print(f"Split 60/20/20 guardado: {split_path_out}")

# ─── Gráficos comparativos ────────────────────────────────────────────────────
print("\n=== Generando gráficos ===")

# Curvas ROC
fig, ax = plt.subplots(figsize=(8, 6))
for nombre, modelo in mejores_modelos.items():
    y_prob = modelo.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(
        y_test, y_prob, ax=ax, name=nombre, color=colores[nombre]
    )
ax.set_title(f"Curvas ROC – Modelos Tuned ({VARIANT})")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"roc_curves_{VARIANT}.png"), dpi=150)
plt.close(fig)
print("  roc_curves guardado")

# Curvas Precision-Recall
fig, ax = plt.subplots(figsize=(8, 6))
for nombre, modelo in mejores_modelos.items():
    y_prob = modelo.predict_proba(X_test)[:, 1]
    PrecisionRecallDisplay.from_predictions(
        y_test, y_prob, ax=ax, name=nombre, color=colores[nombre]
    )
ax.set_title(f"Curvas Precision-Recall – Modelos Tuned ({VARIANT})")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"pr_curves_{VARIANT}.png"), dpi=150)
plt.close(fig)
print("  pr_curves guardado")

# Matrices de confusión (2×2 grid)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (nombre, modelo) in zip(axes.flatten(), mejores_modelos.items()):
    y_pred = modelo.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        ax=ax,
        display_labels=["BENIGN", "ATTACK"],
        colorbar=False,
        cmap="Blues",
    )
    r = recall_score(y_test, y_pred)
    p = precision_score(y_test, y_pred, zero_division=0)
    ax.set_title(f"{nombre}  (Recall={r:.3f}, Prec={p:.3f})")
fig.suptitle(f"Matrices de Confusión – Modelos Tuned ({VARIANT})", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrices_{VARIANT}.png"), dpi=150)
plt.close(fig)
print("  confusion_matrices guardado")

# Barras comparativas de métricas clave (solo tuned)
df_tuned = df_res[df_res["Modelo"].str.endswith("_tuned")].copy()
df_tuned["nombre_corto"] = df_tuned["Modelo"].str.replace("_tuned", "")
metricas_plot = ["Recall_ATTACK", "Precision_ATTACK", "F1_macro", "PR_AUC", "ROC_AUC"]
x = np.arange(len(df_tuned))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 6))
for i, m in enumerate(metricas_plot):
    ax.bar(x + i * width, df_tuned[m], width, label=m)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(df_tuned["nombre_corto"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Valor")
ax.set_title(f"Comparación de métricas – Modelos Tuned ({VARIANT})")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"comparacion_metricas_{VARIANT}.png"), dpi=150)
plt.close(fig)
print("  comparacion_metricas guardado")

# ─── Importancia de características ──────────────────────────────────────────
print("\n=== Importancia de Características ===")
TOP_N = 20

def plot_feature_importance(nombre, importancias, feature_names, top_n=TOP_N, subtitulo=""):
    idx_sorted = np.argsort(importancias)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(top_n), importancias[idx_sorted], color=colores.get(nombre, "steelblue"))
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx_sorted], fontsize=9)
    ax.set_xlabel("Importancia")
    title = f"Top {top_n} Features – {nombre}"
    if subtitulo:
        title += f" ({subtitulo})"
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"feat_importance_{nombre}_{VARIANT}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

for nombre, modelo in mejores_modelos.items():
    if nombre == "LR":
        coefs = np.abs(modelo.named_steps["clf"].coef_[0])
        path = plot_feature_importance(nombre, coefs, feature_names, subtitulo="|coef|")
    elif nombre in ("DT", "RF"):
        fi = modelo.feature_importances_
        path = plot_feature_importance(nombre, fi, feature_names, subtitulo="gini/gain")
    else:
        # HGB no expone feature_importances_ directamente; usar permutation importance
        perm_fi = permutation_importance(
            modelo, X_test, y_test, scoring="recall",
            n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
        )
        path = plot_feature_importance(nombre, perm_fi.importances_mean,
                                       feature_names, subtitulo="permutation")
    print(f"  [{nombre}] guardado: {os.path.basename(path)}")

# ─── Permutation importance del mejor modelo (mayor Recall ATTACK tuned) ──────
print("\n  Calculando Permutation Importance del mejor modelo ...")
best_row = df_tuned.loc[df_tuned["Recall_ATTACK"].idxmax()]
best_name = best_row["nombre_corto"]
best_model = mejores_modelos[best_name]

print(f"  Mejor modelo (Recall ATTACK): {best_name}  →  {best_row['Recall_ATTACK']}")

perm = permutation_importance(
    best_model, X_test, y_test,
    scoring="recall",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
idx_perm = np.argsort(perm.importances_mean)[-TOP_N:]
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(
    range(TOP_N),
    perm.importances_mean[idx_perm],
    xerr=perm.importances_std[idx_perm],
    color="steelblue",
)
ax.set_yticks(range(TOP_N))
ax.set_yticklabels([feature_names[i] for i in idx_perm], fontsize=9)
ax.set_xlabel("Permutation Importance (Δ Recall ATTACK)")
ax.set_title(f"Top {TOP_N} Features – Permutation Importance – {best_name} ({VARIANT})")
fig.tight_layout()
perm_path = os.path.join(OUTPUT_DIR, f"permutation_importance_{best_name}_{VARIANT}.png")
fig.savefig(perm_path, dpi=150)
plt.close(fig)
print(f"  Permutation importance guardado: {os.path.basename(perm_path)}")

# ─── Classification report del mejor modelo ───────────────────────────────────
y_pred_best = best_model.predict(X_test)
report_txt = classification_report(y_test, y_pred_best, target_names=["BENIGN", "ATTACK"])
report_path = os.path.join(OUTPUT_DIR, f"classification_report_{best_name}_{VARIANT}.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Mejor modelo: {best_name}  [variante: {VARIANT}]\n\n")
    f.write(report_txt)
print(f"\nClassification report ({best_name}):\n{report_txt}")

# Guardar classification report de todos los modelos tuned
all_reports_path = os.path.join(OUTPUT_DIR, f"classification_reports_all_{VARIANT}.txt")
with open(all_reports_path, "w", encoding="utf-8") as f:
    for nombre, modelo in mejores_modelos.items():
        yp = modelo.predict(X_test)
        cr = classification_report(y_test, yp, target_names=["BENIGN", "ATTACK"])
        f.write(f"{'='*50}\nModelo: {nombre}_tuned\n{'='*50}\n{cr}\n\n")
print(f"Reportes completos guardados: {all_reports_path}")

# ─── Guardar mejor modelo serializado ─────────────────────────────────────────
model_path = os.path.join(OUTPUT_DIR, f"mejor_modelo_{best_name}_{VARIANT}.joblib")
joblib.dump(best_model, model_path)
print(f"Mejor modelo serializado: {model_path}")

# ─── Resumen final ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(df_res[df_res["Modelo"].str.endswith("_tuned")].to_string(index=False))
print(f"\nMejor modelo: {best_name}  (Recall ATTACK = {best_row['Recall_ATTACK']})")
print(f"Outputs en: {OUTPUT_DIR}")
print("\nEtapa 2 completada exitosamente.")
