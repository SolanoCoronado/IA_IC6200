import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

BASE_DIR = r"C:\Users\pavel\OneDrive - Estudiantes ITCR\Documentos\GitHub\IA_IC6200"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "etapa2")
DATA_PATH = os.path.join(OUT_DIR, "datos_limpios_base.csv")
SPLIT_PATH = os.path.join(OUT_DIR, "split_data.joblib")
RANDOM_STATE = 42

print("=" * 60)
print("AUDITORÍA DE POSIBLE LEAKAGE / SEPARACIÓN PERFECTA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

target_col = "Label_binary" if "Label_binary" in df.columns else "Label"
y = df[target_col].map({"BENIGN": 0, "ATTACK": 1}) if df[target_col].dtype == "object" else df[target_col]
X = df.drop(columns=[c for c in ["Label", "Label_binary"] if c in df.columns])

print(f"\nDataset: {X.shape}")
print(f"ATTACK: {y.sum():,} ({y.mean()*100:.2f}%)")

# 1. Revisar columnas sospechosas
print("\n[1] Columnas sospechosas por nombre")
suspicious_terms = ["label", "attack", "class", "target", "outcome", "benign"]
suspicious_cols = [
    c for c in X.columns
    if any(term in c.lower() for term in suspicious_terms)
]
print(suspicious_cols if suspicious_cols else "No se encontraron columnas sospechosas por nombre.")

# 2. Revisar columnas constantes o casi constantes
print("\n[2] Columnas con pocos valores únicos")
nunique = X.nunique().sort_values()
print(nunique.head(20))

# 3. Revisar separación por puerto
print("\n[3] Separación por Destination Port")
if "Destination Port" in X.columns:
    port_table = (
        df.assign(y=y)
        .groupby("Destination Port")["y"]
        .agg(["count", "sum", "mean"])
        .sort_values(["mean", "count"], ascending=[False, False])
    )
    port_table.columns = ["total", "attacks", "attack_rate"]
    print(port_table.head(20))
    port_table.to_csv(os.path.join(OUT_DIR, "audit_destination_port.csv"))
else:
    print("No existe Destination Port.")

# 4. Revisar features individuales con AUC casi perfecto
print("\n[4] AUC individual por feature")
rows = []
for col in X.columns:
    vals = X[col]
    if vals.nunique() > 1:
        try:
            auc = roc_auc_score(y, vals)
            auc = max(auc, 1 - auc)
            rows.append((col, auc, vals.nunique()))
        except Exception:
            pass

auc_df = pd.DataFrame(rows, columns=["feature", "single_feature_auc", "nunique"])
auc_df = auc_df.sort_values("single_feature_auc", ascending=False)
print(auc_df.head(20))
auc_df.to_csv(os.path.join(OUT_DIR, "audit_single_feature_auc.csv"), index=False)

# 5. Comparar desempeño sin features más sospechosas
print("\n[5] Ablation rápida de features dominantes")
drop_candidates = [
    "Init_Win_bytes_backward",
    "Init_Win_bytes_forward",
    "Destination Port",
    "Flow Bytes/s",
    "Average Packet Size",
]

existing_drop = [c for c in drop_candidates if c in X.columns]
print("Features eliminadas para prueba:", existing_drop)

X_reduced = X.drop(columns=existing_drop)

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

models = {
    "DT_reduced": DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=20,
        min_samples_split=10,
        criterion="entropy",
        random_state=RANDOM_STATE,
    ),
    "HGB_reduced": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.2,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test)

    preds = (probs >= 0.5).astype(int)

    recall = ((preds == 1) & (y_test == 1)).sum() / (y_test == 1).sum()
    precision = ((preds == 1) & (y_test == 1)).sum() / max((preds == 1).sum(), 1)
    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)

    print(f"\n{name}")
    print(f"  Recall ATTACK:    {recall:.4f}")
    print(f"  Precision ATTACK: {precision:.4f}")
    print(f"  ROC-AUC:          {roc:.4f}")
    print(f"  PR-AUC:           {pr:.4f}")

# 6. Revisar probabilidades saturadas del mejor modelo si existe
print("\n[6] Distribución de probabilidades del modelo guardado")
model_files = [
    f for f in os.listdir(OUT_DIR)
    if f.startswith("mejor_modelo") and f.endswith(".joblib")
]

if model_files:
    model_path = os.path.join(OUT_DIR, model_files[0])
    model = joblib.load(model_path)

    split = joblib.load(SPLIT_PATH)
    X_test_saved = split["X_test"]
    y_test_saved = split["y_test"]

    probs = model.predict_proba(X_test_saved)[:, 1]

    print(f"Modelo: {model_files[0]}")
    print(f"Prob min: {probs.min():.6f}")
    print(f"Prob max: {probs.max():.6f}")
    print(f"Prob == 0: {(probs == 0).sum():,}")
    print(f"Prob == 1: {(probs == 1).sum():,}")
    print(f"Prob entre 0.05 y 0.95: {((probs > 0.05) & (probs < 0.95)).sum():,}")

    prob_df = pd.DataFrame({
        "y_true": y_test_saved,
        "prob_attack": probs
    })
    prob_df.to_csv(os.path.join(OUT_DIR, "audit_probabilities.csv"), index=False)
else:
    print("No se encontró modelo guardado.")

print("\nAuditoría terminada.")
print(f"Outputs guardados en: {OUT_DIR}")