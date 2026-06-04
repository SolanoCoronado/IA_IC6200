"""
Etapa 3 — Analisis de etica, privacidad y robustez del agente IDS.

Genera:
  1. Analisis de falsos positivos por Destination Port
  2. Tabla de sesgos conocidos del modelo
  3. Analisis de privacidad (features_hash y advertencias para produccion)
  4. Sensibilidad a perturbaciones en Init_Win_bytes_backward
  5. Impacto de recalibrar el umbral operativo (Recall y FPR por umbral)

Archivos de salida:
  outputs/etapa3/etica_robustez_reporte.json
  outputs/etapa3/etica_robustez_reporte.md
  outputs/etapa3/robustez_sensibilidad.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score, confusion_matrix

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tools.predict_flow import predict_flow

RANDOM_STATE = 42
_REPO_ROOT = _SRC.parent
_OUTPUTS_E2 = _REPO_ROOT / "outputs" / "etapa2"
_OUTPUTS_E3 = _REPO_ROOT / "outputs" / "etapa3"
_OUTPUTS_E3.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_artifacts():
    data = joblib.load(_OUTPUTS_E2 / "split_data.joblib")
    model = joblib.load(_OUTPUTS_E2 / "mejor_modelo_DT_base.joblib")
    with open(_OUTPUTS_E2 / "selected_threshold.json") as f:
        thr = float(json.load(f)["threshold"])
    with open(_OUTPUTS_E2 / "feature_names.json") as f:
        fn = json.load(f)

    X_test = data["X_test"]
    y_test = data["y_test"]
    X_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test, dtype=float)
    y_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
    return model, thr, fn, X_arr, y_arr


def _predict_test(model, X_arr, thr):
    probs = model.predict_proba(X_arr)[:, 1]
    preds = (probs >= thr).astype(int)
    return probs, preds


# ──────────────────────────────────────────────────────────────────────────────
# 1. Analisis de FP por Destination Port
# ──────────────────────────────────────────────────────────────────────────────

def analyze_fp_by_port(model, thr, fn, X_arr, y_arr) -> dict:
    _, preds = _predict_test(model, X_arr, thr)
    fp_mask = (y_arr == 0) & (preds == 1)
    n_fp = int(fp_mask.sum())

    result = {"total_fp": n_fp, "port_distribution": {}}

    if "Destination Port" in fn:
        port_idx = fn.index("Destination Port")
        fp_ports = X_arr[fp_mask, port_idx].astype(int)
        unique_ports, counts = np.unique(fp_ports, return_counts=True)
        port_dist = {int(p): int(c) for p, c in sorted(
            zip(unique_ports, counts), key=lambda x: -x[1]
        )}
        result["port_distribution"] = port_dist
        print(f"\n  Total FP: {n_fp}")
        print(f"  FP por puerto: {port_dist}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2. Tabla de sesgos
# ──────────────────────────────────────────────────────────────────────────────

BIAS_TABLE = [
    {
        "factor": "SQL Injection — soporte de entrenamiento reducido",
        "detail": "Solo 21 instancias de SQL Injection en el conjunto de entrenamiento",
        "risk_level": "ALTO",
        "mitigation": "scope_warning emitido en cada prediccion ATTACK; umbral no reduce este riesgo",
    },
    {
        "factor": "Dataset controlado / trafico sintetico",
        "detail": "CIC-IDS2017 fue generado en entorno de laboratorio, no en produccion real",
        "risk_level": "MEDIO",
        "mitigation": "Evaluar con trafico real antes de despliegue; considerar re-entrenamiento periodico",
    },
    {
        "factor": "Concentracion de ataques en puerto 80",
        "detail": "Mayoria de FP son trafico BENIGN hacia puerto 80 (23/23 FP observados)",
        "risk_level": "MEDIO",
        "mitigation": "Analisis FP por puerto incluido; considerar features de payload para discriminar",
    },
    {
        "factor": "Dominio unico del dataset: Thursday Morning Web Attacks",
        "detail": "El modelo solo vio un dia especifico de trafico web; no cubre ataques de red, DoS, etc.",
        "risk_level": "ALTO",
        "mitigation": "Limitar el alcance operativo a deteccion de ataques web HTTP/HTTPS entrantes",
    },
    {
        "factor": "Desbalance de clases (BENIGN=98.69%, ATTACK=1.31%)",
        "detail": "class_weight='balanced' mitiga en entrenamiento pero el umbral 0.05 introduce FP",
        "risk_level": "BAJO",
        "mitigation": "Umbral 0.05 seleccionado con precision>=0.20; monitorear tasa de FP en produccion",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Analisis de privacidad
# ──────────────────────────────────────────────────────────────────────────────

PRIVACY_ANALYSIS = {
    "log_privacy": {
        "method": "SHA-256 truncado a 16 caracteres (features_hash)",
        "guarantee": "Las features originales NUNCA se persisten en el log",
        "limitation": "El hash no es reversible; no permite auditar features post-hoc",
        "recommendation": "Para auditoria forense, registrar solo el ID del flujo de red, no las features",
    },
    "features_present_in_dataset": {
        "ip_addresses": False,
        "mac_addresses": False,
        "payload_content": False,
        "note": "CIC-IDS2017 no incluye IPs ni MACs en las features estadisticas de flujo",
    },
    "production_warnings": [
        "En un despliegue real, features estadisticas de flujo pueden derivarse de IPs — "
        "revisar legislacion de privacidad aplicable (GDPR, CCPA) antes de usar en produccion.",
        "El agente no almacena contenido de paquetes. Solo estadisticas agrupadas por flujo.",
        "Asegurar que el sistema de captura upstream anonimice IPs si se requiere privacidad.",
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# 4. Analisis de sensibilidad — perturbacion de Init_Win_bytes_backward
# ──────────────────────────────────────────────────────────────────────────────

def analyze_sensitivity(model, thr, fn, X_arr, y_arr) -> dict:
    """Varia Init_Win_bytes_backward en un flujo BENIGN tipico y mide el impacto en prob_attack."""
    feature = "Init_Win_bytes_backward"
    if feature not in fn:
        return {"error": f"Feature '{feature}' not found"}

    feat_idx = fn.index(feature)
    perturbations_pct = [-90, -50, -20, 0, +20, +50, +100, +200]

    # Selecciona un flujo BENIGN con Init_Win_bytes_backward alto (percentil 75)
    benign_idx = np.where(y_arr == 0)[0]
    rng = np.random.RandomState(RANDOM_STATE)

    # Busca un flujo BENIGN con valor alto de la feature objetivo
    benign_vals = X_arr[benign_idx, feat_idx]
    high_val_mask = benign_vals >= np.percentile(benign_vals, 75)
    if high_val_mask.any():
        cands = benign_idx[high_val_mask]
        base_idx = rng.choice(cands)
    else:
        base_idx = rng.choice(benign_idx)

    base_features = {fn[i]: float(X_arr[base_idx, i]) for i in range(len(fn))}
    base_val = base_features[feature]

    print(f"\n  Base flow: BENIGN, {feature}={base_val:.0f}")

    records = []
    for pct in perturbations_pct:
        perturbed = base_features.copy()
        perturbed[feature] = max(0.0, base_val * (1 + pct / 100.0))
        pr = predict_flow(perturbed)
        records.append({
            "perturbation_pct": pct,
            "feature_value": round(perturbed[feature], 2),
            "prob_attack": round(pr["prob_attack"], 4),
            "label": pr["label"],
        })
        print(f"    {pct:+4d}%  {feature}={perturbed[feature]:8.0f}  "
              f"prob={pr['prob_attack']:.4f}  -> {pr['label']}")

    # Plot
    pcts = [r["perturbation_pct"] for r in records]
    probs = [r["prob_attack"] for r in records]
    labels = [r["label"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Izquierda: prob_attack vs perturbacion
    colors = ["#e74c3c" if lbl == "ATTACK" else "#2980b9" for lbl in labels]
    axes[0].bar(range(len(pcts)), probs, color=colors)
    axes[0].axhline(thr, color="black", linestyle="--", linewidth=1.2, label=f"Umbral={thr}")
    axes[0].set_xticks(range(len(pcts)))
    axes[0].set_xticklabels([f"{p:+d}%" for p in pcts], fontsize=9)
    axes[0].set_xlabel(f"Perturbacion de {feature} (%)")
    axes[0].set_ylabel("prob_attack")
    axes[0].set_title(f"Sensibilidad a {feature}\n(rojo=ATTACK, azul=BENIGN)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    # Derecha: valor de la feature vs prob_attack
    feat_vals = [r["feature_value"] for r in records]
    axes[1].plot(feat_vals, probs, "o-", color="#8e44ad", linewidth=2, markersize=6)
    axes[1].axhline(thr, color="black", linestyle="--", linewidth=1.2, label=f"Umbral={thr}")
    axes[1].set_xlabel(f"{feature} (valor absoluto)")
    axes[1].set_ylabel("prob_attack")
    axes[1].set_title(f"prob_attack vs valor de {feature}")
    axes[1].legend()

    fig.suptitle(
        "Analisis de Robustez — Sensibilidad a perturbaciones\n"
        "Flujo BENIGN base con Init_Win_bytes_backward alto",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = _OUTPUTS_E3 / "robustez_sensibilidad.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafico guardado: {out_path.name}")

    return {
        "base_feature": feature,
        "base_value": base_val,
        "perturbations": records,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Tabla de impacto de recalibracion del umbral
# ──────────────────────────────────────────────────────────────────────────────

def analyze_recalibration(model, fn, X_arr, y_arr) -> list[dict]:
    probs = model.predict_proba(X_arr)[:, 1]
    total_benign = int((y_arr == 0).sum())
    rows = []
    thresholds = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]
    print("\n  Umbral   Recall  FPR     FP    FN")
    for t in thresholds:
        preds = (probs >= t).astype(int)
        recall = float(recall_score(y_arr, preds, zero_division=0))
        cm = confusion_matrix(y_arr, preds)
        tn, fp, fn_count, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / total_benign if total_benign > 0 else 0.0
        row = {
            "threshold": t,
            "recall_attack": round(recall, 4),
            "fpr": round(fpr, 5),
            "fp": int(fp),
            "fn": int(fn_count),
            "tp": int(tp),
            "tn": int(tn),
            "is_operational": t == 0.05,
        }
        rows.append(row)
        marker = " <-- operacional" if t == 0.05 else ""
        print(f"  {t:.2f}     {recall:.4f}  {fpr:.5f}  {fp:5d}  {fn_count:5d}{marker}")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Save outputs
# ──────────────────────────────────────────────────────────────────────────────

def _save_json(report: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)


def _save_markdown(report: dict, path: Path) -> None:
    fp_data = report["fp_analysis"]
    sens = report["sensitivity"]
    recal = report["recalibration"]

    lines = [
        "# Etica, Privacidad y Robustez — IDS Agent (Etapa 3)",
        "",
        "## 1. Analisis de Falsos Positivos por Puerto",
        "",
        f"Total FP con umbral operacional (0.05): **{fp_data['total_fp']}**",
        "",
        "| Puerto | FP |",
        "|-------|----|",
    ]
    for port, cnt in list(fp_data["port_distribution"].items())[:10]:
        lines.append(f"| {port} | {cnt} |")

    lines += [
        "",
        "## 2. Tabla de Sesgos Conocidos",
        "",
        "| Factor | Nivel de riesgo | Mitigacion |",
        "|--------|----------------|------------|",
    ]
    for b in report["bias_table"]:
        lines.append(f"| {b['factor']} | {b['risk_level']} | {b['mitigation']} |")

    lines += [
        "",
        "## 3. Analisis de Privacidad",
        "",
        f"**Metodo de anonimizacion en logs:** {report['privacy']['log_privacy']['method']}",
        "",
        f"**Garantia:** {report['privacy']['log_privacy']['guarantee']}",
        "",
        "**Advertencias para produccion:**",
    ]
    for w in report["privacy"]["production_warnings"]:
        lines.append(f"- {w}")

    lines += [
        "",
        "## 4. Sensibilidad a Perturbaciones",
        "",
        f"Feature analizada: `{sens.get('base_feature', 'Init_Win_bytes_backward')}`  ",
        f"Valor base (flujo BENIGN): `{sens.get('base_value', 'N/A')}`",
        "",
        "| Perturbacion | Valor feature | prob_attack | Prediccion |",
        "|-------------|--------------|------------|------------|",
    ]
    for r in sens.get("perturbations", []):
        lines.append(
            f"| {r['perturbation_pct']:+d}% | {r['feature_value']:.0f} | "
            f"{r['prob_attack']:.4f} | {r['label']} |"
        )

    lines += [
        "",
        "![Sensibilidad](robustez_sensibilidad.png)",
        "",
        "## 5. Impacto de Re-calibracion del Umbral",
        "",
        "| Umbral | Recall ATTACK | FPR | FP | FN |",
        "|--------|--------------|-----|----|----|",
    ]
    for r in recal:
        marker = " **<- operacional**" if r["is_operational"] else ""
        lines.append(
            f"| {r['threshold']:.2f} | {r['recall_attack']:.4f} | "
            f"{r['fpr']:.5f} | {r['fp']} | {r['fn']} |{marker}"
        )

    lines += [
        "",
        "---",
        "_Generado automaticamente por etapa3_03_etica_robustez.py_",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  Etica, Privacidad y Robustez — IDS Agent (Etapa 3)")
    print("=" * 65)

    model, thr, fn, X_arr, y_arr = _load_artifacts()

    print("\n[1/4] Analisis de FP por Destination Port...")
    fp_analysis = analyze_fp_by_port(model, thr, fn, X_arr, y_arr)

    print("\n[2/4] Tabla de sesgos (estatica)...")
    for b in BIAS_TABLE:
        print(f"  [{b['risk_level']:5s}] {b['factor'][:60]}")

    print("\n[3/4] Analisis de privacidad (estatico)...")
    print(f"  Metodo: {PRIVACY_ANALYSIS['log_privacy']['method']}")
    print(f"  IPs en dataset: {PRIVACY_ANALYSIS['features_present_in_dataset']['ip_addresses']}")

    print("\n[4/4] Sensibilidad a perturbaciones...")
    sensitivity = analyze_sensitivity(model, thr, fn, X_arr, y_arr)

    print("\n[5/5] Impacto de re-calibracion...")
    recalibration = analyze_recalibration(model, fn, X_arr, y_arr)

    # Construye el diccionario del reporte final
    report = {
        "metadata": {
            "model": "DecisionTreeClassifier",
            "dataset": "CIC-IDS2017 Thursday Morning Web Attacks",
            "operational_threshold": thr,
            "n_test_samples": len(y_arr),
            "n_attack_test": int(y_arr.sum()),
            "n_benign_test": int((y_arr == 0).sum()),
        },
        "fp_analysis": fp_analysis,
        "bias_table": BIAS_TABLE,
        "privacy": PRIVACY_ANALYSIS,
        "sensitivity": sensitivity,
        "recalibration": recalibration,
    }

    json_path = _OUTPUTS_E3 / "etica_robustez_reporte.json"
    md_path = _OUTPUTS_E3 / "etica_robustez_reporte.md"

    _save_json(report, json_path)
    _save_markdown(report, md_path)

    print(f"\n  Guardado: {json_path.name}")
    print(f"  Guardado: {md_path.name}")
    print(f"  Guardado: robustez_sensibilidad.png")
    print(f"  Directorio: {_OUTPUTS_E3}")
    print("\nEtica y robustez completado.")


if __name__ == "__main__":
    main()
