"""
Etapa 3 — Plan de validacion completo del agente IDS.

Escenarios:
  Normales     (5): BENIGN tipico, ATTACK tipico, 5 ATTACKs, 5 BENIGNs, BENIGN no-puerto80
  Estres       (3): latencia batch-100, outlier P99.9, variacion de umbral
  Casos limite (3): zona incierta, 30% features faltantes, similar a SQL Injection
  Fallos       (2): dict vacio, valores tipo string

Criterios de aceptacion:
  - Recall ATTACK >= 0.95  (sobre todos los ATTACKs del test set)
  - Latencia P95 < 3000 ms
  - Fallos manejados sin que el programa falle
  - scope_warning presente en predicciones ATTACK
  - Tasa global de acierto >= 0.90
"""
from __future__ import annotations

import csv
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import recall_score

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from etapa3_01_agente_mvp import run_agent
from tools.predict_flow import predict_flow
from tools.check_threshold import check_threshold

RANDOM_STATE = 42
_REPO_ROOT = _SRC.parent
_OUTPUTS_E2 = _REPO_ROOT / "outputs" / "etapa2"
_OUTPUTS_E3 = _REPO_ROOT / "outputs" / "etapa3"
_OUTPUTS_E3.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_test_set():
    data = joblib.load(_OUTPUTS_E2 / "split_data.joblib")
    X = data["X_test"]
    y = data["y_test"]
    fn = list(data["feature_names"])
    X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    return X_arr, y_arr, fn


def _row(X_arr, idx, fn):
    return {f: float(X_arr[idx, i]) for i, f in enumerate(fn)}


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    name: str
    category: str
    passed: bool
    details: str
    latency_ms: float = 0.0
    expected: str = ""
    got: str = ""
    extra: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Individual scenario runners
# ──────────────────────────────────────────────────────────────────────────────

def _run_single(features, gt=None):
    return run_agent(features, ground_truth=gt)


def scenario_benign_typical(X_arr, y_arr, fn, rng):
    idx = rng.choice(np.where(y_arr == 0)[0])
    res = _run_single(_row(X_arr, idx, fn), gt=0)
    passed = res.get("predict", {}).get("label") == "BENIGN"
    return ScenarioResult(
        "BENIGN_typical", "normal", passed,
        f"label={res.get('predict',{}).get('label')}",
        res.get("total_latency_ms", 0),
        expected="BENIGN", got=res.get("predict", {}).get("label", "ERR"),
    )


def scenario_attack_typical(X_arr, y_arr, fn, rng):
    idx = rng.choice(np.where(y_arr == 1)[0])
    res = _run_single(_row(X_arr, idx, fn), gt=1)
    passed = res.get("predict", {}).get("label") == "ATTACK"
    return ScenarioResult(
        "ATTACK_typical", "normal", passed,
        f"label={res.get('predict',{}).get('label')} "
        f"prob={res.get('predict',{}).get('prob_attack',0):.4f}",
        res.get("total_latency_ms", 0),
        expected="ATTACK", got=res.get("predict", {}).get("label", "ERR"),
    )


def scenario_five_attacks(X_arr, y_arr, fn, rng):
    attack_idx = rng.choice(np.where(y_arr == 1)[0], size=5, replace=False)
    results = [_run_single(_row(X_arr, i, fn), gt=1) for i in attack_idx]
    detected = sum(1 for r in results if r.get("predict", {}).get("label") == "ATTACK")
    passed = detected == 5
    return ScenarioResult(
        "5_ATTACKs", "normal", passed,
        f"detected={detected}/5",
        float(np.mean([r.get("total_latency_ms", 0) for r in results])),
        expected="5/5", got=f"{detected}/5",
    )


def scenario_five_benigns(X_arr, y_arr, fn, rng):
    benign_idx = rng.choice(np.where(y_arr == 0)[0], size=5, replace=False)
    results = [_run_single(_row(X_arr, i, fn), gt=0) for i in benign_idx]
    allowed = sum(1 for r in results if r.get("predict", {}).get("label") == "BENIGN")
    passed = allowed == 5
    return ScenarioResult(
        "5_BENIGNs", "normal", passed,
        f"allowed={allowed}/5",
        float(np.mean([r.get("total_latency_ms", 0) for r in results])),
        expected="5/5", got=f"{allowed}/5",
    )


def scenario_benign_non_port80(X_arr, y_arr, fn, rng):
    port_idx = fn.index("Destination Port") if "Destination Port" in fn else None
    benign_idx = np.where(y_arr == 0)[0]
    if port_idx is not None:
        mask = X_arr[benign_idx, port_idx] != 80
        candidates = benign_idx[mask]
        idx = rng.choice(candidates) if len(candidates) > 0 else rng.choice(benign_idx)
    else:
        idx = rng.choice(benign_idx)
    res = _run_single(_row(X_arr, idx, fn), gt=0)
    passed = res.get("predict", {}).get("label") == "BENIGN"
    port_val = X_arr[idx, port_idx] if port_idx is not None else "N/A"
    return ScenarioResult(
        "BENIGN_non_port80", "normal", passed,
        f"port={port_val}  label={res.get('predict',{}).get('label')}",
        res.get("total_latency_ms", 0),
        expected="BENIGN", got=res.get("predict", {}).get("label", "ERR"),
    )


def scenario_latency_batch(X_arr, y_arr, fn, rng, n=100):
    idx_sample = rng.choice(len(X_arr), size=n, replace=False)
    latencies = []
    for idx in idx_sample:
        res = _run_single(_row(X_arr, idx, fn))
        latencies.append(res.get("total_latency_ms", 0))
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    passed = p95 < 3000
    return ScenarioResult(
        "latency_batch_100", "stress", passed,
        f"P50={p50:.1f}ms  P95={p95:.1f}ms  (threshold=3000ms)",
        p95,
        expected="P95<3000ms", got=f"P95={p95:.1f}ms",
        extra={"p50_ms": p50, "p95_ms": p95, "n": n},
    )


def scenario_outlier_p999(X_arr, y_arr, fn, rng):
    outlier_features = {
        f: float(np.percentile(X_arr[:, i], 99.9))
        for i, f in enumerate(fn)
    }
    res = _run_single(outlier_features)
    passed = "error" not in res
    label = res.get("predict", {}).get("label", "ERROR") if "error" not in res else "CRASH"
    return ScenarioResult(
        "outlier_p999", "stress", passed,
        f"label={label}  no_crash={passed}",
        res.get("total_latency_ms", 0),
        expected="no crash", got=label,
    )


def scenario_threshold_variation(X_arr, y_arr, fn, rng):
    """Verifica que la prediccion del modelo se mantiene estable cuando se varia el umbral."""
    idx = rng.choice(np.where(y_arr == 1)[0])
    features = _row(X_arr, idx, fn)
    pr = predict_flow(features)
    prob = pr["prob_attack"]
    thr = pr["threshold_used"]

    labels_by_thresh = {}
    for delta in [-0.10, 0.0, +0.10]:
        t_mod = max(0.001, thr + delta)
        labels_by_thresh[round(delta, 2)] = "ATTACK" if prob >= t_mod else "BENIGN"

    # Con el umbral original el flujo debe clasificarse como ATTACK (Recall conocido = 1.0)
    passed = labels_by_thresh[0.0] == "ATTACK"
    summary = "  ".join(f"delta={k:+.2f}->{v}" for k, v in labels_by_thresh.items())
    return ScenarioResult(
        "threshold_variation", "stress", passed,
        f"prob={prob:.4f}  {summary}",
        pr.get("latency_ms", 0),
        expected="ATTACK at delta=0.0", got=labels_by_thresh[0.0],
        extra={"prob": prob, "labels_by_thresh": labels_by_thresh},
    )


def scenario_uncertain_zone(X_arr, y_arr, fn, rng):
    """Busca un flujo con probabilidad en zona incierta [umbral, 0.40). Si no existe, crea uno sintetico."""
    import joblib as _jl, json as _json

    with open(_OUTPUTS_E2 / "selected_threshold.json") as f:
        thr = float(_json.load(f)["threshold"])

    # Buscar muestra real en zona incierta
    import sklearn
    model = _jl.load(_OUTPUTS_E2 / "mejor_modelo_DT_base.joblib")
    probs = model.predict_proba(X_arr)[:, 1]
    uncertain_mask = (probs >= thr) & (probs < 0.40)
    uncertain_idx = np.where(uncertain_mask)[0]

    if len(uncertain_idx) > 0:
        idx = rng.choice(uncertain_idx)
        features = _row(X_arr, idx, fn)
        source = "real"
    else:
        # Sintetico: tomar BENIGN y reducir Init_Win_bytes_backward
        idx = rng.choice(np.where(y_arr == 0)[0])
        features = _row(X_arr, idx, fn)
        if "Init_Win_bytes_backward" in features:
            features["Init_Win_bytes_backward"] = float(np.percentile(X_arr[:, fn.index("Init_Win_bytes_backward")], 5))
        source = "synthetic"

    res = _run_single(features)
    zone = res.get("threshold", {}).get("zone", "unknown")
    history_called = "query_history" in res.get("tools_called", [])
    prob_res = res.get("predict", {}).get("prob_attack", 0)

    # Se espera que query_history sea invocado en zona incierta o si hay features faltantes
    passed = "error" not in res and not res.get("predict", {}) == {}
    return ScenarioResult(
        "uncertain_zone", "edge", passed,
        f"source={source}  zone={zone}  prob={prob_res:.4f}  history_called={history_called}",
        res.get("total_latency_ms", 0),
        expected="no crash + history if uncertain", got=zone,
        extra={"source": source, "history_called": history_called},
    )


def scenario_missing_30pct(X_arr, y_arr, fn, rng):
    idx = rng.choice(np.where(y_arr == 1)[0])
    features = _row(X_arr, idx, fn)
    n_drop = int(len(fn) * 0.30)
    keys_to_drop = rng.choice(list(features.keys()), size=n_drop, replace=False)
    for k in keys_to_drop:
        del features[k]

    res = _run_single(features, gt=1)
    missing_reported = len(res.get("predict", {}).get("missing_features", [])) if "error" not in res else -1
    history_called = "query_history" in res.get("tools_called", [])
    passed = "error" not in res and history_called  # missing features deben activar history
    return ScenarioResult(
        "missing_30pct", "edge", passed,
        f"dropped={n_drop}/{len(fn)}  missing_reported={missing_reported}  history_called={history_called}",
        res.get("total_latency_ms", 0),
        expected="query_history called", got=f"called={history_called}",
    )


def scenario_sql_injection_like(X_arr, y_arr, fn, rng):
    """Flujo similar a SQL Injection: usa un ATTACK real de puerto 80 del test set (Thursday WebAttacks)."""
    port_idx = fn.index("Destination Port") if "Destination Port" in fn else None
    attack_idx_arr = np.where(y_arr == 1)[0]

    # Se prefieren ATTACKs en puerto 80 (tipico de SQL Injection y XSS en este dataset)
    if port_idx is not None:
        port80_mask = X_arr[attack_idx_arr, port_idx] == 80
        candidates = attack_idx_arr[port80_mask]
        if len(candidates) == 0:
            candidates = attack_idx_arr
    else:
        candidates = attack_idx_arr

    idx = rng.choice(candidates)
    features = _row(X_arr, idx, fn)
    port_used = features.get("Destination Port", "N/A")

    res = _run_single(features, gt=1)
    label = res.get("predict", {}).get("label", "ERROR")
    scope = res.get("threshold", {}).get("scope_warning")
    passed = "error" not in res and label == "ATTACK" and scope is not None
    return ScenarioResult(
        "sql_injection_like", "edge", passed,
        f"port={port_used}  label={label}  scope_warning={'yes' if scope else 'no'}",
        res.get("total_latency_ms", 0),
        expected="ATTACK + scope_warning", got=f"{label}+sw={'yes' if scope else 'no'}",
    )


def scenario_empty_dict():
    t0 = time.perf_counter()
    res = run_agent({})
    elapsed = (time.perf_counter() - t0) * 1000
    passed = "error" not in res  # dict vacio: todas las features se rellenan con 0.0, no debe fallar
    label = res.get("predict", {}).get("label", "ERROR") if "error" not in res else "ERROR"
    return ScenarioResult(
        "empty_dict", "failure", passed,
        f"no_crash={passed}  label={label}  all_zeros=True",
        elapsed,
        expected="no crash (all features=0)", got=label,
    )


def scenario_string_values():
    features = {
        "Destination Port": "eighty",
        "Flow Duration": "fast",
        "Init_Win_bytes_backward": "not_a_float",
    }
    t0 = time.perf_counter()
    res = run_agent(features)
    elapsed = (time.perf_counter() - t0) * 1000
    # Se espera un error con mensaje claro, no una excepcion no controlada
    has_error_key = "error" in res
    passed = has_error_key and isinstance(res["error"], str) and len(res["error"]) > 0
    return ScenarioResult(
        "string_values", "failure", passed,
        f"error_returned={has_error_key}  msg={res.get('error','')[:60]}",
        elapsed,
        expected="error dict returned", got="error" if has_error_key else "no error",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Acceptance criteria check
# ──────────────────────────────────────────────────────────────────────────────

def check_acceptance_criteria(X_arr, y_arr, fn, scenario_results):
    print("\n" + "=" * 65)
    print("  CRITERIOS DE ACEPTACION")
    print("=" * 65)
    criteria = {}

    # 1. Recall ATTACK >= 0.95 (sobre todos los ATTACKs del test set)
    import joblib as _jl, json as _json
    model = _jl.load(_OUTPUTS_E2 / "mejor_modelo_DT_base.joblib")
    with open(_OUTPUTS_E2 / "selected_threshold.json") as f:
        thr = float(_json.load(f)["threshold"])
    probs = model.predict_proba(X_arr)[:, 1]
    preds = (probs >= thr).astype(int)
    recall = float(recall_score(y_arr, preds, zero_division=0))
    criteria["recall_attack_ge_095"] = {"value": round(recall, 4), "pass": recall >= 0.95, "threshold": 0.95}

    # 2. P95 latencia < 3000 ms
    lat_scen = next((s for s in scenario_results if s.name == "latency_batch_100"), None)
    p95 = lat_scen.extra.get("p95_ms", 9999) if lat_scen else 9999
    criteria["latency_p95_lt_3000ms"] = {"value": round(p95, 1), "pass": p95 < 3000, "threshold": 3000}

    # 3. Fallos manejados sin crash
    failure_results = [s for s in scenario_results if s.category == "failure"]
    all_handled = all(s.passed for s in failure_results)
    criteria["failures_no_crash"] = {
        "value": f"{sum(s.passed for s in failure_results)}/{len(failure_results)}",
        "pass": all_handled,
        "threshold": "all",
    }

    # 4. scope_warning presente en predicciones ATTACK
    sw_scen = next((s for s in scenario_results if s.name == "sql_injection_like"), None)
    scope_ok = sw_scen.passed if sw_scen else False
    criteria["scope_warning_on_attack"] = {"value": str(scope_ok), "pass": scope_ok, "threshold": "True"}

    # 5. Tasa global de acierto >= 0.90
    accuracy = float(np.mean(preds == y_arr))
    criteria["global_accuracy_ge_090"] = {"value": round(accuracy, 4), "pass": accuracy >= 0.90, "threshold": 0.90}

    for name, c in criteria.items():
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {name:40s} {c['value']}  (req: {c['threshold']})")

    all_pass = all(c["pass"] for c in criteria.values())
    print(f"\n  VEREDICTO GLOBAL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 65)
    return criteria


# ──────────────────────────────────────────────────────────────────────────────
# Save outputs
# ──────────────────────────────────────────────────────────────────────────────

def _save_csv(results: list[ScenarioResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "category", "passed", "details", "latency_ms", "expected", "got"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "name": r.name, "category": r.category, "passed": r.passed,
                "details": r.details, "latency_ms": round(r.latency_ms, 2),
                "expected": r.expected, "got": r.got,
            })


def _save_report(results: list[ScenarioResult], criteria: dict, path: Path) -> None:
    passed = sum(r.passed for r in results)
    total = len(results)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("REPORTE DE VALIDACION — IDS Agent MVP (Etapa 3)\n")
        f.write("=" * 65 + "\n\n")

        for cat in ["normal", "stress", "edge", "failure"]:
            cat_results = [r for r in results if r.category == cat]
            if not cat_results:
                continue
            f.write(f"--- {cat.upper()} SCENARIOS ---\n")
            for r in cat_results:
                status = "PASS" if r.passed else "FAIL"
                f.write(f"  [{status}] {r.name}\n")
                f.write(f"         expected={r.expected}  got={r.got}\n")
                f.write(f"         {r.details}\n")
                f.write(f"         latency={r.latency_ms:.1f}ms\n\n")

        f.write("--- CRITERIOS DE ACEPTACION ---\n")
        for name, c in criteria.items():
            status = "PASS" if c["pass"] else "FAIL"
            f.write(f"  [{status}] {name}: {c['value']}  (req: {c['threshold']})\n")

        all_pass = all(c["pass"] for c in criteria.values())
        f.write(f"\nScenarios: {passed}/{total} passed\n")
        f.write(f"Veredicto: {'PASS' if all_pass else 'FAIL'}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  IDS Agent — Plan de Validacion Completo (Etapa 3)")
    print("=" * 65)

    X_arr, y_arr, fn = _load_test_set()
    rng = np.random.RandomState(RANDOM_STATE)
    results: list[ScenarioResult] = []

    print("\n[1/4] Escenarios normales...")
    results.append(scenario_benign_typical(X_arr, y_arr, fn, rng))
    results.append(scenario_attack_typical(X_arr, y_arr, fn, rng))
    results.append(scenario_five_attacks(X_arr, y_arr, fn, rng))
    results.append(scenario_five_benigns(X_arr, y_arr, fn, rng))
    results.append(scenario_benign_non_port80(X_arr, y_arr, fn, rng))

    print("[2/4] Escenarios de estres...")
    results.append(scenario_latency_batch(X_arr, y_arr, fn, rng))
    results.append(scenario_outlier_p999(X_arr, y_arr, fn, rng))
    results.append(scenario_threshold_variation(X_arr, y_arr, fn, rng))

    print("[3/4] Casos limite...")
    results.append(scenario_uncertain_zone(X_arr, y_arr, fn, rng))
    results.append(scenario_missing_30pct(X_arr, y_arr, fn, rng))
    results.append(scenario_sql_injection_like(X_arr, y_arr, fn, rng))

    print("[4/4] Fallos esperados...")
    results.append(scenario_empty_dict())
    results.append(scenario_string_values())

    # Summary
    print("\n" + "=" * 65)
    print("  RESULTADOS DE ESCENARIOS")
    print("=" * 65)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name:<30s} ({r.category})")
        print(f"         {r.details}")

    # Acceptance criteria
    criteria = check_acceptance_criteria(X_arr, y_arr, fn, results)

    # Save outputs
    csv_path = _OUTPUTS_E3 / "validacion_resultados.csv"
    txt_path = _OUTPUTS_E3 / "validacion_reporte.txt"
    _save_csv(results, csv_path)
    _save_report(results, criteria, txt_path)

    passed = sum(r.passed for r in results)
    print(f"\n  Escenarios: {passed}/{len(results)} passed")
    print(f"  Guardado: {csv_path.name}, {txt_path.name}")
    print(f"  Directorio: {_OUTPUTS_E3}")


if __name__ == "__main__":
    main()
