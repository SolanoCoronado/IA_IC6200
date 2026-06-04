"""
Etapa 3 — Agente IDS MVP (arquitectura multi-agente)
OrchestratorAgent coordina DetectorAgent, ExplainerAgent,
ContextAgent y DecisionAgent. No usa ningun LLM externo.

Modos de uso:
    python etapa3_01_agente_mvp.py                  # demo con flujos reales del test set
    python etapa3_01_agente_mvp.py --input flow.json
    python etapa3_01_agente_mvp.py --batch [--n 200]
    python etapa3_01_agente_mvp.py --log-summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# Agrega src/ al path para que los paquetes agents/ y tools/ se encuentren correctamente
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agents.orchestrator_agent import OrchestratorAgent

RANDOM_STATE = 42
_REPO_ROOT = _SRC.parent
_OUTPUTS_E2 = _REPO_ROOT / "outputs" / "etapa2"
_OUTPUTS_E3 = _REPO_ROOT / "outputs" / "etapa3"

# Una sola instancia del orquestador compartida por todos los modos CLI en la sesion
_orchestrator = OrchestratorAgent()


# ──────────────────────────────────────────────────────────────────────────────
# Funcion publica — mantiene compatibilidad con etapa3_02 y etapa3_03
# ──────────────────────────────────────────────────────────────────────────────

def run_agent(features: dict, ground_truth=None) -> dict:
    """Delega la ejecucion al OrchestratorAgent. Mantiene la misma interfaz publica."""
    return _orchestrator.run(features, ground_truth=ground_truth)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load_test_set():
    data = joblib.load(_OUTPUTS_E2 / "split_data.joblib")
    X = data["X_test"]
    y = data["y_test"]
    fn = list(data["feature_names"])
    X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    return X_arr, y_arr, fn


def _row_to_dict(X_arr: np.ndarray, idx: int, feature_names: list) -> dict:
    return {fn: float(X_arr[idx, i]) for i, fn in enumerate(feature_names)}


def _print_result(result: dict, ground_truth=None) -> None:
    if "error" in result:
        print(f"  [ERROR] {result['error']}")
        return
    pr = result["predict"]
    thr = result["threshold"]
    print(f"  Prediction : {pr['label']}  (prob={pr['prob_attack']:.4f}, "
          f"zone={thr['zone']})")
    if ground_truth is not None:
        correct = "OK" if int(ground_truth) == pr["prediction"] else "FAIL"
        print(f"  Ground truth: {'ATTACK' if ground_truth else 'BENIGN'}  [{correct}]")
    print(f"  Agents     : {' -> '.join(result['tools_called'])}")
    print(f"  Latency    : {result['total_latency_ms']:.1f} ms")
    print(f"  Reasoning  :")
    for line in result["reasoning"]["lines"]:
        print(f"    {line}")


def _stratified_sample(y_arr: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    labels = np.unique(y_arr)
    indices: list[int] = []
    for lbl in labels:
        lbl_idx = np.where(y_arr == lbl)[0]
        n_lbl = max(1, int(round(np.mean(y_arr == lbl) * n)))
        chosen = rng.choice(lbl_idx, size=min(n_lbl, len(lbl_idx)), replace=False)
        indices.extend(chosen.tolist())
    indices = sorted(set(indices))
    return np.array(indices[:n])


# ──────────────────────────────────────────────────────────────────────────────
# Modos CLI
# ──────────────────────────────────────────────────────────────────────────────

def _mode_demo() -> None:
    print("\n" + "=" * 65)
    print("  IDS Agent MVP — Demo (flujos reales del test set)")
    print("  Arquitectura: OrchestratorAgent + 4 sub-agentes")
    print("=" * 65)
    X_arr, y_arr, fn = _load_test_set()
    rng = np.random.RandomState(RANDOM_STATE)

    attack_idx_arr = np.where(y_arr == 1)[0]
    benign_idx_arr = np.where(y_arr == 0)[0]

    demo_cases = [
        ("FLUJO ATTACK TIPICO", rng.choice(attack_idx_arr), 1),
        ("FLUJO BENIGN TIPICO", rng.choice(benign_idx_arr), 0),
    ]

    for title, idx, gt in demo_cases:
        print(f"\n{'-'*65}")
        print(f"  {title}")
        print(f"{'-'*65}")
        features = _row_to_dict(X_arr, idx, fn)
        key_vals = {k: features[k] for k in [
            "Destination Port", "Init_Win_bytes_backward",
            "Flow Bytes/s", "min_seg_size_forward"
        ] if k in features}
        print(f"  Key features: {key_vals}")
        result = run_agent(features, ground_truth=gt)
        _print_result(result, ground_truth=gt)

    stats = _orchestrator.session_stats()
    print(f"\n{'='*65}")
    print(f"  Session stats: {stats}")
    print(f"  Log guardado en: outputs/etapa3/agent_run_log.jsonl")


def _mode_input(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Archivo no encontrado: {path}")
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        features = json.load(f)
    print(f"\n[INPUT] {path}")
    result = run_agent(features)
    _print_result(result)


def _mode_batch(n: int) -> None:
    print(f"\n{'='*65}")
    print(f"  IDS Agent MVP — Batch ({n} flujos estratificados)")
    print("=" * 65)
    X_arr, y_arr, fn = _load_test_set()
    rng = np.random.RandomState(RANDOM_STATE)

    sample_idx = _stratified_sample(y_arr, n, rng)
    latencies, decisions, corrects = [], [], []

    for i, idx in enumerate(sample_idx, 1):
        gt = int(y_arr[idx])
        features = _row_to_dict(X_arr, idx, fn)
        result = run_agent(features, ground_truth=gt)
        if "error" not in result:
            latencies.append(result["total_latency_ms"])
            decisions.append(result["predict"]["label"])
            corrects.append(result["predict"]["prediction"] == gt)
        if i % 50 == 0:
            print(f"  [{i}/{n}] ...")

    attacks = decisions.count("ATTACK")
    benigns = decisions.count("BENIGN")
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    acc = float(np.mean(corrects)) if corrects else 0.0

    print(f"\n  Resultados sobre {len(sample_idx)} flujos:")
    print(f"    ATTACK detectados : {attacks} ({attacks/len(decisions)*100:.1f}%)")
    print(f"    BENIGN permitidos : {benigns} ({benigns/len(decisions)*100:.1f}%)")
    print(f"    Accuracy (labeled): {acc:.4f}")
    print(f"    Latencia P50      : {p50:.1f} ms")
    print(f"    Latencia P95      : {p95:.1f} ms")
    stats = _orchestrator.session_stats()
    print(f"  Session stats     : {stats}")
    print(f"  Log guardado en: outputs/etapa3/agent_run_log.jsonl")


def _mode_log_summary() -> None:
    log_path = _OUTPUTS_E3 / "agent_run_log.jsonl"
    if not log_path.exists():
        print("[INFO] No se encontro log. Ejecuta el agente primero.")
        return

    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("[INFO] Log vacio.")
        return

    total = len(entries)
    attacks = sum(1 for e in entries if e.get("decision") == "ATTACK")
    benigns = sum(1 for e in entries if e.get("decision") == "BENIGN")
    errors = total - attacks - benigns

    labeled = [e for e in entries if e.get("correct") is not None]
    accuracy = float(np.mean([e["correct"] for e in labeled])) if labeled else None

    latencies = [e["total_latency_ms"] for e in entries if e.get("total_latency_ms") is not None]
    p50 = float(np.percentile(latencies, 50)) if latencies else None
    p95 = float(np.percentile(latencies, 95)) if latencies else None

    zones = {}
    for e in entries:
        z = e.get("zone") or "unknown"
        zones[z] = zones.get(z, 0) + 1

    tools_freq: dict = {}
    for e in entries:
        for t in e.get("tools_called", []):
            tools_freq[t] = tools_freq.get(t, 0) + 1

    dominant_counts: dict = {}
    for e in entries:
        df = e.get("dominant_feature")
        if df:
            dominant_counts[df] = dominant_counts.get(df, 0) + 1

    print(f"\n{'='*55}")
    print(f"  Agent Log Summary — {log_path.name}")
    print(f"{'='*55}")
    print(f"  Total ejecuciones : {total}")
    print(f"    ATTACK           : {attacks} ({attacks/total*100:.1f}%)")
    print(f"    BENIGN           : {benigns} ({benigns/total*100:.1f}%)")
    if errors:
        print(f"    ERROR            : {errors}")
    if accuracy is not None:
        print(f"  Accuracy (labeled): {accuracy:.4f}  ({len(labeled)} muestras)")
    if p50 is not None:
        print(f"  Latencia P50      : {p50:.1f} ms")
        print(f"  Latencia P95      : {p95:.1f} ms")
    print(f"  Zonas             : {zones}")
    print(f"  Tools (frec.)     : {tools_freq}")
    if dominant_counts:
        top_feat = max(dominant_counts, key=dominant_counts.get)
        print(f"  Feature dominante : {top_feat} ({dominant_counts[top_feat]} veces)")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IDS Agent MVP — Etapa 3 multi-agente (sin LLM externo)"
    )
    parser.add_argument("--input", metavar="FILE", help="JSON con features del flujo")
    parser.add_argument("--batch", action="store_true", help="Modo batch sobre test set")
    parser.add_argument("--n", type=int, default=200, help="Flujos en modo batch (default=200)")
    parser.add_argument("--log-summary", action="store_true", help="Resumen del log JSONL")
    args = parser.parse_args()

    if args.log_summary:
        _mode_log_summary()
    elif args.input:
        _mode_input(args.input)
    elif args.batch:
        _mode_batch(args.n)
    else:
        _mode_demo()


if __name__ == "__main__":
    main()
