"""
agent_logger — guarda cada ejecucion del agente en un archivo JSONL.
Nunca registra las features en texto plano: usa SHA-256 truncado a 16 caracteres.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUTS_E3 = _REPO_ROOT / "outputs" / "etapa3"
_LOG_FILE = _OUTPUTS_E3 / "agent_run_log.jsonl"


def _hash_features(features: dict) -> str:
    serialized = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def log_run(
    features: dict,
    tools_called: list,
    predict_result: dict,
    explain_result: dict,
    threshold_result: dict,
    history_result: dict,
    reasoning: dict,
    total_ms: float,
    ground_truth=None,
) -> None:
    _OUTPUTS_E3.mkdir(parents=True, exist_ok=True)

    dominant_feature = None
    if explain_result and not explain_result.get("explanation_timeout"):
        tops = explain_result.get("top_features", [])
        if tops:
            dominant_feature = tops[0]["feature"]

    correct = None
    if ground_truth is not None and predict_result:
        correct = bool(predict_result.get("prediction") == int(ground_truth))

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features_hash": _hash_features(features),
        "tools_called": tools_called,
        "prob_attack": round(predict_result.get("prob_attack", -1.0), 4) if predict_result else None,
        "decision": predict_result.get("label") if predict_result else "ERROR",
        "zone": threshold_result.get("zone") if threshold_result else None,
        "dominant_feature": dominant_feature,
        "total_latency_ms": round(total_ms, 2),
        "ground_truth": int(ground_truth) if ground_truth is not None else None,
        "correct": correct,
    }

    with open(_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
