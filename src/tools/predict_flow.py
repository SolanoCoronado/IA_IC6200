"""
predict_flow — carga el modelo DT entrenado y retorna la prediccion completa.
Las features que falten se rellenan con 0.0. Valida que los valores sean numericos.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUTS_E2 = _REPO_ROOT / "outputs" / "etapa2"

_model: Any = None
_feature_names: list | None = None
_threshold: float | None = None
_attack_idx: int | None = None


def _load() -> tuple:
    global _model, _feature_names, _threshold, _attack_idx
    if _model is None:
        _model = joblib.load(_OUTPUTS_E2 / "mejor_modelo_DT_base.joblib")
        classes = list(_model.classes_)
        _attack_idx = classes.index(1) if 1 in classes else 1
    if _feature_names is None:
        fn_path = _OUTPUTS_E2 / "feature_names.json"
        if fn_path.exists():
            with open(fn_path, encoding="utf-8") as f:
                _feature_names = json.load(f)
        else:
            data = joblib.load(_OUTPUTS_E2 / "split_data.joblib")
            _feature_names = list(data["feature_names"])
    if _threshold is None:
        with open(_OUTPUTS_E2 / "selected_threshold.json", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, (int, float)):
            _threshold = float(raw)
        else:
            _threshold = float(
                raw.get("threshold")
                or raw.get("selected_threshold")
                or raw.get("best_threshold")
                or 0.05
            )
    return _model, _feature_names, _threshold, _attack_idx


def predict_flow(features: dict) -> dict:
    """
    Parametros:
        features: dict {nombre_feature: valor_numerico}. Claves faltantes se rellenan con 0.0.
    Retorna:
        dict con prob_attack, prediction, label, threshold_used,
              missing_features, latency_ms.
    Lanza:
        ValueError: si algun valor no es numerico.
    """
    for key, val in features.items():
        if not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"Feature '{key}' must be numeric, got {type(val).__name__!r}"
            )

    model, feature_names, threshold, attack_idx = _load()
    t0 = time.perf_counter()

    missing = [f for f in feature_names if f not in features]
    X = np.array([[float(features.get(f, 0.0)) for f in feature_names]])

    proba = model.predict_proba(X)[0]
    prob_attack = float(proba[attack_idx])
    prediction = int(prob_attack >= threshold)
    label = "ATTACK" if prediction == 1 else "BENIGN"
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "prob_attack": round(prob_attack, 6),
        "prediction": prediction,
        "label": label,
        "threshold_used": threshold,
        "missing_features": missing,
        "latency_ms": round(latency_ms, 3),
    }
