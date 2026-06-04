"""
explain_flow — calcula explicaciones SHAP con TreeExplainer sobre el modelo DT.
Solo se ejecuta si prob_attack > 0.03. Limite de 5 segundos via ThreadPoolExecutor.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import shap

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUTS_E2 = _REPO_ROOT / "outputs" / "etapa2"

_model: Any = None
_explainer: Any = None
_feature_names: list | None = None
_attack_idx: int | None = None

_EXPLAIN_TIMEOUT = 5.0  # seconds


def _load() -> tuple:
    global _model, _explainer, _feature_names, _attack_idx
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
    if _explainer is None:
        _explainer = shap.TreeExplainer(_model)
    return _explainer, _feature_names, _attack_idx


def _compute_shap(X: np.ndarray) -> np.ndarray:
    explainer, _, attack_idx = _load()
    sv = explainer.shap_values(X)

    if isinstance(sv, list):
        # API antigua de DT binario: lista [clase0, clase1]
        # Cada elemento puede ser (n_muestras, n_features) o (n_features,)
        vals = np.asarray(sv[attack_idx])
        return vals[0] if vals.ndim == 2 else vals

    sv_arr = np.asarray(sv)
    if sv_arr.ndim == 3:
        # shape (n_samples, n_features, n_classes)
        return sv_arr[0, :, attack_idx]
    if sv_arr.ndim == 2:
        return sv_arr[0]
    return sv_arr


def explain_flow(features: dict, prediction_result: dict | None = None) -> dict:
    """
    Retorna las 5 features mas importantes segun SHAP para la clase ATTACK.
    Retorna {'explanation_timeout': True} si el calculo supera los 5 segundos.
    """
    _, feature_names, _ = _load()
    X = np.array([[float(features.get(f, 0.0)) for f in feature_names]])

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_compute_shap, X)
            vals = future.result(timeout=_EXPLAIN_TIMEOUT)
    except _FuturesTimeout:
        return {"explanation_timeout": True, "top_features": [], "summary": ""}
    except Exception as exc:
        return {
            "explanation_timeout": False,
            "error": str(exc),
            "top_features": [],
            "summary": f"SHAP error: {exc}",
        }

    top_idx = np.argsort(np.abs(vals))[::-1][:5]
    top_features = [
        {
            "feature": feature_names[i],
            "shap_value": round(float(vals[i]), 4),
            "direction": "+" if vals[i] >= 0 else "-",
            "feature_value": round(float(X[0, i]), 4),
        }
        for i in top_idx
    ]

    summary = "Top features: " + ", ".join(
        f"{f['feature']} ({f['direction']}{abs(f['shap_value']):.3f})"
        for f in top_features
    )

    return {
        "explanation_timeout": False,
        "top_features": top_features,
        "summary": summary,
    }
