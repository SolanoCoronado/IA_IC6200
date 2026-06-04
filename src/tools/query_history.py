"""
query_history — busqueda KNN (k=5) sobre X_test usando distancia
euclidiana normalizada en las 5 features mas importantes segun SHAP del modelo DT.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUTS_E2 = _REPO_ROOT / "outputs" / "etapa2"

# Top-5 features por |SHAP| promedio — fijadas desde el analisis de Etapa 2
TOP5_FEATURES = [
    "Init_Win_bytes_backward",
    "min_seg_size_forward",
    "Destination Port",
    "Fwd IAT Min",
    "Flow Bytes/s",
]

_X_test_norm: np.ndarray | None = None
_y_test_arr: np.ndarray | None = None
_scaler: Any = None
_feature_names: list | None = None
_top5_idx: list | None = None


def _normalize_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("/", "").replace(".", "")


def _load() -> tuple:
    global _X_test_norm, _y_test_arr, _scaler, _feature_names, _top5_idx
    if _X_test_norm is None:
        data = joblib.load(_OUTPUTS_E2 / "split_data.joblib")
        X_test = data["X_test"]
        y_test = data["y_test"]
        _feature_names = list(data["feature_names"])

        X_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test, dtype=float)
        _y_test_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

        # Busca los indices de las top-5 features (tolerante a variaciones menores en nombres)
        norm_fn = [_normalize_name(f) for f in _feature_names]
        _top5_idx = []
        for feat in TOP5_FEATURES:
            norm_feat = _normalize_name(feat)
            if norm_feat in norm_fn:
                _top5_idx.append(norm_fn.index(norm_feat))

        X_top5 = X_arr[:, _top5_idx].astype(float)
        np.nan_to_num(X_top5, copy=False)

        _scaler = StandardScaler()
        _X_test_norm = _scaler.fit_transform(X_top5)
    return _X_test_norm, _y_test_arr, _scaler, _top5_idx


def query_history(features: dict, k: int = 5) -> dict:
    """
    Retorna attack_ratio y la senal historica a partir de los k vecinos mas cercanos del test set.
    """
    X_norm, y_arr, scaler, top5_idx = _load()

    query_vals = np.array(
        [[float(features.get(TOP5_FEATURES[i], 0.0)) for i in range(len(top5_idx))]]
    )
    query_norm = scaler.transform(query_vals)

    dists = np.sqrt(np.sum((X_norm - query_norm) ** 2, axis=1))
    knn_idx = np.argsort(dists)[:k]

    neighbors = y_arr[knn_idx]
    attack_ratio = float(np.mean(neighbors))

    if attack_ratio >= 0.60:
        historical_signal = "HIGH_ATTACK"
    elif attack_ratio >= 0.20:
        historical_signal = "UNCERTAIN"
    else:
        historical_signal = "BENIGN"

    return {
        "k": k,
        "attack_ratio": round(attack_ratio, 3),
        "historical_signal": historical_signal,
        "neighbor_distances": [round(float(d), 4) for d in dists[knn_idx]],
        "neighbor_labels": neighbors.tolist(),
    }
