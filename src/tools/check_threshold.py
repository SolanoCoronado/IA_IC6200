"""
check_threshold — clasifica la prediccion en zonas operativas y determina
la accion recomendada y la advertencia de alcance segun el nivel de confianza.
"""
from __future__ import annotations

_SCOPE_WARNING = (
    "SQL Injection training support is limited (21 instances). "
    "Model generalization to novel SQL Injection variants may be unreliable. "
    "Coverage: CIC-IDS2017 Thursday Morning Web Attacks only."
)

# Zone boundaries
_HIGH_ATTACK_THRESH = 0.40
_HIGH_BENIGN_THRESH = 0.02


def check_threshold(predict_result: dict) -> dict:
    """
    Zonas operativas:
      high_attack  : prob >= 0.40
      uncertain    : umbral <= prob < 0.40
      monitor      : 0.02 < prob < umbral  (clasificado BENIGN pero sobre el nivel de ruido)
      high_benign  : prob <= 0.02
    """
    prob = predict_result["prob_attack"]
    threshold = predict_result["threshold_used"]

    if prob >= _HIGH_ATTACK_THRESH:
        zone = "high_attack"
        recommended_action = (
            "BLOCK: high-confidence attack detected. "
            "Initiate incident response immediately."
        )
        scope_warning = _SCOPE_WARNING

    elif prob >= threshold:
        zone = "uncertain"
        recommended_action = (
            "INVESTIGATE: probability in uncertain zone. "
            "Manual analyst review required before action."
        )
        scope_warning = _SCOPE_WARNING

    elif prob <= _HIGH_BENIGN_THRESH:
        zone = "high_benign"
        recommended_action = "ALLOW: high-confidence benign traffic."
        scope_warning = None

    else:
        zone = "monitor"
        recommended_action = (
            "MONITOR: low attack probability, below operational threshold. "
            "Log and continue passive observation."
        )
        scope_warning = None

    return {
        "zone": zone,
        "prob_attack": prob,
        "recommended_action": recommended_action,
        "scope_warning": scope_warning,
        "thresholds": {
            "operational": threshold,
            "high_attack": _HIGH_ATTACK_THRESH,
            "high_benign": _HIGH_BENIGN_THRESH,
        },
    }
