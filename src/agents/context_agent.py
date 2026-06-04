"""
ContextAgent — encargado de consultar el historial de flujos similares via KNN.

Se activa si el flujo cae en zona incierta (umbral <= prob <= 0.40)
o si hay features faltantes en la entrada.

Entrada : dict de features + predict_result
Salida  : dict con k, attack_ratio, historical_signal, neighbor_distances,
          neighbor_labels  (o {'context_skipped': True} si no aplica)
"""
from __future__ import annotations

from tools.query_history import query_history

_HIGH_ATTACK_ZONE = 0.40


class ContextAgent:
    """Agente de contexto historico: llama a query_history con logica de activacion condicional."""

    name = "ContextAgent"

    def __init__(self) -> None:
        self._runs: int = 0
        self._skipped: int = 0

    def should_run(self, predict_result: dict) -> bool:
        prob = predict_result.get("prob_attack", 0.0)
        thr = predict_result.get("threshold_used", 0.05)
        in_uncertain_zone = thr <= prob <= _HIGH_ATTACK_ZONE
        has_missing = bool(predict_result.get("missing_features"))
        return in_uncertain_zone or has_missing

    def run(self, features: dict, predict_result: dict) -> dict:
        """
        Busca los 5 flujos mas similares en el test set cuando la prediccion es incierta.
        Retorna {'context_skipped': True} si las condiciones de activacion no se cumplen.
        """
        if not self.should_run(predict_result):
            self._skipped += 1
            return {"context_skipped": True}

        result = query_history(features)
        self._runs += 1
        result["_agent"] = self.name
        return result

    @property
    def total_runs(self) -> int:
        return self._runs

    @property
    def total_skipped(self) -> int:
        return self._skipped
