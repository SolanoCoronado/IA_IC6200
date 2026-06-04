"""
ExplainerAgent — encargado de generar la explicacion SHAP del flujo.

Solo se activa si prob_attack > 0.03 (umbral de explicacion).
Entrada : dict de features + predict_result opcional
Salida  : dict con explanation_timeout, top_features, summary
          (o {'explanation_skipped': True} si prob es menor al umbral)
"""
from __future__ import annotations

from tools.explain_flow import explain_flow

_EXPLAIN_PROB_THRESHOLD = 0.03


class ExplainerAgent:
    """Agente de explicabilidad: llama a explain_flow con logica de activacion condicional."""

    name = "ExplainerAgent"

    def __init__(self, prob_threshold: float = _EXPLAIN_PROB_THRESHOLD) -> None:
        self._prob_threshold = prob_threshold
        self._runs: int = 0
        self._skipped: int = 0

    def should_run(self, prob_attack: float) -> bool:
        return prob_attack > self._prob_threshold

    def run(self, features: dict, predict_result: dict) -> dict:
        """
        Genera la explicacion SHAP si prob_attack supera el umbral configurado.
        Retorna {'explanation_skipped': True} si no se cumplen las condiciones.
        """
        if not self.should_run(predict_result.get("prob_attack", 0.0)):
            self._skipped += 1
            return {"explanation_skipped": True, "top_features": [], "summary": ""}

        result = explain_flow(features, prediction_result=predict_result)
        self._runs += 1
        result["_agent"] = self.name
        return result

    @property
    def total_runs(self) -> int:
        return self._runs

    @property
    def total_skipped(self) -> int:
        return self._skipped
