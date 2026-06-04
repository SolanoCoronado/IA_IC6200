"""
DetectorAgent — encargado de realizar la prediccion con el modelo DT.

Entrada : dict de features (nombres originales del dataset)
Salida  : dict con prob_attack, prediction, label, threshold_used,
          missing_features, latency_ms
"""
from __future__ import annotations

from tools.predict_flow import predict_flow


class DetectorAgent:
    """Agente de deteccion: llama a predict_flow y lleva conteo de ejecuciones."""

    name = "DetectorAgent"

    def __init__(self) -> None:
        self._runs: int = 0

    def run(self, features: dict) -> dict:
        """
        Ejecuta la prediccion sobre el flujo de red.
        Lanza ValueError o TypeError si los datos de entrada no son validos.
        """
        result = predict_flow(features)
        self._runs += 1
        result["_agent"] = self.name
        return result

    @property
    def total_runs(self) -> int:
        return self._runs
