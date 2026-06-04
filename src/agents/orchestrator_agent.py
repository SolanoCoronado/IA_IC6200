"""
OrchestratorAgent — coordina DetectorAgent, ExplainerAgent,
ContextAgent y DecisionAgent en un pipeline secuencial.

Responsabilidades:
  - Decidir que sub-agentes activar segun los resultados intermedios
  - Registrar la lista de herramientas usadas
  - Medir la latencia total de extremo a extremo
  - Exponer estadisticas de la sesion a traves de session_stats()
"""
from __future__ import annotations

import time

from .detector_agent import DetectorAgent
from .explainer_agent import ExplainerAgent
from .context_agent import ContextAgent
from .decision_agent import DecisionAgent


class OrchestratorAgent:
    """
    Orquestador del sistema multi-agente IDS.

    Ejemplo de uso:
        orch = OrchestratorAgent()
        resultado = orch.run(features, ground_truth=1)
    """

    name = "OrchestratorAgent"

    def __init__(self) -> None:
        self._detector = DetectorAgent()
        self._explainer = ExplainerAgent()
        self._context = ContextAgent()
        self._decision = DecisionAgent()
        self._total_runs: int = 0
        self._errors: int = 0

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def run(self, features: dict, ground_truth=None) -> dict:
        """
        Ejecuta el pipeline completo sobre un flujo de red.

        Retorna un dict con las claves:
            predict, explain, threshold, history, reasoning,
            tools_called, total_latency_ms
        Si la entrada no es valida retorna {'error': <mensaje>, ...}.
        """
        t0 = time.perf_counter()
        tools_called: list[str] = []

        try:
            # 1. Deteccion — siempre
            predict_result = self._detector.run(features)
            tools_called.append("predict_flow")

            # 2. Explicacion — solo si prob > umbral
            explain_result = self._explainer.run(features, predict_result)
            if not explain_result.get("explanation_skipped"):
                tools_called.append("explain_flow")

            # 3. Contexto historico — solo en zona incierta o features faltantes
            history_result = self._context.run(features, predict_result)
            if not history_result.get("context_skipped"):
                tools_called.append("query_history")

            # 4. Decision — siempre (zona + razonamiento + log)
            total_ms = (time.perf_counter() - t0) * 1000
            decision_out = self._decision.run(
                features=features,
                predict_result=predict_result,
                explain_result=explain_result,
                history_result=history_result,
                tools_called=tools_called + ["check_threshold"],
                total_ms=total_ms,
                ground_truth=ground_truth,
            )
            tools_called.append("check_threshold")

            self._total_runs += 1
            return {
                "predict": predict_result,
                "explain": explain_result,
                "threshold": decision_out["threshold"],
                "history": history_result,
                "reasoning": decision_out["reasoning"],
                "tools_called": tools_called,
                "total_latency_ms": round(total_ms, 2),
            }

        except (ValueError, TypeError) as exc:
            total_ms = (time.perf_counter() - t0) * 1000
            self._errors += 1
            return {
                "error": str(exc),
                "predict": None,
                "explain": None,
                "threshold": None,
                "history": None,
                "reasoning": {
                    "text": f"ERROR: {exc}",
                    "lines": [f"ERROR: {exc}"],
                    "final_decision": "ERROR",
                    "confidence": "NONE",
                    "recommended_action": "Invalid input — fix before retry.",
                },
                "tools_called": tools_called,
                "total_latency_ms": round(total_ms, 2),
            }

    # ------------------------------------------------------------------
    # Estadisticas de sesion
    # ------------------------------------------------------------------

    def session_stats(self) -> dict:
        """Retorna un resumen de actividad de todos los sub-agentes en la sesion actual."""
        return {
            "orchestrator_runs": self._total_runs,
            "orchestrator_errors": self._errors,
            "detector_runs": self._detector.total_runs,
            "explainer_runs": self._explainer.total_runs,
            "explainer_skipped": self._explainer.total_skipped,
            "context_runs": self._context.total_runs,
            "context_skipped": self._context.total_skipped,
            "decision_runs": self._decision.total_runs,
        }
