"""
DecisionAgent — encargado de clasificar la zona operativa, sintetizar
el razonamiento y registrar el resultado en el log.

Combina check_threshold + build_reasoning + log_run en un solo agente
que produce el veredicto final y lo guarda.
"""
from __future__ import annotations

from tools.check_threshold import check_threshold
from tools.agent_logger import log_run


class DecisionAgent:
    """Agente de decision: determina la zona operativa, genera razonamiento por reglas y registra en log."""

    name = "DecisionAgent"

    def __init__(self) -> None:
        self._runs: int = 0

    # ------------------------------------------------------------------
    # Razonamiento basado en reglas (sin LLM externo)
    # ------------------------------------------------------------------

    @staticmethod
    def build_reasoning(
        predict_result: dict,
        explain_result: dict | None,
        threshold_result: dict,
        history_result: dict | None,
    ) -> dict:
        lines: list[str] = []

        prob = predict_result["prob_attack"]
        label = predict_result["label"]
        zone = threshold_result["zone"]

        lines.append(
            f"[PREDICT] prob_attack={prob:.4f}  label={label}  zone={zone}"
        )

        if predict_result["missing_features"]:
            n = len(predict_result["missing_features"])
            lines.append(
                f"[WARN] {n} missing feature(s) filled with 0.0: "
                f"{predict_result['missing_features'][:5]}"
                + (" ..." if n > 5 else "")
            )

        if explain_result is not None and not explain_result.get("explanation_skipped"):
            if explain_result.get("explanation_timeout"):
                lines.append("[EXPLAIN] Timeout — SHAP explanation unavailable (>5 s).")
            elif explain_result.get("error"):
                lines.append(f"[EXPLAIN] Error: {explain_result['error']}")
            else:
                lines.append(f"[EXPLAIN] {explain_result.get('summary', '')}")

        lines.append(f"[ACTION] {threshold_result['recommended_action']}")

        if threshold_result.get("scope_warning"):
            lines.append(f"[SCOPE] {threshold_result['scope_warning']}")

        if history_result is not None and not history_result.get("context_skipped"):
            lines.append(
                f"[HISTORY] k=5 neighbours  "
                f"attack_ratio={history_result['attack_ratio']:.3f}"
                f"  signal={history_result['historical_signal']}"
            )
            if label == "BENIGN" and history_result["historical_signal"] == "HIGH_ATTACK":
                lines.append(
                    "[HISTORY-ALERT] Neighbours are predominantly ATTACK — "
                    "consider manual review despite BENIGN prediction."
                )

        confidence = "HIGH" if zone in ("high_attack", "high_benign") else "MODERATE"
        lines.append(f"[CONFIDENCE] {confidence}")

        return {
            "lines": lines,
            "text": "\n".join(lines),
            "confidence": confidence,
            "final_decision": label,
            "recommended_action": threshold_result["recommended_action"],
        }

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(
        self,
        features: dict,
        predict_result: dict,
        explain_result: dict | None,
        history_result: dict | None,
        tools_called: list[str],
        total_ms: float,
        ground_truth=None,
    ) -> dict:
        """
        Clasifica la zona operativa, construye el razonamiento y escribe en el log.
        Retorna threshold_result y reasoning listos para el orquestador.
        """
        threshold_result = check_threshold(predict_result)

        reasoning = self.build_reasoning(
            predict_result, explain_result, threshold_result, history_result
        )

        log_run(
            features=features,
            tools_called=tools_called,
            predict_result=predict_result,
            explain_result=explain_result or {},
            threshold_result=threshold_result,
            history_result=history_result or {},
            reasoning=reasoning,
            total_ms=total_ms,
            ground_truth=ground_truth,
        )

        self._runs += 1
        return {
            "threshold": threshold_result,
            "reasoning": reasoning,
            "_agent": self.name,
        }

    @property
    def total_runs(self) -> int:
        return self._runs
