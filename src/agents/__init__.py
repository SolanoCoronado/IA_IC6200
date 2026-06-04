"""
agents — paquete multi-agente del sistema IDS.

Jerarquia:
    OrchestratorAgent
        DetectorAgent   -> predict_flow
        ExplainerAgent  -> explain_flow
        ContextAgent    -> query_history
        DecisionAgent   -> check_threshold + build_reasoning + log_run
"""
from .detector_agent import DetectorAgent
from .explainer_agent import ExplainerAgent
from .context_agent import ContextAgent
from .decision_agent import DecisionAgent
from .orchestrator_agent import OrchestratorAgent

__all__ = [
    "DetectorAgent",
    "ExplainerAgent",
    "ContextAgent",
    "DecisionAgent",
    "OrchestratorAgent",
]
