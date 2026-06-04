# IDS Agent — Etapa 3
**IC6200 Inteligencia Artificial — Grupo 2 | I Semestre 2026**  
Andrés Arias Corrales - Gabriel Solano Coronado - Pavel Zamora Araya

---

## Descripción

Sistema multi-agente IDS basado en reglas que usa un modelo DecisionTree (Etapa 2) como núcleo de inteligencia. Un `OrchestratorAgent` coordina cuatro sub-agentes especializados: detección, explicabilidad SHAP, contexto histórico KNN y decisión final. No utiliza ningún LLM externo.

---

## Estructura

```
src/
├── agents/
│   ├── __init__.py
│   ├── orchestrator_agent.py — coordina sub-agentes, expone session_stats()
│   ├── detector_agent.py     — wrappea predict_flow
│   ├── explainer_agent.py    — wrappea explain_flow (activación condicional)
│   ├── context_agent.py      — wrappea query_history (activación condicional)
│   └── decision_agent.py     — check_threshold + build_reasoning + log_run
├── tools/
│   ├── __init__.py
│   ├── predict_flow.py       — inferencia DT + validación de entrada
│   ├── explain_flow.py       — SHAP TreeExplainer (timeout 5 s)
│   ├── check_threshold.py    — clasificación en zonas operativas
│   ├── query_history.py      — KNN k=5 sobre top-5 features SHAP
│   └── agent_logger.py       — log JSONL anonimizado (SHA-256)
├── etapa3_01_agente_mvp.py   — CLI: instancia OrchestratorAgent y expone run_agent()
├── etapa3_02_validacion.py   — 13 escenarios de validación
└── etapa3_03_etica_robustez.py — análisis FP, sesgo, sensibilidad

outputs/etapa3/
├── agent_run_log.jsonl         — log acumulado de ejecuciones
├── validacion_resultados.csv   — resultados por escenario
├── validacion_reporte.txt      — reporte de criterios de aceptación
├── etica_robustez_reporte.json — análisis completo en JSON
├── etica_robustez_reporte.md   — reporte legible en Markdown
└── robustez_sensibilidad.png   — gráfica de perturbación de feature
```

---

## Requisitos previos

Los artefactos de Etapa 2 deben estar en `outputs/etapa2/`:

| Archivo | Descripción |
|---|---|
| `mejor_modelo_DT_base.joblib` | Modelo DecisionTree serializado |
| `split_data.joblib` | Split 60/20/20 con feature names |
| `selected_threshold.json` | Umbral operativo (0.05) |
| `feature_names.json` | Lista de 68 features |

Si no existen, correr primero el pipeline de Etapa 2 (ver `README.md`).

---

## Ejecución

Todos los comandos se corren desde la **raíz del repositorio**.

### Agente principal

```powershell
# Demo: 1 flujo ATTACK + 1 flujo BENIGN del test set
python src/etapa3_01_agente_mvp.py

# Batch: N flujos estratificados (default 200)
python src/etapa3_01_agente_mvp.py --batch
python src/etapa3_01_agente_mvp.py --batch --n 500

# Flujo desde archivo JSON
python src/etapa3_01_agente_mvp.py --input mi_flujo.json

# Resumen del log acumulado
python src/etapa3_01_agente_mvp.py --log-summary
```

### Validación

```powershell
python src/etapa3_02_validacion.py
```

Ejecuta 13 escenarios (normal, stress, edge, failure) y guarda resultados en `outputs/etapa3/`.

### Ética y robustez

```powershell
python src/etapa3_03_etica_robustez.py
```

Genera análisis de FP por puerto, tabla de sesgo, privacidad, sensibilidad de features y recalibración de umbrales.

---

## Formato de entrada (--input)

JSON plano con los nombres de feature como claves. Solo se necesitan incluir las features relevantes — las faltantes se rellenan automáticamente con `0.0`.

```json
{
  "Destination Port": 80,
  "Flow Bytes/s": 12345.6,
  "Init_Win_bytes_backward": 0,
  "min_seg_size_forward": 20,
  "Fwd IAT Min": 0.0
}
```

---

## Arquitectura multi-agente

```
OrchestratorAgent.run(features)
    |
    v
DetectorAgent           — siempre activo
    predict_flow -> prob_attack, label, missing_features
    |
    v (si prob > 0.03)
ExplainerAgent          — activación condicional
    explain_flow -> top-5 SHAP features
    |
    v (si zona uncertain o features faltantes)
ContextAgent            — activación condicional
    query_history -> attack_ratio, historical_signal (KNN k=5)
    |
    v
DecisionAgent           — siempre activo
    check_threshold -> zona operativa
    build_reasoning -> síntesis por reglas (sin LLM)
    log_run -> JSONL anonimizado
    |
    v
SALIDA: {predict, explain, threshold, history, reasoning,
         tools_called, total_latency_ms}
```

Cada sub-agente mantiene contadores propios (`total_runs`, `total_skipped`).  
El orquestador expone `session_stats()` con métricas agregadas de la sesión.

---

## Zonas operativas

| Zona | Condición | Acción |
|---|---|---|
| `high_attack` | prob ≥ 0.40 | BLOCK |
| `uncertain` | 0.05 ≤ prob < 0.40 | INVESTIGATE |
| `monitor` | 0.02 < prob < 0.05 | MONITOR |
| `high_benign` | prob ≤ 0.02 | ALLOW |

---

## Resultados de validación

| Criterio | Valor obtenido | Requerido |
|---|:---:|:---:|
| Recall ATTACK | **1.0000** | ≥ 0.95 |
| Latencia P95 | **< 1 ms** | < 3000 ms |
| Fallos sin crash | **2/2** | todos |
| scope_warning en ATTACK | **True** | True |
| Accuracy global | **0.9993** | ≥ 0.90 |

**VEREDICTO: 13/13 escenarios PASS**

---

## Privacidad y logs

- Las features nunca se almacenan en claro
- Cada entrada se identifica con un hash SHA-256 de 16 caracteres (`features_hash`)
- El dataset CIC-IDS2017 no contiene IPs reales — sin riesgo de re-identificación
- Campos en log: `timestamp`, `features_hash`, `tools_called`, `prob_attack`, `decision`, `zone`, `dominant_feature`, `total_latency_ms`, `ground_truth`, `correct`

---

## Limitaciones conocidas

- **23 falsos positivos** concentrados al 100% en puerto 80 (tráfico HTTP benigno)
- SQL Injection tiene soporte de entrenamiento limitado (21 instancias) — el modelo puede no generalizar a variantes no vistas
- Cobertura restringida al subconjunto Thursday Morning Web Attacks de CIC-IDS2017
- El DT produce probabilidades binarias (0 o 1) — el umbral no afecta al modelo en la práctica
