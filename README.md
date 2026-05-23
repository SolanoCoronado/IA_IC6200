# IDS-CIC2017 — Detección de Intrusiones con Machine Learning
**IC6200 Inteligencia Artificial — Grupo 2 | I Semestre 2026**  
Andrés Arias Corrales - Gabriel Solano Coronado - Pavel Zamora Araya

---

## Descripción

Sistema de detección de intrusiones en tráfico de red mediante clasificación supervisada binaria (BENIGN vs ATTACK) sobre el subconjunto Thursday Morning Web Attacks del dataset CIC-IDS2017. Implementa preprocesamiento reproducible, baselines, modelos principales con tuning sistemático, XAI con SHAP y análisis de errores completo.

---

## Estructura del repositorio

```
IA_IC6200/
├── data/
│   └── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv  ← dataset original
├── src/
│   ├── etapa2_01_preprocesamiento.py
│   ├── etapa2_02_modelos.py
│   ├── etapa2_03_baselines_completos.py
│   ├── etapa2_04_shap_xai.py
│   ├── etapa2_05_error_analysis.py
│   ├── etapa2_06_auditoria_leakage.py
│   └── etapa2_07_gaps.py
├── outputs/
│   └── etapa2/
│       ├── datos_limpios_base.csv
│       ├── datos_limpios_ablacion.csv
│       ├── split_data.joblib
│       ├── mejor_modelo_DT_base.joblib
│       ├── selected_threshold.json
│       ├── feature_names.json
│       ├── label_mapping.json
│       ├── model_metadata.json
│       ├── inference_example.py
│       ├── shap/
│       └── error_analysis/
├── README.md
└── requirements.txt
```

---

## Requisitos

Python 3.10+ recomendado.

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
shap
joblib
```

---

## Orden de ejecución

Todos los scripts se corren desde la carpeta `src/`:

```bash
cd src/
```

### Paso 1 — Preprocesamiento
```bash
python etapa2_01_preprocesamiento.py
```
Limpia el CSV original: normaliza nombres, convierte etiquetas a binario, elimina duplicados, infinitos, negativos inválidos y columnas constantes. Produce `datos_limpios_base.csv` (163,916 filas × 69 columnas) y `datos_limpios_ablacion.csv`.

### Paso 2 — Modelos principales + tuning
```bash
python etapa2_02_modelos.py
```
Entrena LR, DT, RF y HGB con baseline y tuned (RandomizedSearchCV, 15 iter, 5-fold CV). Guarda el mejor modelo serializado y el split 60/20/20. **Tarda ~5-10 minutos.**

### Paso 3 — Baselines completos
```bash
python etapa2_03_baselines_completos.py
```
Entrena Dummy MostFrequent, Dummy Stratified y LogReg Balanced. Genera matrices de confusión, curvas y tabla comparativa.

### Paso 4 — XAI con SHAP + selección de umbral
```bash
python etapa2_04_shap_xai.py
```
Selecciona umbral óptimo sobre validación, evalúa en test (primera y única vez), genera SHAP global (bar + beeswarm) y SHAP local (TP×2, FP×2, TN×1).

### Paso 5 — Análisis de errores + artefactos Etapa 3
```bash
python etapa2_05_error_analysis.py
```
Analiza FP/FN, casos límite, diagnóstico overfitting, curva de aprendizaje. Guarda artefactos para Etapa 3.

### Paso 6 — Auditoría de leakage
```bash
python etapa2_06_auditoria_leakage.py
```
Verifica ausencia de data leakage: columnas sospechosas, AUC por feature individual, ablación de features dominantes, distribución de probabilidades.

### Paso 7 — Gaps pendientes
```bash
python etapa2_07_gaps.py
```
Genera distribución final de clases, documenta FN=0 formalmente y calcula media/std de CV por fold.

---

## Reproducibilidad

- Semilla global: `RANDOM_STATE = 42` en todos los scripts
- Split estratificado guardado en `split_data.joblib` — todos los scripts usan el mismo split
- Umbral seleccionado guardado en `selected_threshold.json`
- Test permanece ciego hasta evaluación final en `etapa2_04_shap_xai.py`

---

## Resultados principales

| Modelo | Recall ATTACK | Precision ATTACK | F1 macro | PR-AUC | ROC-AUC |
|--------|:---:|:---:|:---:|:---:|:---:|
| Dummy MostFrequent | 0.000 | 0.000 | 0.497 | 0.013 | 0.500 |
| Dummy Stratified | 0.005 | 0.005 | 0.496 | 0.013 | 0.496 |
| LogReg Balanced | 0.993 | 0.389 | 0.775 | 0.710 | 0.997 |
| **DT tuned** | **1.000** | **0.949** | **0.987** | **0.991** | **0.9999** |
| HGB tuned | 0.998 | 1.000 | 0.999 | 1.000 | 1.000 |
| RF tuned | 0.981 | 0.998 | 0.995 | 0.998 | 1.000 |

Umbral seleccionado: **0.05** sobre validación  
Mejor modelo: **Decision Tree tuned** (Recall ATTACK = 1.0, FN = 0)

---
