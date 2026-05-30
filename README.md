# IDS-CIC2017 — Detección de Intrusiones con Machine Learning
**IC6200 Inteligencia Artificial — Grupo 2 | I Semestre 2026**  
Andrés Arias Corrales - Gabriel Solano Coronado - Pavel Zamora Araya

---

## Descripción

Sistema de detección de intrusiones en tráfico de red mediante clasificación supervisada binaria (BENIGN vs ATTACK) sobre el subconjunto Thursday Morning Web Attacks del dataset CIC-IDS2017. Implementa preprocesamiento reproducible, baselines, modelos principales con tuning sistemático, XAI con SHAP y análisis de errores completo.

**Track:** ML Tabular/Clásico  
**Métrica principal:** Recall ATTACK (minimizar ataques no detectados)  
**Dataset:** CIC-IDS2017 — Thursday Morning Web Attacks

---

## Entregas

| Etapa | Descripción |
|-------|-------------|
| Etapa 1 | Análisis del problema y diseño de la solución |
| **Etapa 2** | **Modelado, Entrenamiento, Explicabilidad y Evaluación** |
| Etapa 3 | Integración en Agentes Inteligentes y Validación Científica |

### Reporte Etapa 2

**[Etapa2_IA.docx](https://estudianteccr-my.sharepoint.com/:w:/g/personal/andco97_estudiantec_cr/IQDddfcKzbWCR6ffVwCquC_KAeO-nAX3mNa5cJyTsJmVoUk?e=ihX8SY)** — Reporte completo de modelado, entrenamiento, XAI y evaluación.

Cubre: 
- Protocolo experimental 
- Auditoría de leakage 
- Baselines 
- Tuning sistemático 
- Selección de umbral 
- Diagnóstico de overfitting 
- Análisis de errores (FP/FN) 
- SHAP global y local 
- Limitaciones y conclusiones

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
│   ├── etapa2_07_gaps.py
│   └── requirements.txt
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
└── README.md
```

---

## Requisitos

Python 3.10+ recomendado.

```bash
pip install -r src/requirements.txt
```

Dependencias (`src/requirements.txt`):
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
Normaliza nombres de columnas, convierte etiquetas a binario (BENIGN=0, ATTACK=1), elimina duplicados exactos (6,066), infinitos (250), negativos inválidos (317) y columnas constantes (10). Produce `datos_limpios_base.csv` (163,916 filas × 69 columnas) y `datos_limpios_ablacion.csv`.

### Paso 2 — Modelos principales + tuning

```bash
python etapa2_02_modelos.py
```
Entrena LR, DT, RF y HGB con configuración base y tuned (RandomizedSearchCV, 15 iter, 5-fold CV, scoring=recall). Guarda el mejor modelo serializado (`mejor_modelo_DT_base.joblib`) y el split estratificado 60/20/20. **Tarda ~5-10 minutos.**

### Paso 3 — Baselines completos

```bash
python etapa2_03_baselines_completos.py
```
Entrena Dummy MostFrequent, Dummy Stratified y LogReg Balanced. Genera matrices de confusión, curvas PR/ROC y tabla comparativa de métricas.

### Paso 4 — XAI con SHAP + selección de umbral evaluación final en test

```bash
python etapa2_04_shap_xai.py
```
Selecciona umbral óptimo sobre validación (criterio: máximo recall con precision ≥ 0.20). **Evalúa en test por primera y única vez.** Genera SHAP global (bar + beeswarm) y SHAP local (TP×2, FP×2, TN×1).

### Paso 5 — Análisis de errores + artefactos Etapa 3

```bash
python etapa2_05_error_analysis.py
```
Analiza FP/FN con características observadas, diagnóstico de overfitting (gap train-validation), curva de aprendizaje (PR-AUC vs tamaño de entrenamiento) y variabilidad por fold. Guarda artefactos para Etapa 3.

### Paso 6 — Auditoría de leakage

```bash
python etapa2_06_auditoria_leakage.py
```
Verifica ausencia de data leakage: revisión de columnas sospechosas por nombre, AUC individual por feature, ablación de variables dominantes (DT_reduced, HGB_reduced) y distribución de probabilidades del modelo.

### Paso 7 — Gaps pendientes

```bash
python etapa2_07_gaps.py
```
Genera distribución final de clases, documenta FN=0 formalmente y reporta media/std de recall por fold en validación cruzada.

---

## Reproducibilidad

- Semilla global: `RANDOM_STATE = 42` en todos los scripts
- Split estratificado 60/20/20 guardado en `split_data.joblib` — compartido por todos los scripts
- Umbral operativo guardado en `selected_threshold.json` (valor: 0.05)
- Test permanece ciego hasta la evaluación final en `etapa2_04_shap_xai.py`
- Metadatos del modelo en `model_metadata.json`; ejemplo de inferencia en `inference_example.py`

---

## Resultados principales

Evaluación final en test con umbral 0.05 (seleccionado sobre validación):

| Modelo | Recall ATTACK | Precision ATTACK | F1 macro | PR-AUC | ROC-AUC |
|--------|:---:|:---:|:---:|:---:|:---:|
| Dummy MostFrequent | 0.000 | 0.000 | 0.497 | 0.013 | 0.500 |
| Dummy Stratified | 0.005 | 0.005 | 0.496 | 0.013 | 0.496 |
| LogReg Balanced | 0.993 | 0.389 | 0.775 | 0.710 | 0.997 |
| **DT tuned**  | **1.000** | **0.949** | **0.987** | **0.991** | **0.9999** |
| HGB tuned | 0.998 | 1.000 | 0.999 | 1.000 | 1.000 |
| RF tuned | 0.981 | 0.998 | 0.995 | 0.998 | 1.000 |

**Modelo seleccionado: Decision Tree tuned** — Recall ATTACK = 1.0, FN = 0, TP = 429, FP = 23  
Validación cruzada adicional: recall medio = 0.9938 ± 0.0053

---

## Limitaciones conocidas

- El dataset cubre únicamente tres tipos de ataque web (Brute Force, XSS, SQL Injection) en entorno controlado; la generalización a producción no está garantizada.
- SQL Injection contaba con solo 21 instancias originales — capacidad de generalización estadísticamente limitada para ese subtipo.
- La auditoría detectó variables dominantes y probabilidades saturadas; las métricas casi perfectas deben interpretarse con cautela.
- El umbral 0.05 prioriza recall y acepta 23 falsos positivos en test.

---