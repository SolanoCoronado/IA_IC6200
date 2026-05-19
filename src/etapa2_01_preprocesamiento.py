"""
Etapa 2 – Paso 1: Preprocesamiento reproducible
Dataset: CIC-IDS2017 Thursday Morning Web Attacks
Objetivo: limpiar y normalizar el dataset para clasificación BENIGN/ATTACK
"""

import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ─── Configuración ────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "etapa2")

# Columnas constantes identificadas en EDA (no aportan señal predictiva)
CONSTANT_COLS = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "CWE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]

# Pares de columnas exactamente duplicadas: (conservar, eliminar en ablación)
DUPLICATE_PAIRS = [
    ("Total Fwd Packets",           "Subflow Fwd Packets"),
    ("Total Backward Packets",      "Subflow Bwd Packets"),
    ("Total Length of Fwd Packets", "Subflow Fwd Bytes"),
    ("Fwd Packet Length Mean",      "Avg Fwd Segment Size"),
    ("Fwd PSH Flags",               "SYN Flag Count"),
    ("Fwd Header Length",           "Fwd Header Length.1"),
    ("RST Flag Count",              "ECE Flag Count"),
]

# Columnas donde valores negativos son físicamente inválidos
# Excepción: Init_Win_bytes_forward e Init_Win_bytes_backward (-1 es válido)
NONNEG_COLS = [
    "Flow Duration",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean",   "Active Std",   "Active Max",   "Active Min",
    "Idle Mean",     "Idle Std",     "Idle Max",     "Idle Min",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 3.1 Carga y normalización de nombres ─────────────────────────────────────
print("=" * 60)
print("ETAPA 2 – Preprocesamiento")
print("=" * 60)
print(f"\n[3.1] Cargando {DATA_PATH} ...")

df = pd.read_csv(DATA_PATH)
original_columns = df.columns.tolist()
stripped_columns = df.columns.str.strip().tolist()
columns_with_spaces = [col for col in original_columns if col != col.strip()]

# Normalizar nombres y validar variable objetivo
assert "Label" in [col.strip() for col in original_columns], "ERROR: columna 'Label' no encontrada tras strip()"

df.columns = stripped_columns
shape_inicial = df.shape
print(f"      Shape inicial: {shape_inicial[0]} filas × {shape_inicial[1]} columnas")

# Estadísticas verificadas en el CSV original
numeric_columns = df.drop(columns=["Label"]).select_dtypes(include=[np.number])
int_cols = numeric_columns.select_dtypes(include=["integer"]).shape[1]
float_cols = numeric_columns.select_dtypes(include=["floating"]).shape[1]
non_numeric = df.drop(columns=["Label"]).select_dtypes(exclude="number").columns.tolist()
missing_original = df.isna().sum().sum()
inf_original = np.isinf(df.drop(columns=["Label"]).values).sum()
exact_duplicates = df.duplicated().sum()

print("\n[1] Hallazgos verificados en el CSV")
print(f"      Filas originales: {shape_inicial[0]:,}")
print(f"      Columnas originales: {shape_inicial[1]}")
print(f"      Variables predictoras: {shape_inicial[1] - 1}")
print(f"      Variable objetivo: Label")
print(f"      Columnas numéricas enteras: {int_cols}")
print(f"      Columnas numéricas flotantes: {float_cols}")
print(f"      Columnas categóricas: 1, Label")
print(f"      Columnas con espacios al inicio/final del nombre: {len(columns_with_spaces)}")
print(f"      Duplicados exactos: {exact_duplicates:,}")
print(f"      Valores faltantes originales: {missing_original:,}")
print(f"      Valores infinitos: {inf_original:,}")
print(f"      Columnas constantes: {len(CONSTANT_COLS)}")

label_counts = df["Label"].value_counts()
for label_name, count in label_counts.items():
    print(f"      {label_name}: {count:,}")

if non_numeric:
    print(f"      Columnas no numéricas (se eliminan): {non_numeric}")
else:
    print("      Todos los predictores son numéricos. OK")

# ─── 3.2 Normalización de etiquetas ──────────────────────────────────────────
print("\n[3.2] Normalizando etiquetas ...")
dist_original = df["Label"].value_counts().to_dict()
print(f"      Distribución original: {len(dist_original)} clases")

# Regla segura: cualquier etiqueta distinta de "BENIGN" es ATTACK
# Esto evita dependencia del carácter de codificación en los nombres de ataque
df["Label"] = (df["Label"] != "BENIGN").astype(int)

n_benign = (df["Label"] == 0).sum()
n_attack = (df["Label"] == 1).sum()
print(f"      BENIGN=0: {n_benign:,}   ATTACK=1: {n_attack:,}")
print(f"      Proporción ATTACK: {n_attack / len(df) * 100:.2f}%")

# ─── 3.3 Limpieza de valores inválidos ────────────────────────────────────────
print("\n[3.3] Limpieza de valores inválidos ...")

# Inf → NaN
features = df.drop(columns=["Label"])
n_inf = np.isinf(features.values).sum()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"      Valores Inf/−Inf reemplazados por NaN: {n_inf:,}")

# Duplicados exactos
n_dup = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"      Filas duplicadas exactas eliminadas: {n_dup:,}")

# Negativos inválidos en columnas conceptualmente no negativas
present_nonneg = [c for c in NONNEG_COLS if c in df.columns]
n_neg_total = 0
for col in present_nonneg:
    mask = df[col] < 0
    n_neg = mask.sum()
    if n_neg > 0:
        df.loc[mask, col] = np.nan
        n_neg_total += n_neg
print(f"      Negativos inválidos → NaN en {len(present_nonneg)} columnas: {n_neg_total:,} valores")

# Eliminar filas con NaN restantes
n_before_dropna = len(df)
df.dropna(inplace=True)
n_dropped_na = n_before_dropna - len(df)
print(f"      Filas con NaN eliminadas: {n_dropped_na:,}  → quedan {len(df):,} filas")

# ─── 3.4 Eliminar columnas constantes ────────────────────────────────────────
print("\n[3.4] Eliminando columnas constantes ...")
present_const = [c for c in CONSTANT_COLS if c in df.columns]
# Verificar cuáles siguen siendo constantes tras la limpieza
actually_const = [c for c in present_const if df[c].nunique() <= 1]
df.drop(columns=present_const, inplace=True)
n_predictoras_base = df.shape[1] - 1  # sin Label
print(f"      Columnas eliminadas: {len(present_const)}")
print(f"      Predictoras restantes (variante base): {n_predictoras_base}")

# Distribución final en variante base
n_b = (df["Label"] == 0).sum()
n_a = (df["Label"] == 1).sum()
print(f"      Distribución final: BENIGN={n_b:,}  ATTACK={n_a:,}")

# ─── Guardar variante BASE ────────────────────────────────────────────────────
out_base = os.path.join(OUTPUT_DIR, "datos_limpios_base.csv")
df.to_csv(out_base, index=False)
print(f"\n[BASE] Guardado: {out_base}  — {df.shape[0]:,} filas × {df.shape[1]} columnas")

# ─── 3.5 Variante ABLACIÓN (eliminar una columna de cada par duplicado) ───────
print("\n[3.5] Variante ablación: eliminando columnas duplicadas ...")
df_abl = df.copy()
cols_to_drop_abl = []
for keep, drop in DUPLICATE_PAIRS:
    if drop in df_abl.columns:
        cols_to_drop_abl.append(drop)

df_abl.drop(columns=cols_to_drop_abl, inplace=True)
n_predictoras_abl = df_abl.shape[1] - 1
print(f"      Columnas eliminadas por duplicación: {len(cols_to_drop_abl)}")
print(f"        {cols_to_drop_abl}")
print(f"      Predictoras restantes (variante ablación): {n_predictoras_abl}")

out_abl = os.path.join(OUTPUT_DIR, "datos_limpios_ablacion.csv")
df_abl.to_csv(out_abl, index=False)
print(f"[ABLACIÓN] Guardado: {out_abl}  — {df_abl.shape[0]:,} filas × {df_abl.shape[1]} columnas")

# ─── Reporte de texto ──────────────────────────────────────────────────────────
report = [
    "=" * 60,
    "REPORTE DE PREPROCESAMIENTO – ETAPA 2",
    "=" * 60,
    f"Dataset fuente:          {DATA_PATH}",
    f"Shape inicial:           {shape_inicial[0]} filas × {shape_inicial[1]} columnas",
    "",
    "1. Hallazgos verificados en el CSV",
    f"  Filas originales: {shape_inicial[0]:,}",
    f"  Columnas originales: {shape_inicial[1]}",
    f"  Variables predictoras: {shape_inicial[1] - 1}",
    "  Variable objetivo: Label",
    f"  Columnas numéricas enteras: {int_cols}",
    f"  Columnas numéricas flotantes: {float_cols}",
    "  Columnas categóricas: 1, Label",
    f"  Columnas con espacios al inicio/final del nombre: {len(columns_with_spaces)}",
    f"  Duplicados exactos: {exact_duplicates:,}",
    f"  Valores faltantes originales: {missing_original:,}",
    f"  Valores infinitos: {inf_original:,}",
    f"  Columnas constantes: {len(CONSTANT_COLS)}",
    "",
    "  Distribución original de etiquetas:",
] + [f"    {label_name}: {count:,}" for label_name, count in label_counts.items()] + [
    "",
    "  Distribución binaria original:",
    f"    BENIGN: {n_benign:,} ({n_benign / (n_benign + n_attack) * 100:.4f}%)",
    f"    ATTACK: {n_attack:,} ({n_attack / (n_benign + n_attack) * 100:.4f}%)",
    "",
    "3.1 Carga",
    f"  Columnas no numéricas eliminadas: {non_numeric if non_numeric else 'ninguna'}",
    "",
    "3.2 Etiquetas",
    f"  BENIGN final: {n_benign:,}",
    f"  ATTACK final: {n_attack:,}",
    f"  Proporción ATTACK: {n_attack / (n_benign + n_attack) * 100:.2f}%",
    "",
    "3.3 Limpieza",
    f"  Valores Inf/−Inf reemplazados:  {n_inf:,}",
    f"  Filas duplicadas eliminadas:    {n_dup:,}",
    f"  Negativos inválidos → NaN:      {n_neg_total:,}",
    f"  Filas con NaN eliminadas:       {n_dropped_na:,}",
    "",
    "3.4 Columnas constantes eliminadas",
    f"  {present_const}",
    "",
    "3.5 Columnas duplicadas (ablación)",
    f"  Pares: {DUPLICATE_PAIRS}",
    f"  Eliminadas en ablación: {cols_to_drop_abl}",
    "",
    "Resultados",
    f"  Variante base:     {df.shape[0]:,} filas × {n_predictoras_base} predictoras",
    f"  Variante ablación: {df_abl.shape[0]:,} filas × {n_predictoras_abl} predictoras",
    f"  BENIGN final: {n_b:,}   ATTACK final: {n_a:,}",
]

report_path = os.path.join(OUTPUT_DIR, "reporte_preprocesamiento.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"\nReporte guardado: {report_path}")
print("\nPreprocesamiento completado exitosamente.")
