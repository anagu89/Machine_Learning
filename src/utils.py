# src/utils.py
"""
Utilidades comunes de limpieza/enriquecimiento usadas por data_processing.py
- CleaningConfig: configuración (ATC/columns)
- ensure_datetime(df, col): asegura fecha tipo datetime y orden temporal
- compute_total_sales(df, atc_cols): convierte ATC a numérico y crea 'Total sales'
- add_time_parts(df, level): añade partes temporales coherentes por granularidad
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import pandas as pd
import numpy as np


__all__ = [
    "CleaningConfig",
    "ensure_datetime",
    "compute_total_sales",
    "add_time_parts",
]


# ===============================
# Configuración
# ===============================
@dataclass(frozen=True)
class CleaningConfig:
    """
    Configuración de limpieza.
    - atc_cols: columnas ATC (ventas por categoría) usadas para 'Total sales'.
    """
    atc_cols: List[str] = field(
        default_factory=lambda: [
            "M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"
        ]
    )


# ===============================
# Helpers internos
# ===============================
def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Convierte columnas a numérico (silencioso). Si no existen, las ignora.
    """
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


# ===============================
# Funciones públicas
# ===============================
def ensure_datetime(df: pd.DataFrame, date_col: str = "datum") -> pd.DataFrame:
    """
    Asegura que `date_col` sea datetime, ordena por fecha y resetea índice.

    - No elimina filas con NaT de forma automática (se deja a decisión del caller).
    - Mantiene el resto de columnas tal cual.
    """
    if date_col not in df.columns:
        raise KeyError(f"No existe la columna de fecha '{date_col}' en el DataFrame.")

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    # Orden temporal (NaT al final), índice limpio
    d = d.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    return d


def compute_total_sales(df: pd.DataFrame, atc_cols: Iterable[str]) -> pd.DataFrame:
    """
    Convierte columnas ATC a numérico y crea 'Total sales' como suma fila a fila.
    - Columnas inexistentes se ignoran.
    - Si ninguna columna ATC existe, crea 'Total sales' = NaN.
    """
    d = _coerce_numeric(df, atc_cols)
    present = [c for c in atc_cols if c in d.columns]

    if present:
        d["Total sales"] = d[present].sum(axis=1, skipna=True)
    else:
        # No hay columnas ATC presentes
        d["Total sales"] = np.nan
        print("[AVISO] Ninguna columna ATC presente; 'Total sales' se ha dejado como NaN.")

    return d


def add_time_parts(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """
    Añade variables temporales estándar según granularidad:
    - Siempre que exista 'datum':
        * Year, Month, Day, Weekday Number (1=Lun..7=Dom)
    - Para 'Hourly' añade además Hour.
    - No pisa columnas si ya existen (solo las crea si faltan).
    """
    if "datum" not in df.columns:
        raise KeyError("No existe la columna 'datum' para derivar partes temporales.")

    d = df.copy()
    # Asegurar datetime correcto (por si llega sin convertir)
    d["datum"] = pd.to_datetime(d["datum"], errors="coerce")

    # Partes comunes
    if "Year" not in d.columns:
        d["Year"] = d["datum"].dt.year
    if "Month" not in d.columns:
        d["Month"] = d["datum"].dt.month
    if "Day" not in d.columns:
        d["Day"] = d["datum"].dt.day
    if "Weekday Number" not in d.columns:
        # 1 = Lunes, ..., 7 = Domingo (como en los notebooks)
        d["Weekday Number"] = d["datum"].dt.dayofweek + 1

    # Partes específicas por granularidad
    level_norm = (level or "").strip().lower()
    if level_norm == "hourly":
        if "Hour" not in d.columns:
            d["Hour"] = d["datum"].dt.hour

    # Nota: no añadimos Week/Quarter para evitar ruido; se puede ampliar si lo necesitase.
    return d

# ========= UTILIDADES DE MODELADO / FEATURES (para training.py) =========
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ATC_COLS_DEFAULT = ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluar(y_true, y_pred):
    return {
        "MAPE": mape(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

def temporal_split(df, date_col="datum", test_frac=0.2):
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    cut = int(n * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def add_time_features_daily(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Features para Daily: calendario + lags + medias móviles cortas (+ATC si existen)
    Devuelve (dataframe_con_features, lista_de_columnas_de_features)
    """
    d = df.copy()
    d["datum"] = pd.to_datetime(d["datum"], errors="coerce")
    d = d.sort_values("datum").reset_index(drop=True)

    d["Year"] = d["datum"].dt.year
    d["Month"] = d["datum"].dt.month
    d["Weekday"] = d["datum"].dt.dayofweek + 1  # 1..7

    d["lag_1"]  = d["Total sales"].shift(1)
    d["lag_7"]  = d["Total sales"].shift(7)
    d["roll7"]  = d["Total sales"].shift(1).rolling(7, min_periods=3).mean()
    d = d.dropna().reset_index(drop=True)

    feat_cols = ["Year","Month","Weekday","lag_1","lag_7","roll7"]

    atc_present = [c for c in ATC_COLS_DEFAULT if c in d.columns]
    feat_cols += atc_present
    return d, feat_cols

def make_supervised_daily(df: pd.DataFrame):
    """
    Devuelve X, y, fechas, feat_cols para Daily usando add_time_features_daily
    """
    d, feat_cols = add_time_features_daily(df)
    d = d.dropna(subset=["Total sales"])
    X = d[feat_cols].copy()
    y = d["Total sales"].astype(float).values
    fechas = d["datum"].values
    return X, y, fechas, feat_cols

def train_test_last_fraction(X, y, frac=0.2):
    """
    Split temporal simple: el último 'frac' como test
    """
    n = len(X)
    cut = int(n*(1-frac))
    return X[:cut], X[cut:], y[:cut], y[cut:]