# src/plots.py
"""
Utilidades de gráficos para EDA y negocio.
Genera los mismos plots que en notebooks para Daily/Weekly/Monthly/Hourly.

Salida:
- docs/plots/*.png
- (opcional) docs/artifacts/*.csv (ranking categorías si hay ATC)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# Backend no interactivo para guardado en scripts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


ATC_COLS = ["M01AB","M01AE","N02BA","N02BE","N05B","N05C","R03","R06"]


# ----------------------------
# Helpers internos
# ----------------------------
def _ensure_dirs(plots_dir: Path, artifacts_dir: Path | None = None):
    plots_dir.mkdir(parents=True, exist_ok=True)
    if artifacts_dir:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

def _ensure_datetime_sorted(df: pd.DataFrame, date_col="datum") -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return d

def _ensure_total_sales(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Total sales" not in d.columns:
        cols = [c for c in ATC_COLS if c in d.columns]
        if cols:
            d["Total sales"] = pd.to_numeric(d[cols], errors="coerce").sum(axis=1, skipna=True)
        else:
            raise ValueError("No existe 'Total sales' ni ATC para construirla.")
    d["Total sales"] = pd.to_numeric(d["Total sales"], errors="coerce")
    return d

def _has_atc(df: pd.DataFrame) -> bool:
    return any(c in df.columns for c in ATC_COLS)


# ----------------------------
# Plots EDA genéricos por nivel
# ----------------------------
def plot_ts_line(df: pd.DataFrame, level: str, out: Path):
    d = _ensure_total_sales(_ensure_datetime_sorted(df))
    plt.figure(figsize=(12,4))
    plt.plot(d["datum"], d["Total sales"], lw=1.8)
    plt.title(f"Serie temporal — {level}")
    plt.xlabel("Fecha"); plt.ylabel("Total sales")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / f"ts_line_{level.lower()}.png", dpi=150)
    plt.close()

def plot_smoothed(df: pd.DataFrame, level: str, out: Path):
    d = _ensure_total_sales(_ensure_datetime_sorted(df))
    s = d["Total sales"].astype(float)

    if level.lower() == "daily":
        wins = [7, 30]
    elif level.lower() == "weekly":
        wins = [4, 12]
    elif level.lower() == "monthly":
        wins = [3, 12]
    elif level.lower() == "hourly":
        wins = [24, 168]
    else:
        wins = [7, 30]

    plt.figure(figsize=(12,4))
    plt.plot(d["datum"], s, label="Original", alpha=0.5)
    for w in wins:
        plt.plot(d["datum"], s.rolling(w, min_periods=max(2, w//3)).mean(), label=f"roll{w}", lw=1.8)
    plt.title(f"Serie suavizada — {level}")
    plt.xlabel("Fecha"); plt.ylabel("Total sales")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / f"serie_suavizada_{level.lower()}.png", dpi=150)
    plt.close()

def plot_hist_kde(df: pd.DataFrame, level: str, out: Path):
    from scipy.stats import gaussian_kde

    d = _ensure_total_sales(df)
    data = d["Total sales"].dropna().astype(float).values

    plt.figure(figsize=(12,4))
    count, bins, _ = plt.hist(data, bins=50, alpha=0.4, edgecolor="black")
    if len(data) >= 2 and not np.allclose(np.std(data), 0.0):
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 1000)
        bin_w = bins[1] - bins[0]
        y_kde_scaled = kde(x_vals) * len(data) * bin_w
        plt.plot(x_vals, y_kde_scaled, lw=2)
    plt.title(f"Distribución 'Total sales' (Hist + KDE) — {level}")
    plt.xlabel("Total sales"); plt.ylabel("Frecuencia")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / f"hist_kde_total_sales_{level.lower()}.png", dpi=150)
    plt.close()

def plot_boxplot(df: pd.DataFrame, level: str, out: Path):
    d = _ensure_total_sales(df)
    plt.figure(figsize=(5,5))
    plt.boxplot(d["Total sales"].dropna().astype(float).values, vert=True, patch_artist=True)
    plt.title(f"Boxplot 'Total sales' — {level}")
    plt.ylabel("Total sales")
    plt.tight_layout()
    plt.savefig(out / f"boxplot_total_sales_{level.lower()}.png", dpi=150)
    plt.close()

def plot_acf_pacf(df: pd.DataFrame, level: str, out: Path, lags: int = 40):
    d = _ensure_total_sales(_ensure_datetime_sorted(df))
    s = d["Total sales"].astype(float)
    fig, axs = plt.subplots(2, 1, figsize=(12,6))
    plot_acf(s, lags=lags, ax=axs[0])
    plot_pacf(s, lags=lags, ax=axs[1], method="ywm")
    axs[0].set_title(f"ACF — {level}")
    axs[1].set_title(f"PACF — {level}")
    plt.tight_layout()
    plt.savefig(out / f"acf_pacf_{level.lower()}.png", dpi=150)
    plt.close()


# ----------------------------
# Plots de negocio (ATC) si existen
# ----------------------------
def save_category_plots(df: pd.DataFrame, plots_dir: Path, artifacts_dir: Path, level: str = "Daily", top_n: int = 8):
    """
    - Barras por categoría
    - Pareto (barras + línea acumulada)
    - Stacked por fecha (top N categorías)
    - Ranking CSV (artifacts)
    """
    d = _ensure_datetime_sorted(_ensure_total_sales(df))
    atc_present = [c for c in ATC_COLS if c in d.columns]
    if not atc_present:
        print(f"[AVISO] {level}: no hay columnas ATC; omito plots de categorías.")
        return

    # Ranking total
    totals = d[atc_present].sum().sort_values(ascending=False)
    (artifacts_dir / "artifacts_marker.txt").parent.mkdir(parents=True, exist_ok=True)
    totals.to_frame("ventas").to_csv(artifacts_dir / ("ranking_categorias_daily.csv" if level.lower()=="daily" else f"ranking_categorias_{level.lower()}.csv"))
    
    # Barras
    plt.figure(figsize=(10,4))
    totals.plot(kind="bar")
    plt.title(f"Ventas por categoría — {level}")
    plt.ylabel("Ventas")
    plt.tight_layout()
    plt.savefig(plots_dir / "ventas_por_categoria_barras.png", dpi=150)
    plt.close()

    # Pareto
    cum = totals.cumsum() / totals.sum() * 100
    fig, ax1 = plt.subplots(figsize=(10,4))
    totals.plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Ventas")
    ax2 = ax1.twinx()
    ax2.plot(cum.index, cum.values, marker="o")
    ax2.set_ylabel("% acumulado"); ax2.set_ylim(0, 110)
    ax1.set_title(f"Pareto categorías — {level}")
    plt.tight_layout()
    plt.savefig(plots_dir / "ventas_por_categoria_pareto.png", dpi=150)
    plt.close()

    # Stacked por fecha (top N)
    top_cols = list(totals.head(top_n).index)
    df_stack = d[["datum"] + top_cols].dropna().copy()
    df_stack = df_stack.set_index("datum").astype(float)
    plt.figure(figsize=(12,5))
    plt.stackplot(df_stack.index, *[df_stack[c].values for c in top_cols], labels=top_cols)
    plt.legend(loc="upper left", ncol=min(4, len(top_cols)))
    plt.title(f"Stacked por fecha (Top {top_n}) — {level}")
    plt.ylabel("Ventas")
    plt.tight_layout()
    plt.savefig(plots_dir / "ventas_por_categoria_stacked.png", dpi=150)
    plt.close()


# ----------------------------
# Orquestador EDA por nivel
# ----------------------------
def generate_eda_plots(df: pd.DataFrame, level: str, plots_dir: Path, artifacts_dir: Path | None = None):
    """
    Genera todos los plots EDA estándar para un nivel (Daily/Weekly/Monthly/Hourly).
    """
    _ensure_dirs(plots_dir, artifacts_dir)
    # Básicos
    plot_ts_line(df, level, plots_dir)
    plot_smoothed(df, level, plots_dir)
    plot_hist_kde(df, level, plots_dir)
    plot_boxplot(df, level, plots_dir)
    plot_acf_pacf(df, level, plots_dir)
    # Negocio (si procede)
    if artifacts_dir and _has_atc(df) and level.lower() == "daily":
        save_category_plots(df, plots_dir, artifacts_dir, level="Daily", top_n=8)