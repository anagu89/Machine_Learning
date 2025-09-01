# 📚 Data Dictionary — Pharma Sales Forecasting & Analytics

Este documento describe el significado, formato y procedencia de las variables utilizadas en el proyecto, así como el **data lineage** (origen → transformación → destino) y las **features** (FE) generadas para modelado.

---

## 1️⃣ Vista general de datasets

| Nivel | Fichero (processed) | Grano temporal | Filas aprox. | Uso principal |
|---|---|---:|---:|---|
| Hourly | `data/processed/saleshourly_clean.csv` | 1 registro por **hora** | ~50k | Patrones intradiarios, gráficos |
| Daily  | `data/processed/salesdaily_clean.csv`  | 1 registro por **día**  | ~2.1k | **Entrenamiento supervisado** y evaluación |
| Weekly | `data/processed/salesweekly_clean.csv` | 1 registro por **semana** | ~300 | EDA y benchmarking |
| Monthly| `data/processed/salesmonthly_clean.csv`| 1 registro por **mes**   | ~70  | EDA y benchmarking |

> Los conjuntos **train/test** se guardan en `data/train/` y `data/test/` con el mismo nivel de granularidad y las **mismas columnas** de features usadas en el entrenamiento.

---

## 2️⃣ Columnas comunes

| Columna | Tipo | Descripción |
|---|---|---|
| `datum` | fecha/tiempo (`YYYY-MM-DD` o `YYYY-MM-DD HH:MM:SS`) | Marca temporal del registro. Se normaliza con `pd.to_datetime(..., errors="coerce")`. |
| `Total sales` | numérico (float) | Suma de ventas por categorías ATC disponibles en cada fila (ver sección ATC). Unidad: **unidades** (si el origen está en unidades) o **importe** (si el origen está monetizado). |
| `Year` | entero | Año extraído de `datum`. |
| `Month` | entero (1–12) | Mes extraído de `datum`. |
| `Day` | entero (1–31) | Día del mes (se añade en Hourly y, si procede, Daily). |
| `Weekday` | entero (1–7) | Día de la semana: **1 = Lunes … 7 = Domingo**. |
| `Week` | entero (1–53) | Semana ISO (sólo en Weekly). |
| `Hour` | entero (0–23) | Hora del día (sólo en Hourly). |

> Algunas columnas temporales se añaden **sólo si aplican** al nivel (p. ej., `Hour` en Hourly). La función base es `add_time_parts()` (ver `src/utils.py` y `src/data_processing.py`).

---

## 3️⃣ Columnas ATC (categorías)

Las columnas ATC representan ventas por subfamilias terapéuticas. Se convierten a numérico con coerción silenciosa y pueden faltar en algunos niveles.

| Columna | Descripción (abreviada) | Ejemplos |
|---|---|---|
| `M01AB` | Antiinflamatorios/antirreumáticos no esteroideos derivados del ácidos acético | *diclofenaco, indometacina* |
| `M01AE` | Antiinflamatorios/antirreumáticos no esteroideos derivados del ácidos propiónicos | *ibuprofeno, naproxeno, ketoprofeno* |
| `N02BA` | Analgésicos del ácido salicílico y derivados | *aspirina* |
| `N02BE` | Analgésicos | *paracetamol* |
| `N05B`  | Ansiolíticos | *benzodiazepinas*| 
| `N05C`  | Hipnóticos y sedantes | *zolpidem* |
| `R03`   | Antiasmáticos/COPD | *salbutamol, budesonida, montelukast* |
| `R06`   | Antihistamínicos para uso sistémico | *loratadina, cetirizina* |

> **Nota:** En el pipeline, si alguna ATC no existe en el fichero, simplemente se ignora.

---

## 4️⃣ Cálculo de `Total sales`

`Total sales = sum(ATC_presentes_en_la_fila)`

- Implementado en `compute_total_sales()` (ver `src/utils.py`) y aplicado en `src/data_processing.py`.
- Si no existen columnas ATC presentes, `Total sales` se deja como `NaN` y se **avisa por consola**.
- Datos convertidos con `pd.to_numeric(..., errors="coerce")` para robustez.

---

## 5️⃣ Feature Engineering - resumen ejecutivo 

| Variable       | Tipo     | Descripción                                                                 |
|----------------|----------|-----------------------------------------------------------------------------|
| `Year`         | Entero   | Año de la observación (extraído de `datum`).                                |
| `Month`        | Entero   | Mes de la observación (1–12).                                               |
| `Weekday`      | Entero   | Día de la semana (1=Lunes … 7=Domingo).                                     |
| `lag_1`        | Numérico | Valor de `Total sales` con retraso de 1 día.                                |
| `lag_7`        | Numérico | Valor de `Total sales` con retraso de 7 días.                               |
| `roll7`        | Numérico | Media móvil de los últimos 7 días con retraso de 1 día.                     |
| `Total sales`  | Numérico | Suma de todas las ventas por categoría ATC en la fecha correspondiente.     |
| `ATC` (ej: M01AB, N02BA…) | Numérico | Columnas de ventas por categoría terapéutica según clasificación ATC. |

---

## 6️⃣ Feature Engineering - granulado 

### 6.1) Daily (nivel usado para **training**)
Generado en `utils.add_time_features_daily()` y `utils.make_supervised_daily()`:

| Feature | Tipo | Descripción |
|---|---|---|
| `Year`, `Month`, `Weekday` | entero | Partes temporales de calendario. |
| `lag_1` | float | `Total sales` desplazado 1 día. |
| `lag_7` | float | `Total sales` desplazado 7 días. |
| `roll7` | float | Media móvil de 7 días con **lag 1** (no usa el día actual). |
| `ATC` disponibles | float | Se añaden todas las ATC presentes como features adicionales. |

> Tras crear lags/medias móviles, se **dropean NaNs** (primeras filas) para asegurar integridad de `X`.

### 6.2) Weekly
En `training._make_supervised_generic(..., level="Weekly")`:

| Feature | Descripción |
|---|---|
| `Year`, `Week` | Partes temporales |
| `lag_1`, `lag_2` | Desplazamientos semanales |
| `roll4` | Media móvil de 4 semanas (lag 1) |

### 6.3) Monthly
En `training._make_supervised_generic(..., level="Monthly")`:

| Feature | Descripción |
|---|---|
| `Year`, `Month` | Partes temporales |
| `lag_1`, `lag_12` | Desplazamientos mensuales (1 y 12) |
| `roll3` | Media móvil de 3 meses (lag 1) |

### 6.4) Hourly
En `training._make_supervised_generic(..., level="Hourly")`:

| Feature | Descripción |
|---|---|
| `Year`, `Month`, `Day`, `Hour`, `Weekday` | Partes temporales |
| `lag_1`, `lag_24` | Desplazamientos horarios (1 y 24) |
| `roll24` | Media móvil de 24 horas (lag 1) |

---

## 7️⃣ Data lineage (origen → transformación → destino)

Representa cómo fluyen los datos desde la fuente original hasta los artefactos finales:

```plaintext
Kaggle Pharma Sales Data (raw .csv)
        │
        ▼
src/data_processing.py
  - Conversión de fechas (datum → datetime)
  - Cálculo de Total sales (suma ATC)
  - Creación de features temporales
  - Guardado en data/processed/*.csv
        │
        ▼
src/training.py
  - Split train/test (80/20)
  - Feature engineering (lag, roll, time parts)
  - Entrenamiento de modelos ML
  - Guardado modelo en models/final_model.pkl
  - Export metrics_summary.csv
        │
        ▼
src/evaluation.py
  - Carga del modelo final
  - Evaluación sobre test (MAPE, MAE, MSE, R²)
  - Export eval_metrics.csv, y_true_pred.csv
        │
        ▼
docs/artifacts/
  - KPIs por dataset
  - Ranking categorías, Pareto, HHI
  - Métricas de evaluación
        │
        ▼
docs/plots/
  - Gráficas ACF/PACF
  - Importancias de features
  - Plots de clustering
        │
        ▼
app_streamlit/app.py
  - Dashboard interactivo con todas las visualizaciones
```
---

## 8️⃣ Tipos, nulos y normalización

- **Tipos**
  - Fechas normalizadas con:  
    ```python
    pd.to_datetime(..., errors="coerce")
    ```
  - Numéricos forzados con:  
    ```python
    pd.to_numeric(..., errors="coerce")
    ```

- **Nulos**
  - Filas con `NaT` en `datum` → se eliminan antes de ordenar.  
  - Tras crear lags/rolling, las primeras filas con `NaN` se eliminan para entrenamiento.

- **Orden temporal**
  - Todos los datasets se ordenan por `datum` en orden ascendente antes de FE/splits.

- **Unidades**
  - `Total sales` y columnas ATC representan la **suma de ventas** de las categorías ATC originales.

---

## 9️⃣ Convenciones y validaciones

- **Día de la semana:** 1 = Lunes … 7 = Domingo.  
- **Semana ISO:** usado en Weekly (`dt.isocalendar().week`).  
- **Integridad mínima (Daily):**
  - Deben existir las columnas `datum`, `Total sales`.  
  - Debe quedar longitud suficiente tras FE para hacer split temporal (80/20 por defecto).

---

## 1️⃣0️⃣ Glosario

| Término     | Definición                                                                 |
|-------------|-----------------------------------------------------------------------------|
| **ATC**     | Anatomical Therapeutic Chemical classification (familias de fármacos).     |
| **FE**      | Feature Engineering: creación de variables a partir de señales temporales. |
| **MAPE/MAE/MSE/R²** | Métricas de evaluación de regresión.                                |
| **Silhouette** | Métrica de calidad de clustering.                                        |
| **Artifacts**  | Ficheros de salida para reporting (`CSV`, `MD`, `PNG`).                  |

---

## 1️⃣1️⃣ Mantenimiento

- **Fuente de datos:** [Kaggle — Pharma Sales Data](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data)  
- **Scripts clave:**  
  - `src/data_processing.py`  
  - `src/training.py`  
  - `src/evaluation.py`  
  - `src/plots.py`  
- **Dashboard:** `app_streamlit/app.py`