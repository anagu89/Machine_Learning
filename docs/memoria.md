# 💊 Pharma Sales Forecasting & Analytics — Memoria Técnica  

## 📌 Introducción y objetivos  
El objetivo del proyecto es **predecir las ventas de fármacos** en distintas granularidades temporales (horaria, diaria, semanal y mensual) utilizando técnicas de Machine Learning y análisis de series temporales.  
Se busca no solo generar predicciones precisas, sino también:  
- **Identificar patrones de consumo** (estacionalidad, ciclos, variaciones semanales).  
- **Detectar categorías dominantes** de medicamentos.  
- **Aportar valor de negocio** mediante un dashboard interactivo en Streamlit que permita explorar los resultados.  

---

Además del enfoque técnico, este proyecto tiene un **impacto directo en negocio**:

- **Farmacias y distribuidores locales**: optimizan inventario antes de picos de demanda (ej. temporada de gripe), evitan roturas de stock de medicamentos críticos y mejoran la eficiencia en pedidos semanales.  
- **Laboratorios farmacéuticos**: planifican la producción con meses de antelación, detectan qué categorías ATC ganan relevancia y ajustan campañas de marketing según la estacionalidad.  
- **Gestores estratégicos**: identifican tendencias de largo plazo (ej. aumento en ansiolíticos post-pandemia), deciden en qué categorías invertir más recursos y cuantifican el impacto económico de reducir el error de predicción (ej. *MAPE < 2% implica ahorro en sobrestock*).  

---

## 📊 Fuente de datos (Kaggle)  
El dataset utilizado proviene de:  
**[Kaggle: Pharma Sales Data](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data)**  

- Registros de ventas desagregadas por código **ATC**.  
- Cuatro granularidades: **Hourly, Daily, Weekly, Monthly**.  
- Columnas principales:  
  - `datum`: fecha de la observación.  
  - `M01AB`, `M01AE`, `N02BA`, `N02BE`, `N05B`, `N05C`, `R03`, `R06`: ventas por categoría.  
  - `Total sales`: suma de todas las categorías (calculada en el procesamiento).  

---

## 🛠️ Procesamiento de datos  
Se implementó un **pipeline reproducible en Python** (`src/data_processing.py`) que:  
1. **Convierte las fechas** a formato datetime y ordena las series.  
2. **Calcula `Total sales`** como suma de todas las columnas ATC.  
3. **Añade variables temporales**:  
   - Año, mes, día, día de la semana, hora (para granularidad horaria).  
4. **Genera features adicionales para Daily**:  
   - Lags (`lag_1`, `lag_7`)  
   - Medias móviles (`roll7`)  
5. **Exporta datasets procesados** en `data/processed/` y artefactos de negocio en `docs/artifacts/`.  

---

## 📊 Justificación de la granularidad — ¿por qué entrenamos con *Daily*?

Elegimos **Daily** como granularidad principal por un balance entre:
- Volumen suficiente de datos   
- Comparabilidad de métricas   
- Escalabilidad hacia otras frecuencias   

---

### ⏳ ¿Por qué no usamos `hourly/weekly/monthly`?

**1️⃣ Volumen y granularidad de los datos**

- **Daily (~2100 filas, ~6 años)** → suficiente para entrenar modelos supervisados robustos (RF, GBR, XGB, CatBoost).  
- **Hourly (~50k filas)** → demasiado grande y ruidoso, dominado por patrones intradiarios (picos mañana/tarde). Más adecuado para modelos de series de alta frecuencia (SARIMAX intradiario, Prophet con estacionalidad horaria) que para regresión clásica.  
- **Weekly (302 filas)** y **Monthly (70 filas)** → demasiado cortos para Random Forest/XGBoost → alto riesgo de *overfitting*.  

**2️⃣ Comparabilidad**

- Los modelos supervisados necesitan la misma granularidad para que las métricas (MAPE, MAE, etc.) sean comparables.  
- Si mezclamos Daily con Weekly/Monthly, las métricas no serían equivalentes, ya que miden horizontes distintos.  
- Por eso fijamos **Daily** como dataset de referencia para entrenar y comparar modelos.  

---

### 📈 Escalabilidad de entrenar sólo con *Daily*

**1️⃣ Escalabilidad en negocio / forecast**

Un modelo entrenado en Daily puede **agregarse hacia arriba**:  
- Para Weekly → sumar predicciones en bloques de 7 días.  
- Para Monthly → agregación de los días de cada mes.  
- Esto permite escalar Daily hacia granularidades superiores sin pérdida de comparabilidad.  

El caso contrario (**Hourly → Daily**) es más costoso porque habría que suavizar mucho ruido y manejar grandes volúmenes de datos.  

**2️⃣ Limitaciones**

- **Horizonte largo de predicción**: a 6–12 meses, el error acumulado en Daily puede crecer → en ese caso modelos mensuales podrían ser más estables.  
- **Big data futuro**: con datos de muchos países/años, habría que pensar en pipelines distribuidos.  
- **No capta patrones intradiarios**: entrenar con Daily ignora los picos horarios (mañana/tarde), aunque eso no afecta al escalado hacia semanas/meses.  

---

## 🤖 Modelos probados  
Se implementaron y compararon distintos algoritmos de predicción:  

- **ARIMA / SARIMAX** → Modelos clásicos de series temporales.  
- **Random Forest Regressor** → Ensamble de árboles de decisión.  
- **Gradient Boosting Regressor (GBR)** → Boosting secuencial de árboles.  
- **XGBoost Regressor** → Algoritmo optimizado de boosting.  
- **CatBoost Regressor** → Modelo ganador 🏆 por su bajo error y estabilidad.  
- **KMeans Clustering** → Segmentación no supervisada de días de ventas.  
- *(Reservado para futuro)* **Prophet** para estacionalidad compleja.  

---

## 📏 Evaluación con métricas  
Se evaluaron todos los modelos usando **test temporal (último 20%)** y métricas estándar:  

- **MAPE** (Error porcentual absoluto medio).  
- **MAE** (Error absoluto medio).  
- **MSE** (Error cuadrático medio).  
- **R²** (Coeficiente de determinación).  

📌 **Resultados clave**:  
- **CatBoost (base)** alcanzó un **MAPE < 2%** y R² ≈ 0.99.  
- Otros modelos como XGBoost y GBR tuvieron desempeños cercanos, pero con ligeras desventajas.  
- Modelos clásicos (ARIMA/SARIMAX) quedaron por debajo en precisión.  

---

## 🧩 Clustering y patrones detectados  
Mediante **KMeans** se agruparon días con patrones similares de ventas:  
- Se identificaron segmentos diferenciados de consumo (alta demanda, media, baja).  
- Permite detectar estacionalidad y posibles cambios de comportamiento.  

Patrones adicionales:  
- **Estacionalidad intra–anual**: mayor consumo en determinados meses.  
- **Patrón semanal**: picos recurrentes en días laborales frente a fines de semana.  
- **Patrón horario**: horas de mayor demanda concentradas en franjas específicas.  

---

## 📊 Visualizaciones clave  
- Series temporales de ventas.  
- Distribuciones (histogramas, KDE, boxplots).  
- Pareto y ranking de categorías ATC.  
- Stacked de categorías en el tiempo.  
- ACF / PACF para autocorrelación y dependencia.  
- Resultados de clustering (PCA scatter, heatmap de medias).  
- Dashboard interactivo en **Streamlit** con:  
  - Filtros de granularidad y fechas.  
  - Visualización de métricas y predicciones.  
  - Galería de plots.  

---

## 🧠 Explicabilidad y ética  
- **Importancia de variables**: se analizaron feature importances en RF, GBR, XGB y CatBoost.  
- **Sesgos potenciales**:  
  - Dependencia de datos históricos → riesgo si cambian patrones externos.  
  - Diferencias en categorías con menor representación.  
- **Mitigaciones**:  
  - Uso de varios modelos en paralelo.  
  - Evaluación continua y actualización del pipeline.  
- **Consideraciones éticas**:  
  - La predicción de ventas no debe usarse para restringir disponibilidad de fármacos esenciales.  
  - Transparencia en el uso del modelo mediante **memoria técnica y dashboard**.  

---

## ✅ Conclusiones y futuro  
- Se logró un **sistema de forecasting robusto** con precisión alta (MAPE < 2%).  
- El **CatBoost Regressor** fue el mejor modelo global.  
- El análisis de categorías y clustering aporta **valor de negocio** adicional.  

📌 **Líneas futuras**:  
- Integrar **Prophet** para capturar estacionalidad avanzada.  
- Ampliar despliegue con API y dashboard web escalable.  
- Incorporar explicabilidad más profunda (SHAP values).  
- Monitorizar el modelo en producción con datos en tiempo real.  