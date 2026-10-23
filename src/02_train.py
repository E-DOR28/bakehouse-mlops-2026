# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Experimentos, registro y puerta de calidad
# MAGIC
# MAGIC 1. Partición **temporal**: entrena con los primeros días y valida con los últimos.
# MAGIC 2. Tres runs en MLflow con las mismas features y la misma métrica (MAE):
# MAGIC    línea base (promedio de 3 días), regresión lineal y random forest.
# MAGIC 3. El mejor modelo se registra en Unity Catalog.
# MAGIC 4. **Puerta de calidad:** recibe el alias `@champion` solo si mejora el MAE del champion actual;
# MAGIC    si no, queda como `@challenger`.

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.1" "scikit-learn>=1.3"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bakehouse_dev")
dbutils.widgets.text("model_name", "ventas_diarias")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

tabla_features = f"{catalog}.{schema}.ventas_diarias_features"
nombre_modelo = f"{catalog}.{schema}.{model_name}"

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils_ventas import (
    FEATURES,
    FEATURES_CATEGORICAS,
    OBJETIVO,
    decidir_promocion,
    metricas_regresion,
    particion_temporal,
)

mlflow.set_registry_uri("databricks-uc")
usuario = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{usuario}/bakehouse-ventas-{schema}")

# COMMAND ----------

datos = spark.table(tabla_features).toPandas()
datos["fecha"] = pd.to_datetime(datos["fecha"])

entrenamiento, validacion = particion_temporal(datos, "fecha", frac_validacion=0.2)
X_train, y_train = entrenamiento[FEATURES], entrenamiento[OBJETIVO]
X_val, y_val = validacion[FEATURES], validacion[OBJETIVO]

print(f"Entrenamiento: {len(entrenamiento)} filas, hasta {entrenamiento['fecha'].max():%Y-%m-%d}")
print(f"Validación:    {len(validacion)} filas, desde {validacion['fecha'].min():%Y-%m-%d}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run 1 · Línea base
# MAGIC Predecir con el promedio de los 3 días anteriores. Si un modelo no le gana a esto, no vale la pena.

# COMMAND ----------

resultados = {}

with mlflow.start_run(run_name="baseline_promedio_3_dias"):
    m = metricas_regresion(y_val, X_val["promedio_3_dias"])
    mlflow.log_param("modelo", "promedio_3_dias")
    mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
    resultados["baseline"] = {"mae": m["mae"], "model_uri": None}

print(resultados["baseline"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Runs 2 y 3 · Modelos
# MAGIC El preprocesamiento va dentro del pipeline, así el modelo registrado recibe las columnas tal como están en la tabla.

# COMMAND ----------

def crear_pipeline(estimador):
    preprocesamiento = ColumnTransformer(
        [("categoricas", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAS)],
        remainder="passthrough",
    )
    return Pipeline([("preprocesamiento", preprocesamiento), ("modelo", estimador)])


candidatos = {
    "regresion_lineal": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
}

for nombre, estimador in candidatos.items():
    with mlflow.start_run(run_name=nombre) as run:
        pipeline = crear_pipeline(estimador).fit(X_train, y_train)
        m = metricas_regresion(y_val, pipeline.predict(X_val))

        parametros = {
            k: v for k, v in estimador.get_params().items() if isinstance(v, (int, float, str, bool))
        }
        mlflow.log_param("modelo", nombre)
        mlflow.log_params(parametros)
        mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
        info = mlflow.sklearn.log_model(pipeline, name="modelo", input_example=X_train.head(5))

        resultados[nombre] = {"mae": m["mae"], "model_uri": info.model_uri, "run_id": run.info.run_id}

display(pd.DataFrame(resultados).T[["mae"]].sort_values("mae"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registro y puerta de calidad

# COMMAND ----------

modelos = {k: v for k, v in resultados.items() if v["model_uri"]}
mejor = min(modelos, key=lambda k: modelos[k]["mae"])
mae_mejor = modelos[mejor]["mae"]

if mae_mejor >= resultados["baseline"]["mae"]:
    print("Atención: el mejor modelo no le gana a la línea base. Revisa las features.")

version = mlflow.register_model(modelos[mejor]["model_uri"], nombre_modelo)
cliente = MlflowClient()
cliente.set_model_version_tag(nombre_modelo, version.version, "val_mae", f"{mae_mejor:.4f}")
cliente.set_model_version_tag(nombre_modelo, version.version, "algoritmo", mejor)

# Champion actual evaluado sobre la MISMA validación, para comparar en igualdad de condiciones.
try:
    cliente.get_model_version_by_alias(nombre_modelo, "champion")
    champion = mlflow.pyfunc.load_model(f"models:/{nombre_modelo}@champion")
    mae_champion = metricas_regresion(y_val, champion.predict(X_val))["mae"]
except MlflowException:
    mae_champion = None

promover = decidir_promocion(mae_mejor, mae_champion)
alias = "champion" if promover else "challenger"
cliente.set_registered_model_alias(nombre_modelo, alias, version.version)

print(f"Mejor run: {mejor} · MAE validación {mae_mejor:,.2f}")
print(f"Champion anterior: {'ninguno' if mae_champion is None else f'MAE {mae_champion:,.2f}'}")
print(f"Versión {version.version} de {nombre_modelo} → @{alias}")
