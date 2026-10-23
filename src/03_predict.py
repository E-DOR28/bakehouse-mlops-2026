# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Predicción con @champion y monitoreo
# MAGIC
# MAGIC - Carga el modelo por **alias**, no por versión: promover otro modelo no cambia este código.
# MAGIC - Escribe `predicciones_ventas` guardando la versión del modelo que generó cada predicción.
# MAGIC - Escribe `metricas_monitoreo`: MAE por día y una alerta si supera el umbral.
# MAGIC
# MAGIC Como `samples.bakehouse` es histórico, "predecimos" los últimos días y los comparamos
# MAGIC con las ventas reales. En producción las ventas reales llegarían después.

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.1" "scikit-learn>=1.3"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bakehouse_dev")
dbutils.widgets.text("model_name", "ventas_diarias")
dbutils.widgets.text("dias_a_puntuar", "3")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
dias_a_puntuar = int(dbutils.widgets.get("dias_a_puntuar"))

nombre_modelo = f"{catalog}.{schema}.{model_name}"
tabla_features = f"{catalog}.{schema}.ventas_diarias_features"
tabla_predicciones = f"{catalog}.{schema}.predicciones_ventas"
tabla_metricas = f"{catalog}.{schema}.metricas_monitoreo"

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as F

from utils_ventas import FEATURES, OBJETIVO, metricas_regresion, umbral_alerta

mlflow.set_registry_uri("databricks-uc")
version = MlflowClient().get_model_version_by_alias(nombre_modelo, "champion").version
modelo = mlflow.pyfunc.load_model(f"models:/{nombre_modelo}@champion")
print(f"Usando {nombre_modelo} versión {version} (@champion)")

# COMMAND ----------

datos = spark.table(tabla_features).toPandas()
fechas_recientes = sorted(datos["fecha"].unique())[-dias_a_puntuar:]
recientes = datos[datos["fecha"].isin(fechas_recientes)].copy()
recientes["prediccion"] = modelo.predict(recientes[FEATURES])

salida = recientes[["franchiseID", "fecha", OBJETIVO, "prediccion"]].rename(
    columns={OBJETIVO: "ventas_reales"}
)
salida["version_modelo"] = str(version)

(
    spark.createDataFrame(salida)
    .withColumn("fecha_ejecucion", F.current_timestamp())
    .write.mode("append")
    .saveAsTable(tabla_predicciones)
)
print(f"{len(salida)} predicciones agregadas a {tabla_predicciones}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoreo
# MAGIC Umbral de alerta = 1,5 × la mediana del MAE diario observado (incluye el histórico de la tabla).

# COMMAND ----------

import pandas as pd

filas = []
for fecha, grupo in salida.groupby("fecha"):
    filas.append({"fecha": fecha, "mae": metricas_regresion(grupo["ventas_reales"], grupo["prediccion"])["mae"]})
metricas = pd.DataFrame(filas)

historico = []
if spark.catalog.tableExists(tabla_metricas):
    historico = [r["mae"] for r in spark.table(tabla_metricas).select("mae").collect()]

umbral = umbral_alerta(historico + metricas["mae"].tolist())
metricas["umbral"] = umbral
metricas["alerta"] = metricas["mae"] > umbral
metricas["version_modelo"] = str(version)

(
    spark.createDataFrame(metricas)
    .withColumn("fecha_ejecucion", F.current_timestamp())
    .write.mode("append")
    .saveAsTable(tabla_metricas)
)

display(metricas)
if metricas["alerta"].any():
    print("ALERTA: el error superó el umbral. Toca reentrenar y comparar contra el champion.")
