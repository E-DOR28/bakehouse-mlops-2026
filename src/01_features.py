# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Tabla de features
# MAGIC
# MAGIC Construye una fila por **franquicia y día** a partir de `samples.bakehouse`.
# MAGIC
# MAGIC | Columna | Rol |
# MAGIC |---|---|
# MAGIC | `ventas_totales` | Objetivo: suma de `totalPrice` del día |
# MAGIC | `dia_semana`, `es_fin_de_semana` | Patrón semanal |
# MAGIC | `ventas_dia_anterior`, `promedio_3_dias` | Rezagos: solo usan días **anteriores** (sin fuga de información) |
# MAGIC | `tamano_tienda`, `pais` | Contexto de la franquicia |
# MAGIC
# MAGIC Los parámetros llegan desde el job del bundle; si lo corres a mano, usa los widgets.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bakehouse_dev")
#testing
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
tabla_features = f"{catalog}.{schema}.ventas_diarias_features"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
print(f"Destino: {tabla_features}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explorar el rango de fechas
# MAGIC Si el rango es corto, considera cambiar el grano a franquicia y **hora**.

# COMMAND ----------

display(
    spark.sql(
        """
        SELECT MIN(dateTime) AS primera, MAX(dateTime) AS ultima,
               COUNT(DISTINCT DATE(dateTime)) AS dias, COUNT(DISTINCT franchiseID) AS franquicias
        FROM samples.bakehouse.sales_transactions
        """
    )
)

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

transacciones = spark.table("samples.bakehouse.sales_transactions")
franquicias = spark.table("samples.bakehouse.sales_franchises").select(
    "franchiseID",
    F.col("size").alias("tamano_tienda"),
    F.col("country").alias("pais"),
)

ventas_diarias = (
    transacciones.withColumn("fecha", F.to_date("dateTime"))
    .groupBy("franchiseID", "fecha")
    .agg(F.sum("totalPrice").cast("double").alias("ventas_totales"))
)

# Ventana por franquicia ordenada por fecha. Nota: los días sin ventas no
# aparecen, así que "día anterior" es el último día con ventas registradas.
ventana = Window.partitionBy("franchiseID").orderBy("fecha")

features = (
    ventas_diarias.withColumn("dia_semana", F.dayofweek("fecha"))
    .withColumn("es_fin_de_semana", F.col("dia_semana").isin(1, 7).cast("int"))
    .withColumn("ventas_dia_anterior", F.lag("ventas_totales", 1).over(ventana))
    .withColumn("promedio_3_dias", F.avg("ventas_totales").over(ventana.rowsBetween(-3, -1)))
    .join(franquicias, "franchiseID", "left")
    .dropna(subset=["ventas_dia_anterior", "promedio_3_dias"])
)

(
    features.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(tabla_features)
)

print(f"{features.count()} filas escritas en {tabla_features}")
display(spark.table(tabla_features).orderBy("franchiseID", "fecha").limit(20))