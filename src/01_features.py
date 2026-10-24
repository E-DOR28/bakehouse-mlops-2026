# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Feature table
# MAGIC
# MAGIC Construye una fila por **franquicia y día** a partir de `samples.bakehouse`.
# MAGIC
# MAGIC Es una **feature table** de Unity Catalog: una tabla Delta con llave primaria
# MAGIC `(franchiseID, fecha)`. La llave es lo que la hace aparecer en **Features**, con su linaje
# MAGIC hacia los modelos que la usan.
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

#Un comentario
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
    .dropna(subset=["franchiseID", "fecha", "ventas_dia_anterior", "promedio_3_dias"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guardar como feature table
# MAGIC - La primera vez se crea la tabla; las siguientes solo se reemplazan los **datos**.
# MAGIC   Por eso no usamos `overwriteSchema`: borraría la llave primaria en cada ejecución.
# MAGIC - Si la tabla no tiene llave primaria (por ejemplo, la creó una versión anterior de este
# MAGIC   notebook), se la agregamos. Las columnas de la llave deben ser `NOT NULL`.

# COMMAND ----------


def tiene_llave_primaria(tabla: str) -> bool:
    catalogo, esquema, nombre = tabla.split(".")
    return (
        spark.sql(
            f"""
            SELECT 1 FROM {catalogo}.information_schema.table_constraints
            WHERE table_schema = '{esquema}' AND table_name = '{nombre}'
              AND constraint_type = 'PRIMARY KEY'
            """
        ).count()
        > 0
    )


features.write.mode("overwrite").saveAsTable(tabla_features)

if not tiene_llave_primaria(tabla_features):
    spark.sql(f"ALTER TABLE {tabla_features} ALTER COLUMN franchiseID SET NOT NULL")
    spark.sql(f"ALTER TABLE {tabla_features} ALTER COLUMN fecha SET NOT NULL")
    spark.sql(
        f"ALTER TABLE {tabla_features} "
        "ADD CONSTRAINT ventas_diarias_features_pk PRIMARY KEY (franchiseID, fecha)"
    )
    print("Llave primaria (franchiseID, fecha) agregada")

spark.sql(
    f"COMMENT ON TABLE {tabla_features} IS "
    "'Features de ventas diarias por franquicia (caso Bakehouse). Una fila por franquicia y día.'"
)

print(f"{features.count()} filas escritas en la feature table {tabla_features}")
display(spark.table(tabla_features).orderBy("franchiseID", "fecha").limit(20))