"""Funciones puras del caso Bakehouse.

No dependen de Spark ni de Databricks, así que se pueden probar con pytest
en GitHub Actions. Los notebooks de esta misma carpeta las importan.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd

OBJETIVO = "ventas_totales"
FEATURES_NUMERICAS = ["dia_semana", "es_fin_de_semana", "ventas_dia_anterior", "promedio_3_dias"]
FEATURES_CATEGORICAS = ["tamano_tienda", "pais"]
FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS


def es_fin_de_semana(dia_semana: int) -> int:
    """Devuelve 1 si es sábado o domingo.

    Usa la convención de Spark `dayofweek`: 1 = domingo, 7 = sábado.
    """
    if not 1 <= int(dia_semana) <= 7:
        raise ValueError(f"dia_semana debe estar entre 1 y 7, llegó {dia_semana}")
    return int(int(dia_semana) in (1, 7))


def particion_temporal(
    df: pd.DataFrame, col_fecha: str = "fecha", frac_validacion: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa entrenamiento y validación por fecha, nunca al azar.

    Las fechas más recientes (frac_validacion del total de fechas distintas)
    van a validación, así el modelo se evalúa sobre "el futuro".
    """
    if not 0 < frac_validacion < 1:
        raise ValueError("frac_validacion debe estar entre 0 y 1")
    fechas = sorted(df[col_fecha].unique())
    if len(fechas) < 2:
        raise ValueError("Se necesitan al menos dos fechas distintas para una partición temporal")
    n_val = min(len(fechas) - 1, max(1, math.ceil(len(fechas) * frac_validacion)))
    corte = fechas[-n_val]
    entrenamiento = df[df[col_fecha] < corte].copy()
    validacion = df[df[col_fecha] >= corte].copy()
    return entrenamiento, validacion


def metricas_regresion(y_real: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """MAE y RMSE."""
    real = np.asarray(list(y_real), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if real.shape != pred.shape:
        raise ValueError("y_real y y_pred deben tener el mismo tamaño")
    if real.size == 0:
        raise ValueError("No hay datos para calcular métricas")
    error = pred - real
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2)))}


def decidir_promocion(
    mae_nuevo: float, mae_champion: Optional[float], mejora_minima: float = 0.0
) -> bool:
    """Puerta de calidad: el modelo nuevo es champion solo si mejora el MAE.

    Si todavía no hay champion, el primer modelo se promueve.
    """
    if mae_champion is None:
        return True
    return mae_nuevo < mae_champion - mejora_minima


def umbral_alerta(maes: Iterable[float], factor: float = 1.5) -> Optional[float]:
    """Umbral de alerta de monitoreo: factor × mediana del MAE observado."""
    serie = pd.Series(list(maes), dtype=float).dropna()
    if serie.empty:
        return None
    return float(serie.median() * factor)
