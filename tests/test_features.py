"""Pruebas de las funciones de src/utils_ventas.py. Corren en el CI sin Databricks."""
import pandas as pd
import pytest

from utils_ventas import (
    FEATURES,
    decidir_promocion,
    es_fin_de_semana,
    metricas_regresion,
    particion_temporal,
    umbral_alerta,
)


@pytest.mark.parametrize("dia, esperado", [(1, 1), (7, 1), (2, 0), (4, 0), (6, 0)])
def test_es_fin_de_semana_usa_convencion_de_spark(dia, esperado):
    assert es_fin_de_semana(dia) == esperado


@pytest.mark.parametrize("dia", [0, 8])
def test_es_fin_de_semana_rechaza_dias_invalidos(dia):
    with pytest.raises(ValueError):
        es_fin_de_semana(dia)


def _datos(n_fechas=10, franquicias=3):
    fechas = pd.date_range("2024-05-01", periods=n_fechas, freq="D")
    return pd.DataFrame(
        [{"franchiseID": f, "fecha": d, "ventas_totales": 100.0} for f in range(franquicias) for d in fechas]
    )


def test_particion_temporal_no_mezcla_fechas():
    entrenamiento, validacion = particion_temporal(_datos(), "fecha", 0.2)
    assert entrenamiento["fecha"].max() < validacion["fecha"].min()


def test_particion_temporal_respeta_la_fraccion():
    _, validacion = particion_temporal(_datos(n_fechas=10), "fecha", 0.2)
    assert validacion["fecha"].nunique() == 2


def test_particion_temporal_siempre_deja_datos_de_entrenamiento():
    entrenamiento, _ = particion_temporal(_datos(n_fechas=2), "fecha", 0.9)
    assert len(entrenamiento) > 0


def test_particion_temporal_necesita_dos_fechas():
    with pytest.raises(ValueError):
        particion_temporal(_datos(n_fechas=1), "fecha", 0.2)


def test_metricas_regresion():
    m = metricas_regresion([10, 20, 30], [12, 18, 30])
    assert m["mae"] == pytest.approx(4 / 3)
    assert m["rmse"] == pytest.approx((8 / 3) ** 0.5)


def test_metricas_regresion_tamanos_distintos():
    with pytest.raises(ValueError):
        metricas_regresion([1, 2], [1])


def test_primer_modelo_siempre_se_promueve():
    assert decidir_promocion(50.0, None) is True


def test_promueve_solo_si_mejora():
    assert decidir_promocion(40.0, 50.0) is True
    assert decidir_promocion(50.0, 50.0) is False
    assert decidir_promocion(60.0, 50.0) is False


def test_mejora_minima():
    assert decidir_promocion(48.0, 50.0, mejora_minima=5.0) is False


def test_umbral_alerta():
    assert umbral_alerta([10, 20, 30]) == pytest.approx(30.0)
    assert umbral_alerta([]) is None


def test_lista_de_features_sin_el_objetivo():
    assert "ventas_totales" not in FEATURES
