# bakehouse-mlops

Caso resuelto del curso **Experiencias en Analítica**: pronóstico de ventas diarias por franquicia
con el dataset `samples.bakehouse` de Databricks Free Edition, desplegado con CI/CD desde GitHub.

```
bakehouse-mlops/
├── databricks.yml            # Bundle: variables y ambientes dev / prod
├── resources/jobs.yml        # Jobs: entrenamiento y predicción diaria
├── src/
│   ├── 01_features.py        # Tabla Delta de features (franquicia × día)
│   ├── 02_train.py           # 3 runs en MLflow, registro en UC y puerta de calidad
│   ├── 03_predict.py         # Predicción con @champion y tabla de monitoreo
│   └── utils_ventas.py       # Funciones puras (se prueban en CI)
├── tests/test_features.py    # Pruebas con pytest
└── .github/workflows/
    ├── ci.yml                # Pull request: pruebas + validate + deploy a dev
    └── cd.yml                # Merge a main: deploy a prod + entrenamiento
```

## Qué se crea en Databricks

| Objeto | dev | prod |
|---|---|---|
| Esquema | `workspace.bakehouse_dev` | `workspace.bakehouse_prod` |
| Tabla de features | `ventas_diarias_features` | igual |
| Modelo en Unity Catalog | `ventas_diarias` con `@champion` / `@challenger` | igual |
| Tablas de salida | `predicciones_ventas`, `metricas_monitoreo` | igual |
| Experimento MLflow | `/Users/<tu_usuario>/bakehouse-ventas-bakehouse_dev` | `…-bakehouse_prod` |

Free Edition tiene **un solo workspace**, así que dev y prod son destinos del mismo bundle que escriben en
esquemas distintos. En dev los jobs llevan el prefijo `[dev tu_usuario]` y su programación queda en pausa.

## Puesta en marcha

### 1. Token de Databricks
En tu workspace: **Settings › Developer › Access tokens › Generate new token**. Cópialo una sola vez.

### 2. Repo en GitHub
1. Crea el repo a partir de esta plantilla (o sube estos archivos) y comparte el acceso con la docente.
2. En **Settings › Secrets and variables › Actions › New repository secret** crea:
   - `DATABRICKS_HOST`: la URL del workspace, por ejemplo `https://dbc-xxxxxxxx-xxxx.cloud.databricks.com`
   - `DATABRICKS_TOKEN`: el token del paso 1

> El token **nunca** va en el código ni en un commit. Si se filtra, revócalo y crea otro.

### 3. Git folder en Databricks (opcional, para editar desde el workspace)
**Workspace › Create › Git folder**, pega la URL del repo. Para hacer push desde Databricks
necesitas vincular GitHub en **Settings › Linked accounts**.

### 4. Primer despliegue desde tu computador (opcional)
```bash
# Instala la CLI: https://docs.databricks.com/dev-tools/cli/install.html
databricks auth login --host https://dbc-xxxxxxxx-xxxx.cloud.databricks.com
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run entrenamiento -t dev
databricks bundle run prediccion_diaria -t dev
```

## Flujo de trabajo

1. Crea una rama, haz tu cambio y abre un **pull request** hacia `main`.
2. **CI** corre las pruebas, valida el bundle y lo despliega a dev. Ahí puedes correr los jobs para probar.
3. Al hacer **merge**, **CD** despliega a prod y corre `entrenamiento`.
4. La puerta de calidad (en `02_train.py`) compara el modelo nuevo con el `@champion` actual sobre la misma
   validación. Si mejora el MAE, se lleva `@champion`; si no, queda como `@challenger`.
5. `prediccion_diaria` corre todos los días a las 6:00 a. m. (hora de Bogotá) en prod.

## Notas para Free Edition

- **Cuota diaria:** si se agota, el cómputo se apaga hasta el día siguiente. Cada merge corre un
  entrenamiento: no hagas merge de cambios mínimos uno por uno.
- Si no quieres que la predicción diaria consuma cuota, cambia `estado_programacion` a `PAUSED` en el target `prod`.
- Revisa el rango de fechas en `01_features.py`. Si hay pocos días, cambia el grano a franquicia × hora.

## Problemas comunes

| Síntoma | Causa probable |
|---|---|
| `Error: cannot resolve bundle auth` en Actions | Faltan los secretos o el host tiene una barra `/` al final |
| `RESOURCE_DOES_NOT_EXIST ... @champion` en `03_predict` | Aún no corre `entrenamiento` en ese ambiente |
| `ModuleNotFoundError: utils_ventas` | El notebook se movió fuera de `src/`; el módulo debe estar en la misma carpeta |
| El modelo nunca pasa a `@champion` | Con los mismos datos el MAE empata; la puerta exige mejorar |

## Bonus del reto
- Análisis de drift: comparar la distribución de las features recientes contra el entrenamiento.
- Endpoint de Model Serving (CPU) apuntando a `@champion`.
- Exigir una mejora mínima en la puerta de calidad (`mejora_minima` en `decidir_promocion`).
