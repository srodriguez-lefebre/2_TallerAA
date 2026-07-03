# Estado de entrega - Proyecto 2

Este archivo resume la auditoria del estado actual de la entrega.

La politica vigente es: presentar un sistema simple, defendible y copiable en 8
carpetas. El laboratorio historico queda como soporte, no como narrativa
principal.

## Decision final actual

La submission oficial de presentacion es:

```text
100. Entregable/submission.csv
```

Tambien se conserva una copia validada en:

```text
07_final/submission.csv
```

Origen:

```text
investigation/results/submissions/parallel100_20260702/e100_headsep25_globalmel375_f1024_375.csv
```

Formula:

```text
0.25  * separable_headsep_e100_seed42
+ 0.375 * globalmel_sep_temporal_e100_seed42
+ 0.375 * sep_temporal_f1024_e100_seed42
```

Evidencia:

- Kaggle private LB: `0.67126`;
- SHA-256: `4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab`;
- formato esperado: `3361` filas y `81` columnas;
- competencia: `freesound-audio-tagging-2019`.

## Estructura de entrega

| Carpeta | Rol |
|---|---|
| `01_analisis_datos/` | entender problema, clases, splits, duraciones y metrica |
| `02_preprocesamiento/` | explicar audio a log-mel, normalizaciones y dimensiones |
| `03_entrenamiento/` | explicar entrenamiento de las tres ramas finales |
| `04_experimentacion_camino/` | resumir como se llego a modelos fuertes |
| `05_experimentacion_general/` | separar pruebas laterales y descartes |
| `06_experimentacion_final/` | comparar variantes alrededor del modelo final |
| `07_final/` | fijar decision final y validar CSV |
| `100. Entregable/` | pipeline preparado para entorno seguro |

## Referencias experimentales

El sistema anterior de tres ramas con pesos iguales tenia private LB `0.66649`.
La ronda a 100 epocas mejora esa base:

```text
3-way original                  0.66649
3-way e100 equal                0.67055
3-way e100 weighted             0.67126
```

El mejor experimento con bagging alcanza `0.67674`, pero no se selecciona como
final oficial porque agrega bagging interno en `globalmel` y `f1024`. Queda
documentado en `06_experimentacion_final/`.

## Validaciones requeridas para cerrar

```bash
python scripts/check_delivery_notebooks_executed.py \
  --include 01_analisis_datos 02_preprocesamiento 03_entrenamiento \
            04_experimentacion_camino 05_experimentacion_general \
            06_experimentacion_final 07_final \
  --exclude "100. Entregable"
```

```bash
python 07_final/validate_final_artifacts.py
```

`100. Entregable/00_pipeline_entregable.ipynb` no se ejecuta en este entorno.

## Criterio para cambios futuros

Una mejora nueva solo reemplaza esta entrega si mantiene una explicacion
comparable o superior en defensa. Si el aumento de score viene de complejidad
adicional, se documenta como experimento y no reemplaza automaticamente al final.
