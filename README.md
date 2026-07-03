# Taller AA - Proyecto 2

Proyecto academico para Freesound Audio Tagging 2019.

La entrega se organiza como un paquete de 8 carpetas que se puede copiar y leer
sin depender de todo el laboratorio historico. El laboratorio completo queda en
`investigation/`; la narrativa de defensa queda en las carpetas numeradas.

## Estructura de entrega

- `01_analisis_datos/`: problema, metrica, splits, clases y primeros chequeos.
- `02_preprocesamiento/`: flujo de audio a log-mel, normalizaciones y formas de
  entrada `128 x 512` y `128 x 1024`.
- `03_entrenamiento/`: entrenamiento del modelo final de tres ramas.
- `04_experimentacion_camino/`: camino hasta llegar a modelos fuertes.
- `05_experimentacion_general/`: pruebas laterales, descartes y experimentos que
  no forman la narrativa principal.
- `06_experimentacion_final/`: pruebas alrededor del modelo final y decision de
  elegir el 3-way e100 reponderado.
- `07_final/`: seleccion final, CSV, hash y validacion de formato.
- `100. Entregable/`: pipeline final preparado para ejecutarse en un entorno
  seguro.

## Modelo final oficial

```text
0.25  * separable_headsep_e100_seed42
+ 0.375 * globalmel_sep_temporal_e100_seed42
+ 0.375 * sep_temporal_f1024_e100_seed42
```

Kaggle private LB: `0.67126`.

El mejor experimento con bagging alcanza `0.67674`, pero no se selecciona como
entrega principal porque agrega bagging interno en dos ramas temporales. Queda
documentado en `06_experimentacion_final/`.

## Regla de alcance

Solo se publican submissions en el desafio publico de Kaggle Freesound Audio
Tagging 2019. El proyecto de Kaggle del curso queda fuera de esta estructura.

## Validacion de cierre

El contrato de validacion esta en:

```text
docs/plans/2026-07-02-reorganizacion-entrega-validacion.md
```

Los notebooks de `01` a `07` deben quedar ejecutados y guardados con outputs.
`100. Entregable` no se ejecuta en este entorno.

## Fuentes

- Consigna: `docs/proyecto_2_audio.pdf` y `docs/proyecto_2_freesound.pdf`.
- Curso: PDFs en `course/`.
- Evidencia experimental previa: `investigation/` y `extra_result/`.
