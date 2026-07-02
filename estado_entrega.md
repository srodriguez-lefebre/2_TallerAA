# Estado de entrega - Proyecto 2

Este archivo resume la auditoria de cumplimiento del objetivo actual.

La politica vigente es la de `100. Entregable`: la entrega principal debe ser
simple, reproducible y defendible. El mejor score historico se conserva como
referencia, pero no reemplaza a la seleccion de presentacion si depende de un
ensemble anidado dificil de explicar.

## Requisitos y evidencia

| Requisito | Evidencia actual | Estado |
|---|---|---|
| Leer consigna del Proyecto 2 | `roadmap.md` resume audio tagging multi-label, log-mel/MFCC y `lwlrap` desde `docs/proyecto_2_audio.pdf` y `docs/proyecto_2_freesound.pdf` | cumplido |
| Usar menos carpetas que Taller 1 | flujo comprimido en `01_analisis_datos/`, `02_preprocesamiento/`, `03_entrenamiento/`, `04_final/` y cierre en `100. Entregable/` | cumplido |
| Inspirarse en Taller 1 sin copiar su proliferacion | `100. Entregable/00_pipeline_entregable.ipynb` sigue el estilo de notebook final autonomo: datos, configuracion, preprocesamiento, modelo, blend y validacion | cumplido |
| Revisar `investigation/` y promover solo ideas defendibles | `03_entrenamiento/decision_matrix.csv` separa `keep`, `blend-only`, `discard` y `reference` | cumplido |
| Usar material del curso como soporte | `roadmap.md`, `decision_matrix.csv` y las fichas de `100. Entregable/componentes/` conectan CNN, regularizacion, BatchNorm, dropout, scheduler, normalizacion y ensembles con el curso | cumplido |
| Incorporar experimentos nuevos de `investigation/results/` | `investigation/results/README.md`, `experiment_log.csv` y `theory_supported_experiments_2026_06_29.md` registran los candidatos finales | cumplido |
| Entrenar modelos planteados para la entrega | `03_entrenamiento/training_results.csv` registra baselines y modelos neurales; `100. Entregable/componentes/` fija las tres ramas finales | cumplido |
| Revisar mejoras nuevas de `investigation/results/` | el candidato `0.67025` queda como mejor historico expandido; no se selecciona como entrega porque contiene el ensemble anidado `current` | cumplido |
| No tocar Kaggle del curso | `README.md`, `roadmap.md`, `04_final/01_pipeline_final.ipynb` y `100. Entregable/00_pipeline_entregable.ipynb` limitan submissions al desafio publico `freesound-audio-tagging-2019` | cumplido |
| Dejar pipeline final y submission validada | `100. Entregable/00_pipeline_entregable.ipynb`, `100. Entregable/submission.csv`, `04_final/submission.csv`, `final_pipeline_manifest.csv`, `submission_candidates.csv`, `final_pipeline_metadata.json` y `final_selection.md` | cumplido |

## Decision final actual

La submission de presentacion es:

```text
100. Entregable/submission.csv
```

Debe coincidir byte a byte con el artefacto auditado:

```text
04_final/submission.csv
```

Origen:

```text
investigation/results/submissions/simple_headsep_globalmel_f1024_equal.csv
```

Formula:

```text
1/3 * separable_headsep
+ 1/3 * globalmel_sep_temporal
+ 1/3 * sep_temporal_f1024
```

Evidencia:

- SHA-256: `81ce2b49e836ca89b27e07b2f281eebce3efc103d223c29aa6a3731b7659be9b`;
- coincide con `investigation/results/submissions/simple_headsep_globalmel_f1024_equal.csv`;
- Kaggle private score: `0.66649`;
- descripcion Kaggle: `simple current-free headsep globalmel f1024 equal`;
- competencia: `freesound-audio-tagging-2019`.

## Referencia historica

El mejor score historico expandido sigue siendo:

```text
investigation/results/submissions/current475_globalmel200_se125_f1024_200.csv
```

con private LB `0.67025`.

No se usa como entrega principal porque su formula contiene `current`, que se
expande a varias ramas previas. Para la presentacion se prioriza un soft vote
directo de tres componentes reales, con una perdida pequena de leaderboard
frente a una explicacion mucho mas limpia.

## Validaciones ejecutadas

```bash
python 04_final/validate_final_artifacts.py
```

Resultado esperado: `final_validation_ok`.

```bash
jupyter nbconvert --execute --inplace "100. Entregable/00_pipeline_entregable.ipynb"
```

Resultado esperado: se reconstruye `100. Entregable/submission.csv` en modo
seguro, sin reentrenar ramas largas.

```bash
cd investigation
python -m unittest tests.test_train_logmel_cnn
```

Resultado historico: `11 tests OK`.

Tambien se ejecutaron los notebooks principales de trabajo:

- `01_analisis_datos/01_analisis_datos.ipynb`;
- `02_preprocesamiento/00_C00_base_audio.ipynb`;
- `02_preprocesamiento/01_P00_logmel_stats_basic.ipynb`;
- `02_preprocesamiento/02_P01_logmel_image_512.ipynb`;
- `02_preprocesamiento/03_P02_globalmel_y_f1024.ipynb`;
- `02_preprocesamiento/10_p00vsp01vsp02.ipynb`;
- `03_entrenamiento/01_baselines_y_modelos.ipynb`;
- `04_final/01_pipeline_final.ipynb`.

## Criterio para cambios futuros

Una nueva idea de `investigation/results/` solo reemplaza la entrega si cumple
las dos condiciones:

1. mejora o empata de forma razonable el private LB de `0.66649`;
2. mantiene una explicacion comparable de presentacion: pocas ramas reales,
   pesos defendibles y sin esconder ensembles anidados.

Si solo mejora el score usando una mezcla grande, queda como referencia
historica o como candidato de investigacion, no como entrega principal.
