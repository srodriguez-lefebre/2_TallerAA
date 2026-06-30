# Estado de entrega - Proyecto 2

Este archivo resume la auditoria de cumplimiento del objetivo actual.

## Requisitos y evidencia

| Requisito | Evidencia actual | Estado |
|---|---|---|
| Leer consigna del Proyecto 2 | `roadmap.md` resume audio tagging multi-label, log-mel/MFCC y `lwlrap` desde `docs/proyecto_2_audio.pdf` y `docs/proyecto_2_freesound.pdf` | cumplido |
| Usar menos carpetas que Taller 1 | estructura reducida: `01_analisis_datos/`, `02_preprocesamiento/`, `03_entrenamiento/`, `04_final/` | cumplido |
| Inspirarse en Taller 1 sin copiar su proliferacion | `roadmap.md` conserva el flujo problema -> datos -> preprocesamiento -> modelos -> final, comprimido en cuatro bloques | cumplido |
| Revisar `investigation/` y promover solo ideas defendibles | `03_entrenamiento/decision_matrix.csv` separa `keep`, `blend-only`, `discard` y `reference` | cumplido |
| Usar material del curso como soporte | `roadmap.md` y `decision_matrix.csv` conectan CNN, regularizacion, BatchNorm, dropout, scheduler, transfer y ensembles con los PDFs/libro del curso | cumplido |
| Incorporar experimentos nuevos de `investigation/results/` | `investigation/results/README.md`, `experiment_log.csv` y `theory_supported_experiments_2026_06_29.md` registran la mejora final | cumplido |
| Entrenar modelos planteados para la entrega | `03_entrenamiento/training_results.csv` registra sklearn, CNN estandar, CNN head256 y separable-residual frescos | cumplido |
| Revisar mejoras nuevas de `investigation/results/` | `theory_supported_experiments_2026_06_29.md` sube la mejor Kaggle a `0.67025` | cumplido |
| No tocar Kaggle del curso | `README.md`, `roadmap.md` y `04_final/01_pipeline_final.ipynb` limitan submissions al desafio publico `freesound-audio-tagging-2019` | cumplido |
| Dejar pipeline final y submission validada | `04_final/submission.csv`, `final_pipeline_manifest.csv`, `submission_candidates.csv`, `final_pipeline_metadata.json` y `final_selection.md` | cumplido |

## Decision final actual

La submission seleccionada es:

```text
04_final/submission.csv
```

Origen:

```text
investigation/results/submissions/current475_globalmel200_se125_f1024_200.csv
```

Evidencia:

- SHA-256: `e17afe43a164809a6c7cc4ad5ba419c029f01f779cc8bc41759584b14eea5644`;
- coincide byte a byte con `investigation/results/submissions/current475_globalmel200_se125_f1024_200.csv`;
- Kaggle private score: `0.67025`;
- descripcion Kaggle: `current475 globalmel200 se125 f1024_200`;
- competencia: `freesound-audio-tagging-2019`.

## Validaciones ejecutadas

```bash
python 04_final/validate_final_artifacts.py
```

Resultado: `final_validation_ok`.

```bash
cd investigation
python -m unittest tests.test_train_logmel_cnn
```

Resultado: `11 tests OK`.

Tambien se ejecutaron los notebooks principales de entrega:

- `01_analisis_datos/01_analisis_datos.ipynb`;
- `02_preprocesamiento/00_C00_base_audio.ipynb`;
- `02_preprocesamiento/01_P00_logmel_stats_basic.ipynb`;
- `02_preprocesamiento/02_P01_logmel_image_512.ipynb`;
- `02_preprocesamiento/03_P02_globalmel_y_f1024.ipynb`;
- `02_preprocesamiento/10_p00vsp01vsp02.ipynb`;
- `03_entrenamiento/01_baselines_y_modelos.ipynb`;
- `04_final/01_pipeline_final.ipynb`.

Resultado: notebooks ejecutados y guardados con outputs.

## Pendiente normal de trabajo

La entrega actual tiene una final Kaggle-verificada:

```text
04_final/submission.csv
```

La ronda anterior dejo un candidato intermedio:

```text
investigation/results/submissions/current85_sep_temporal_full15.csv
```

Ese candidato tiene mejora local estimada (`0.841179 -> 0.846742`) y formato
validado, pero todavia no reemplaza a la final porque falta score de Kaggle. La
API directa rechazo la subida con:

```text
Submission not allowed: This competition only accepts Submissions from Notebooks.
```

El siguiente paso operativo es medir ese CSV desde un Notebook de Kaggle, igual
que las submissions historicas.

Revision 2026-06-28 10:20 America/Montevideo:

- el kernel `santiagorod247/fat2019-current85-sep-temporal-full15-copy` esta
  `COMPLETE`;
- su output `submission.csv` coincide byte a byte con el candidato local;
- Kaggle todavia no muestra score para ese candidato en la tabla de submissions;
- en ese chequeo, la mejor submission evaluada seguia siendo `0.65289`;
- aparecio una mejora local nueva, `geron_ensemble_search`, con mejor candidato
  `investigation/results/submissions/current_rowz205_rowrank190_avg.csv` y
  `valid_lwlrap=0.848191` en la reconstruccion local;
- ese candidato quedaba como `candidate_needs_kaggle_notebook`, no como final.

Revision 2026-06-29:

- `current85_sep_temporal_full15` fue evaluado y dio private LB `0.65848`;
- `current835_sep_temporal_full165` mejoro a `0.65928`;
- la ronda teorica agrego normalizacion global mel, squeeze-and-excitation y
  ventana temporal 1024;
- mejor resultado actual: `current475_globalmel200_se125_f1024_200`, private LB
  `0.67025`;
- `04_final/submission.csv` fue actualizado a esa submission.
