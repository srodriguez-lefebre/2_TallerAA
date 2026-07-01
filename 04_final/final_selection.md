# Seleccion final

## Objetivo

Dejar trazable que entra al pipeline final y por que. Esta seleccion no reemplaza
la corrida final, pero fija el contrato metodologico: ningun componente entra sin
evidencia, soporte teorico o rol claro en el ensemble.

## Componentes seleccionados para la entrega

| Componente | Rol | Motivo |
|---|---|---|
| `separable_headsep` | diversidad arquitectonica | convoluciones separables/residuales justificadas por curso |
| `globalmel_sep_temporal` | normalizacion global mel | mejora de preprocesamiento defendible |
| `sep_temporal_f1024` | contexto temporal largo | aporta una escala temporal distinta sin agregar otro ensemble |

Componentes historicos como `sklearn`, `head256`, `resnet50` y `sepres_se` quedan
documentados como evidencia de investigacion, pero no entran en el CSV final de
presentacion porque aumentan el conteo real del ensemble.

## Artefactos candidatos validados

| Artefacto | Estado | SHA-256 |
|---|---|---|
| `investigation/kaggle_dataset_download_headsep_conservative/submission.csv` | formato valido; Kaggle private `0.65061` | `00e9c359a402f12880a239ca627133fa752d6f14f0995f5d029e1a45e8713a98` |
| `investigation/kaggle_dataset_download_headsep_translated/submission.csv` | formato valido; Kaggle private `0.65289` | `17eea377cd4ae277029c5215f657381ccb6f1204d4de0b43cfce7a704ff24f9b` |
| `investigation/results/submissions/simple_headsep_globalmel_f1024_equal.csv` | formato valido; Kaggle private `0.66649`; seleccionado | `81ce2b49e836ca89b27e07b2f281eebce3efc103d223c29aa6a3731b7659be9b` |
| `investigation/results/submissions/simple_sepres_headsep_globalmel_f1024_equal.csv` | formato valido; Kaggle private `0.66597` | `3d5dbe6c660426bdadb40ed662fd1d0d8028f49c435b93b8adffb415319f768e` |
| `investigation/results/submissions/current475_globalmel200_se125_f1024_200.csv` | mejor historico expandido; Kaggle private `0.67025`; no seleccionado para presentacion | `e17afe43a164809a6c7cc4ad5ba419c029f01f779cc8bc41759584b14eea5644` |

Seleccion actual:

- `04_final/submission.csv` queda copiado desde
  `investigation/results/submissions/simple_headsep_globalmel_f1024_equal.csv`.
- La seleccion se basa en la consulta de Kaggle:
  `kaggle competitions submissions -c freesound-audio-tagging-2019`.
- La fila usada fue `simple current-free headsep globalmel f1024 equal`, enviada
  el `2026-07-01 19:13:14`, con private score `0.66649`.
- El `publicScore` que devuelve Kaggle es `0.00000` para estas submissions, por
  lo que no se usa como criterio de seleccion.

Equivalencias locales verificadas:

- `kaggle_dataset_download_headsep_conservative/submission.csv` coincide con
  `investigation/submissions/catsdogs_headsep_final/conservative_h10.csv`.
- `kaggle_dataset_download_headsep_translated/submission.csv` coincide con
  `investigation/submissions/catsdogs_headsep_final/translated_local.csv`.
- `04_final/submission.csv` coincide con
  `investigation/results/submissions/simple_headsep_globalmel_f1024_equal.csv`.

Validado contra `data/sample_submission.csv`:

- 3361 filas;
- 81 columnas (`fname` + 80 clases);
- mismo orden de columnas;
- probabilidades en `[0, 1]`.

## Que no entra al final todavia

| Idea | Motivo |
|---|---|
| `logmel_cnn_temporal_bigru` | buena hipotesis y test, pero valid `0.719812` y degrada el blend local |
| `extended_logmel_stats` | sobreajuste local: sube validacion y baja leaderboard |
| `time_reverse_contrast_aug` | degrada contra head256 |
| `noisy_direct_concat` | baja mucho la validacion curated |

## Candidato nuevo a medir

### Ronda teorica 2026-06-29

La carpeta `investigation/results/` agrego una ronda de experimentos apoyados en
ideas del curso:

- TTA multi-crop;
- MC Dropout;
- normalizacion global por banda mel;
- squeeze-and-excitation;
- ventana temporal larga de 1024 frames.

Resultados Kaggle principales:

| Candidato | Private LB | Decision |
|---|---:|---|
| `current475_globalmel200_se125_f1024_200` | `0.67025` | `historical_best_expanded` |
| `current550_globalmel250_f1024_200` | `0.66996` | `keep` |
| `simple_headsep_globalmel_f1024_equal` | `0.66649` | `selected_final` |
| `simple_sepres_headsep_globalmel_f1024_equal` | `0.66597` | `keep` |
| `current645_globalmel_sep_temporal_full355` | `0.66561` | `keep` |
| `current575_globalmel300_se125` | `0.66519` | `blend-only` |
| `current835_sep_temporal_mcdrop20_full165` | `0.65965` | `blend-only` |
| `current835_sep_temporal_tta1024_full165` | `0.65835` | `discard` |

La final con mayor score historico era:

```text
0.475 * current
+ 0.200 * globalmel_sep_temporal
+ 0.125 * sepres_se_head256
+ 0.200 * sep_temporal_f1024
```

Pero `current` se expande en siete piezas anteriores. Por eso la final de
presentacion pasa a ser:

```text
1/3 * separable_headsep
+ 1/3 * globalmel_sep_temporal
+ 1/3 * sep_temporal_f1024
```

Lectura: se pierde `0.00376` de private LB contra el mejor historico
(`0.67025 -> 0.66649`), pero se obtiene una explicacion mucho mas limpia:
soft voting de tres modelos log-mel diversos.

### Candidatos previos superados

`logmel_cnn_separable_temporal_bigru` si aporta como rama de blend: en holdout,
el sistema historico sube de `0.841179` a `0.846742` con 15% de peso de esta
rama. Se completo el full-train `e40 seed42` y se genero:

```text
investigation/results/submissions/current85_sep_temporal_full15.csv
```

No reemplaza todavia a `04_final/submission.csv` porque falta score de Kaggle.
La API directa rechazo la subida con `Submission not allowed: This competition
only accepts Submissions from Notebooks`, asi que el siguiente paso operativo es
subir ese CSV mediante un Notebook de Kaggle.

Despues aparecio `geron_ensemble_search`, que no entrena una red nueva sino que
explora ensambles y post-procesado de ranking por fila inspirados en Geron. El
mejor candidato local es:

```text
investigation/results/submissions/current_rowz205_rowrank190_avg.csv
```

Evidencia local:

- `current + separable_temporal_bigru` aritmetico: `0.846742`;
- mejor aritmetico refinado: `0.846913`;
- `row_z` por fila: `0.847759`;
- `row_z + row_rank`: `0.848191`.

No reemplaza a `04_final/submission.csv` hasta tener score de Kaggle. Es una
mejora local chica y el post-procesado puede sobreajustar el holdout.

Chequeo 2026-06-28 10:20 America/Montevideo:

- kernel `santiagorod247/fat2019-current85-sep-temporal-full15-copy`:
  `COMPLETE`;
- output descargado coincide con SHA
  `5b402849b689f32922d5bd0443e4f94db33b306d8cc165e6e9a78be6f4e3c4b6`;
- la tabla de submissions de `freesound-audio-tagging-2019` todavia no muestra
  el candidato evaluado en ese momento;
- luego fue evaluado y la mejor variante aritmetica alcanzo `0.65928`, superada
  por la ronda teorica del 2026-06-29.

## Narrativa defendible

La evolucion del sistema final se explica por cinco decisiones:

1. Pasar de priors a features log-mel para usar informacion acustica real.
2. Pasar de estadisticas globales a CNN sobre espectrogramas para preservar
   estructura tiempo-frecuencia.
3. Mejorar el entrenamiento neural con inicializacion, scheduler, BatchNorm,
   dropout y cabeza densa.
4. Combinar familias con errores distintos durante la investigacion para medir
   diversidad real.
5. Refinar la entrega hacia un ensemble chico: separable residual fuerte,
   normalizacion global por banda mel y ventana temporal de 1024 frames.

Los pesos exactos del blend deben reportarse como seleccion empirica si vienen
de barridos contra validacion/Kaggle. La justificacion fuerte es la diversidad y
la evidencia de mejora, no que un peso particular sea teoricamente especial.

## Siguiente paso

Para cerrar el pipeline final de forma reproducible:

1. Mantener `04_final/submission.csv` como artefacto final defendible con private
   LB `0.66649`, salvo que un nuevo experimento simple de `investigation/results/`
   supere este score con evidencia comparable.
2. Usar el candidato `0.67025` solo como referencia de mejor score historico
   expandido, no como final de presentacion.
3. Ejecutar el notebook `04_final/01_pipeline_final.ipynb` y dejar la validacion
   de formato en verde.
