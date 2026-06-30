# Seleccion final

## Objetivo

Dejar trazable que entra al pipeline final y por que. Esta seleccion no reemplaza
la corrida final, pero fija el contrato metodologico: ningun componente entra sin
evidencia, soporte teorico o rol claro en el ensemble.

## Componentes seleccionados

| Componente | Rol | Motivo |
|---|---|---|
| `sklearn_logmel_c001` | diversidad tabular | baseline regularizado, barato y con errores distintos |
| `head256_relu_cnn` | rama neural principal | mejora fuerte con He, scheduler y cabeza densa |
| `separable_headsep` | diversidad arquitectonica | convoluciones separables/residuales justificadas por curso |
| `resnet50_frozen` | transfer/blend-only | debil solo, pero aporta diversidad al blend |
| `globalmel_sep_temporal` | normalizacion global mel | mejora de preprocesamiento defendible |
| `sepres_se_head256` | atencion por canales | aporta en el ensamble final aunque no gane solo |
| `sep_temporal_f1024` | contexto temporal largo | mejor rama nueva de la ronda teorica |

## Artefactos candidatos validados

| Artefacto | Estado | SHA-256 |
|---|---|---|
| `investigation/kaggle_dataset_download_headsep_conservative/submission.csv` | formato valido; Kaggle private `0.65061` | `00e9c359a402f12880a239ca627133fa752d6f14f0995f5d029e1a45e8713a98` |
| `investigation/kaggle_dataset_download_headsep_translated/submission.csv` | formato valido; Kaggle private `0.65289` | `17eea377cd4ae277029c5215f657381ccb6f1204d4de0b43cfce7a704ff24f9b` |
| `investigation/results/submissions/current475_globalmel200_se125_f1024_200.csv` | formato valido; Kaggle private `0.67025` | `e17afe43a164809a6c7cc4ad5ba419c029f01f779cc8bc41759584b14eea5644` |

Seleccion actual:

- `04_final/submission.csv` queda copiado desde
  `investigation/results/submissions/current475_globalmel200_se125_f1024_200.csv`.
- La seleccion se basa en la consulta de Kaggle:
  `kaggle competitions submissions -c freesound-audio-tagging-2019`.
- La fila usada fue `current475 globalmel200 se125 f1024_200`, enviada el
  `2026-06-29 09:58:43`, con private score `0.67025`.
- El `publicScore` que devuelve Kaggle es `0.00000` para estas submissions, por
  lo que no se usa como criterio de seleccion.

Equivalencias locales verificadas:

- `kaggle_dataset_download_headsep_conservative/submission.csv` coincide con
  `investigation/submissions/catsdogs_headsep_final/conservative_h10.csv`.
- `kaggle_dataset_download_headsep_translated/submission.csv` coincide con
  `investigation/submissions/catsdogs_headsep_final/translated_local.csv`.
- `04_final/submission.csv` coincide con
  `investigation/results/submissions/current475_globalmel200_se125_f1024_200.csv`.

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
| `current475_globalmel200_se125_f1024_200` | `0.67025` | `selected_final` |
| `current550_globalmel250_f1024_200` | `0.66996` | `keep` |
| `current645_globalmel_sep_temporal_full355` | `0.66561` | `keep` |
| `current575_globalmel300_se125` | `0.66519` | `blend-only` |
| `current835_sep_temporal_mcdrop20_full165` | `0.65965` | `blend-only` |
| `current835_sep_temporal_tta1024_full165` | `0.65835` | `discard` |

La nueva final es:

```text
0.475 * current
+ 0.200 * globalmel_sep_temporal
+ 0.125 * sepres_se_head256
+ 0.200 * sep_temporal_f1024
```

Lectura: la mejora no viene de un truco aislado, sino de combinar tres ideas
defendibles: normalizacion global por banda mel, atencion por canales y mayor
contexto temporal.

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

La evolucion del sistema final se explica por cuatro decisiones:

1. Pasar de priors a features log-mel para usar informacion acustica real.
2. Pasar de estadisticas globales a CNN sobre espectrogramas para preservar
   estructura tiempo-frecuencia.
3. Mejorar el entrenamiento neural con inicializacion, scheduler, BatchNorm,
   dropout y cabeza densa.
4. Combinar familias con errores distintos: sklearn, CNN head256, separable
   residual y transfer congelado.
5. Refinar preprocesamiento y contexto temporal: normalizacion global por banda
   mel y ventana temporal de 1024 frames.

Los pesos exactos del blend deben reportarse como seleccion empirica si vienen
de barridos contra validacion/Kaggle. La justificacion fuerte es la diversidad y
la evidencia de mejora, no que un peso particular sea teoricamente especial.

## Siguiente paso

Para cerrar el pipeline final de forma reproducible:

1. Mantener `04_final/submission.csv` como artefacto final elegido con private
   LB `0.67025`, salvo que un nuevo experimento de `investigation/results/`
   supere este score con evidencia comparable.
2. Completar los pesos exactos del blend si se necesita reconstruir la submission
   desde cero y no solo entregar el CSV validado.
3. Ejecutar el notebook `04_final/01_pipeline_final.ipynb` y dejar la validacion
   de formato en verde.
