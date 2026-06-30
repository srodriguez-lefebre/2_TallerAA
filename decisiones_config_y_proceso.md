# Decisiones de configuracion y proceso

Este documento consolida, en un solo lugar, las decisiones tomadas para el
Proyecto 2 de Taller AA. La idea es que sirva como base de informe: no solo dice
que modelo se uso, sino por que se eligio, contra que se comparo, que numero dio
y que limitacion queda.

## Criterio general

Una decision entra al proyecto si cumple al menos una de estas condiciones:

- esta motivada por la consigna, por ejemplo `lwlrap`, STFT, escala Mel o
  representaciones logaritmicas;
- esta respaldada por el curso, por ejemplo CNNs, regularizacion, BatchNorm,
  dropout, scheduling de learning rate, transfer learning o ensembles;
- mejora contra un baseline comparable en validacion local o Kaggle;
- no mejora como modelo individual, pero aporta diversidad medible al blend.

Una decision se descarta si:

- mejora solo un split local pero baja en Kaggle;
- agrega complejidad sin una hipotesis tecnica clara;
- degrada `valid_lwlrap`;
- requiere datos ruidosos sin control;
- no tiene artefactos suficientes para reproducir o auditar.

## Estructura del proyecto

Decision: usar cuatro bloques principales.

| Carpeta | Rol | Motivo |
|---|---|---|
| `01_analisis_datos/` | entender consigna, datos, clases, splits y metrica | separa el problema antes de modelar |
| `02_preprocesamiento/` | preparar representaciones de audio | concentra STFT/log-mel/MFCC y caches |
| `03_entrenamiento/` | comparar modelos y mejoras | contiene resultados, matriz de decisiones y corridas |
| `04_final/` | armar submission final | deja la seleccion y validacion final trazable |

Esto sigue la idea metodologica del Taller 1, pero sin repetir la proliferacion
de carpetas numeradas. `investigation/` queda como laboratorio historico y
`investigation/results/` como registro vivo de experimentos nuevos.

## Datos y metrica

Decision: usar `lwlrap` como metrica principal.

Motivo: es la metrica de la consigna Freesound Audio Tagging 2019 y evalua el
ranking de clases positivas en un problema multi-label. No se usa accuracy
porque cada audio puede tener varias etiquetas y el desbalance de clases es
fuerte.

Decision: usar `train_curated.csv` para validacion controlada inicial.

Motivo: las etiquetas curated son mas confiables. `train_noisy.csv` no se usa
por defecto; queda como material experimental, porque concatenarlo directo
degrado mucho la validacion curated.

Evidencia:

- `noisy_direct_concat`: `valid_lwlrap=0.3567`, descartado.

## Preprocesamiento

Decision: usar log-mel como representacion central.

Motivo: la consigna de audio motiva STFT, magnitudes logaritmicas y escala Mel.
El log-mel conserva informacion frecuencia-tiempo y es una entrada natural para
CNNs.

Representaciones usadas:

| Representacion | Uso | Estado |
|---|---|---|
| priors por clase | smoke test de submission | `reference` |
| stats log-mel | baseline tabular | `keep` |
| imagen log-mel `128 x 512` | entrada principal para CNN | `keep` |
| extended log-mel stats | experimento de features | `discard` |
| imagen log-mel `128 x 1024` | prueba de mayor largo | no promover por defecto |

Evidencia relevante:

- `sklearn_logmel_stats`: Private LB `0.32714`, primer salto usando audio real.
- `extended_logmel_stats`: valid local sube, pero Private LB baja a `0.36167`;
  se interpreta como sobreajuste local.
- `1024 frames`: bajo contra 512 en la prueba historica, no se promovio.

## Baseline tabular

Decision: entrenar `LogisticRegression` One-vs-Rest sobre stats log-mel.

Configuracion:

- features: estadisticas log-mel basicas;
- clasificador: `OneVsRestClassifier(LogisticRegression)`;
- `class_weight="balanced"`;
- `StandardScaler`;
- split holdout 80/20 con seed 42.

Motivo: es barato, reproducible, multilabel via One-vs-Rest y usa audio real.
Sirve como baseline fuerte y como rama diversa para blends.

Corridas frescas:

| Configuracion | valid_lwlrap | Decision |
|---|---:|---|
| `C=0.01` | `0.576401` | `keep` |
| `C=0.02` | `0.576353` | `discard` |
| `C=0.03` | `0.570624` | `discard` |

Decision final: mantener `C=0.01`.

Motivo: fue el mejor punto fresco y tambien el mejor historico en Kaggle dentro
de esta familia. Seguir micro-ajustando `C` no dio mejoras robustas.

## CNN sobre log-mel

Decision: pasar de stats globales a CNN sobre imagen log-mel.

Motivo: las stats colapsan el eje temporal; una CNN puede aprender patrones
locales frecuencia-tiempo.

Evidencia historica:

- mejor sklearn `C=0.01`: Private LB `0.37607`;
- CNN log-mel historica: Private LB `0.52257`;
- salto: `+0.14650`.

Corrida fresca de referencia:

| Corrida | Configuracion | valid_lwlrap | Lectura |
|---|---|---:|---|
| `cnn_standard_e12` | CNN estandar, 12 epochs, cosine scheduler | `0.493959` | documenta el paso, pero queda subentrenada |

La corrida corta no supera al baseline tabular, pero la corrida historica larga
si lo hace. Por eso se mantiene como escalon metodologico y no como modelo final
aislado.

## Mejoras de entrenamiento neural

Decision: usar inicializacion He, scheduler por plateau y cabeza densa de 256.

Configuracion:

- `initializer=he_normal`;
- `scheduler=plateau`;
- `lr=5e-4`;
- `head_hidden=256`;
- BatchNorm en la cabeza;
- dropout de cabeza;
- salida sigmoid/BCE multilabel.

Motivo: el curso respalda inicializacion, BatchNorm, dropout, learning-rate
scheduling y regularizacion. En la evidencia historica, esta familia fue el
salto fuerte posterior a la CNN base.

Corrida fresca:

| Corrida | valid_lwlrap | Decision |
|---|---:|---|
| `cnn_head256_he_plateau_e24` | `0.673265` | `keep` |

Referencia historica:

| Corrida | valid_lwlrap | Decision |
|---|---:|---|
| `logmel_cnn_fashion_he_plateau_head256` | `0.783814` | `keep` |

Lectura: la corrida fresca confirma la direccion. La corrida historica larga es
la que justifica usar esta familia para full-train y blends.

## Arquitectura separable-residual

Decision: usar CNN separable-residual con cabeza densa.

Configuracion fresca:

- `architecture=separable_residual`;
- `activation=relu`;
- `initializer=he_normal`;
- `scheduler=plateau`;
- `head_hidden=256`;
- `head_dropout=0.30`;
- 24 epochs.

Motivo: el curso cubre ResNet, Xception/separable convolutions y arquitecturas
profundas. Las convoluciones separables reducen parametros y los atajos
residuales facilitan entrenamiento de bloques mas expresivos.

Corrida fresca:

| Corrida | valid_lwlrap | Decision |
|---|---:|---|
| `cnn_sepres_head256_e24` | `0.738289` | `keep` |

Referencia historica:

| Corrida | valid_lwlrap | Decision |
|---|---:|---|
| `logmel_cnn_catsdogs_sepres_head256` | `0.807813` | `keep` |

Decision final: promover como rama arquitectonica fuerte.

## Transfer learning con ResNet50

Decision: usar ResNet50 congelada solo como diversidad de ensemble.

Motivo: la rama es debil individualmente, pero sus errores son distintos. El
curso respalda transfer learning, pero no se presenta como mejor modelo.

Evidencia:

- `resnet50_imagenet_frozen`: decision `blend-only`;
- aporta en blends finales con peso bajo.

Limitacion: no debe describirse como modelo principal.

## Blends y ensembles

Decision: combinar familias con errores distintos.

Motivo: el curso y el libro respaldan ensembles. En este problema, modelos
distintos capturan senales distintas:

- sklearn: resumen global por bandas, duracion, RMS, ZCR;
- CNN: patrones locales del espectrograma;
- separable/residual: arquitectura mas expresiva;
- ResNet50: transferencia visual, debil individual pero diversa;
- ramas temporales: lectura explicita del orden temporal.

Evidencia historica:

| Cambio | Resultado |
|---|---:|
| CNN + sklearn | Private LB sube hasta `0.56682` |
| full-train + 2 seeds CNN | Private LB sube a `0.58612` |
| etapa head256/ReLU/literal | Private LB sube a `0.64665` |
| separable + ResNet50 + headsep | Private LB sube a `0.65289` |
| separable temporal + global-mel + SE + 1024 frames | Private LB final `0.67025` |

Nota metodologica: los pesos exactos de blend se reportan como seleccion
empirica. La justificacion fuerte es la diversidad de errores y la mejora
observada, no que un peso especifico sea teoricamente especial.

### Busqueda de ensambles inspirada en Geron

Decision: registrar `geron_ensemble_search` como candidato de blend/post-procesado,
pero no como final.

Motivo: el libro de Geron respalda soft voting, stacking y ensembles cuando los
predictores cometen errores distintos. La busqueda nueva probo:

- promedios simples;
- pesos aleatorios Dirichlet;
- stacking logistico por clase;
- transformaciones por fila (`row_z`, `row_rank`, logit/geometrica).

Evidencia:

| Variante | valid_lwlrap local | Kaggle private | Decision |
|---|---:|---:|---|
| ensambles crudos OOF | `0.841229` | n/a | no supera baseline local |
| stacking logistico OOF | `0.825933` | n/a | descartar |
| aritmetico `current + sep temporal` refinado | `0.846913` | `0.65928` | keep, superado luego |
| `row_z` por fila | `0.847759` | `0.65783` | blend-only, superado |
| `row_z + row_rank` | `0.848191` | `0.65772` | overfit local |

Artefacto principal:

```text
investigation/results/submissions/current_rowz205_rowrank190_avg.csv
```

SHA-256:

```text
e5a851ae653d4b0cb3245f0504f51f4384432c0a7217f51165e1a09999461ee0
```

Limitacion: la mejora local del post-procesado por fila no generalizo a Kaggle.
El mejor resultado Kaggle de esa ronda fue el blend aritmetico simple
`current835_sep_temporal_full165`, con private LB `0.65928`, luego superado por
la ronda teorica del 2026-06-29.

### Ronda teorica 2026-06-29

Decision: probar solo ideas defendibles por el curso o por teoria de modelos y
promoverlas si mejoran Kaggle o aportan diversidad clara.

Ideas evaluadas:

| Idea | Justificacion | Resultado | Decision |
|---|---|---:|---|
| TTA multi-crop | ventanas temporales e invariancia en CNNs | Kaggle `0.65835` | descartar |
| MC Dropout | dropout como regularizacion e inferencia aproximada | Kaggle `0.65965` | blend-only |
| TTA + MC Dropout | combinar vistas temporales e incertidumbre | Kaggle `0.65795` | descartar |
| normalizacion global por banda mel | preprocesamiento con media/varianza controlada | holdout `0.813036`, blend Kaggle `0.66561` | keep |
| squeeze-and-excitation | atencion por canales en CNNs profundas | holdout `0.805921`, blend Kaggle `0.66519` | blend-only |
| temporal `frames=1024` | mayor contexto temporal y multiescala | holdout `0.801006`, blend Kaggle `0.67025` | keep |

Formula del mejor blend:

```text
0.475 * current
+ 0.200 * globalmel_sep_temporal
+ 0.125 * sepres_se_head256
+ 0.200 * sep_temporal_f1024
```

Lectura: la mejora no viene de una sola perilla aislada. La normalizacion global
por banda mel aporta una rama mas estable, `frames=1024` agrega contexto
temporal, y SE ayuda poco de forma individual pero aporta diversidad en el
ensamble de cuatro ramas. El mejor CSV medido es
`current475_globalmel200_se125_f1024_200`, con private LB `0.67025`.

## Full-train y seeds

Decision: usar full-train solo para configuraciones ya elegidas.

Motivo: cuando una arquitectura fue seleccionada con holdout, se reentrena con
todo `train_curated.csv` para generar submission final. El seed ensemble reduce
varianza.

Evidencia:

- full-train CNN mejora el blend frente a holdout;
- dos seeds mejoran mas que una;
- agregar una tercera seed historicamente no mejoro.

Regla: no agregar seeds indefinidamente; solo si aportan mejora medible.

## Decisiones descartadas

| Idea | Motivo de descarte |
|---|---|
| priors como modelo | solo valida formato; no escucha audio |
| calibracion por priors | rompe ranking y baja leaderboard |
| extended log-mel stats | mejora local, baja Kaggle |
| noisy directo | `valid_lwlrap` cae a `0.3567` |
| time reverse / contrast augmentations | degradan contra head256 |
| `temporal_bigru` simple | `valid_lwlrap=0.719812` y empeora blend local |
| micro-ajustes sin registro | no son defendibles |

## Ramas temporales

Decision: probar temporalidad solo si se compara contra un baseline fuerte.

### `temporal_bigru`

Configuracion:

- CNN liviana;
- BiGRU temporal;
- pooling temporal `mean + max`.

Resultado:

- `valid_lwlrap=0.719812`;
- blend local baja de `0.841179` a `0.840031` con 2.5% de peso.

Decision: `discard`.

### `separable_temporal_bigru`

Configuracion:

- backbone separable-residual;
- sin `AdaptiveAvgPool2d` final;
- promedio sobre frecuencia;
- BiGRU temporal;
- pooling temporal `mean + max`;
- full-train posterior de 40 epochs.

Resultado holdout:

- individual: `valid_lwlrap=0.797079`;
- sistema historico: `0.841179`;
- sistema + 15% rama temporal: `0.846742`.

Decision: `blend-only` y candidato a medir.

Se completo full-train:

```text
investigation/submissions/logmel_cnn_separable_temporal_bigru_full_e40_seed42/small_logmel_cnn.csv
```

Blend candidato:

```text
investigation/results/submissions/current85_sep_temporal_full15.csv
```

Formula:

```text
0.85 * investigation/submissions/catsdogs_headsep_final/translated_local.csv
+ 0.15 * investigation/submissions/logmel_cnn_separable_temporal_bigru_full_e40_seed42/small_logmel_cnn.csv
```

SHA-256:

```text
5b402849b689f32922d5bd0443e4f94db33b306d8cc165e6e9a78be6f4e3c4b6
```

Limitacion: todavia no reemplaza la final porque falta score Kaggle. La API
directa fue rechazada por la competencia:

```text
Submission not allowed: This competition only accepts Submissions from Notebooks.
```

## Submission final actual

Decision: mantener como final verificada:

```text
04_final/submission.csv
```

Origen:

```text
investigation/results/submissions/current475_globalmel200_se125_f1024_200.csv
```

Evidencia:

- SHA-256:
  `e17afe43a164809a6c7cc4ad5ba419c029f01f779cc8bc41759584b14eea5644`;
- Kaggle private score: `0.67025`;
- descripcion Kaggle: `current475 globalmel200 se125 f1024_200`;
- competencia: `freesound-audio-tagging-2019`;
- formato validado contra `data/sample_submission.csv`.

La submission anterior `catsdogs headsep translated local verified` queda como
antecedente reproducible con private LB `0.65289`, pero fue superada por la rama
temporal, global-mel y el blend final de la ronda teorica.

No se publica nada en el Kaggle del curso.

## Comandos clave

Baseline sklearn:

```bash
python investigation/scripts/train_sklearn_variants.py \
  --data-dir data \
  --models logreg_c001,logreg_c002,logreg_c003 \
  --feature-set basic \
  --seed 42 \
  --test-size 0.2
```

CNN head256:

```bash
python investigation/scripts/train_logmel_cnn.py \
  --data-dir data \
  --n-mels 128 \
  --frames 512 \
  --architecture standard \
  --initializer he_normal \
  --scheduler plateau \
  --head-hidden 256 \
  --epochs 24 \
  --seed 42 \
  --test-size 0.2
```

CNN separable-residual:

```bash
python investigation/scripts/train_logmel_cnn.py \
  --data-dir data \
  --n-mels 128 \
  --frames 512 \
  --architecture separable_residual \
  --activation relu \
  --initializer he_normal \
  --scheduler plateau \
  --head-hidden 256 \
  --head-dropout 0.30 \
  --epochs 24 \
  --seed 42 \
  --test-size 0.2
```

Full-train temporal candidato:

```bash
python investigation/scripts/train_logmel_cnn.py \
  --data-dir data \
  --models-dir investigation/models/logmel_cnn_separable_temporal_bigru_full_e40_seed42 \
  --submissions-dir investigation/submissions/logmel_cnn_separable_temporal_bigru_full_e40_seed42 \
  --experiments-dir investigation/experiments/logmel_cnn_separable_temporal_bigru_full_e40_seed42 \
  --n-mels 128 \
  --frames 512 \
  --epochs 40 \
  --batch-size 24 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --optimizer adamw \
  --initializer he_normal \
  --architecture separable_temporal_bigru \
  --activation silu \
  --head-dropout 0.30 \
  --scheduler multistep \
  --lr-milestones 32,38 \
  --plateau-factor 0.5 \
  --seed 42 \
  --test-size 0.2 \
  --num-workers 2 \
  --full-train
```

Blend candidato:

```bash
python investigation/scripts/blend_submissions.py \
  --sample data/sample_submission.csv \
  --input investigation/submissions/catsdogs_headsep_final/translated_local.csv \
  --weight 0.85 \
  --input investigation/submissions/logmel_cnn_separable_temporal_bigru_full_e40_seed42/small_logmel_cnn.csv \
  --weight 0.15 \
  --output investigation/results/submissions/current85_sep_temporal_full15.csv
```

Validacion final:

```bash
python 04_final/validate_final_artifacts.py
```

Resultado esperado:

```text
final_validation_ok
```
