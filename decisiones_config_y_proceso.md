# Decisiones de configuracion y proceso

Este documento concentra las decisiones que definen la entrega actual.

## Decision principal

La entrega final usa el `3-way e100 reponderado`:

```text
0.25  * separable_headsep_e100_seed42
+ 0.375 * globalmel_sep_temporal_e100_seed42
+ 0.375 * sep_temporal_f1024_e100_seed42
```

Private LB: `0.67126`.

Se elige porque conserva tres componentes fisicos directos, mejora al sistema
simple previo y evita volver a un ensemble anidado dificil de explicar.

## Preprocesamiento

El audio se transforma en espectrogramas log-mel:

```text
.wav
-> mono
-> 44.1 kHz
-> MelSpectrogram
-> log/dB
-> crop/pad temporal
-> normalizacion
-> tensor 128 x T
```

Se usan dos longitudes temporales:

- `512` frames para `separable_headsep` y `globalmel_sep_temporal`;
- `1024` frames para `sep_temporal_f1024`.

Se usan dos politicas de normalizacion:

- por clip, para `separable_headsep` y `sep_temporal_f1024`;
- global por banda mel, para `globalmel_sep_temporal`.

## Arquitecturas

`separable_headsep_e100_seed42` usa una CNN separable-residual sobre log-mel
`128 x 512` y una cabeza densa. Su rol es capturar patrones locales
tiempo-frecuencia.

`globalmel_sep_temporal_e100_seed42` usa normalizacion global mel y una CNN
separable-residual con BiGRU. Su rol es estabilizar escalas por banda y modelar
evolucion temporal.

`sep_temporal_f1024_e100_seed42` usa CNN separable-residual con BiGRU sobre
`128 x 1024`. Su rol es aportar contexto temporal largo.

## Entrenamiento

La ronda final entrena las tres ramas durante `100` epocas. La evidencia de
Kaggle muestra:

```text
3-way original                  0.66649
3-way e100 equal                0.67055
3-way e100 weighted             0.67126
```

La reponderacion final da mayor peso conjunto a las ramas temporales.

## Regularizacion

La regularizacion efectiva del modelo final combina mecanismos dentro del
modelo y del proceso de entrenamiento:

- `head_dropout=0.3` en las tres ramas;
- `BatchNorm` en los bloques convolucionales separables/residuales, y tambien
  en la cabeza densa de `separable_headsep`;
- augmentation tipo SpecAugment sobre el log-mel de entrenamiento: corrimiento
  temporal, mascara temporal y mascara en frecuencia;
- `AdamW` con `weight_decay=1e-4` en las dos ramas temporales;
- perdida multilabel `BCEWithLogitsLoss` con `pos_weight` calculado desde las
  clases de train para compensar desbalance.

Tambien importa lo que no se uso: `block_dropout=0.0`, `early_stopping=0`, sin
time reverse y sin contraste aleatorio. En la rama `separable_headsep`, el
optimizador final es `Adam`; aunque el argumento `weight_decay` tiene default
`1e-4`, el trainer no lo pasa a `Adam`, por lo que esa rama no tiene
regularizacion L2 efectiva.

## Experimentacion separada

La entrega separa experimentos en tres carpetas:

- `04_experimentacion_camino/`: camino hasta modelos fuertes;
- `05_experimentacion_general/`: pruebas laterales y descartes;
- `06_experimentacion_final/`: variantes alrededor del modelo final.

El mejor experimento con bagging (`0.67674`) se conserva como evidencia, pero no
se elige como final oficial porque agrega bagging interno.

## Entregable

`100. Entregable/` contiene el pipeline preparado para reconstruir el CSV final.
No se ejecuta en este entorno; se deja para una corrida posterior en entorno
seguro.
