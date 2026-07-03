# 100 - Entregable

Esta carpeta contiene el notebook final de entrega:

- `00_pipeline_entregable.ipynb`

El notebook queda preparado para reconstruir el CSV final en un entorno seguro.
No se ejecuta durante esta reorganizacion local.

## Modelo final oficial

```text
0.25  * separable_headsep_e100_seed42
+ 0.375 * globalmel_sep_temporal_e100_seed42
+ 0.375 * sep_temporal_f1024_e100_seed42
```

Kaggle private LB verificado: `0.67126`.

## Flujo del pipeline

```text
audios .wav
-> mono
-> resampling 44.1 kHz
-> MelSpectrogram
-> log/dB
-> crop/pad a 512 o 1024 frames
-> normalizacion por clip o global-mel
-> tres ramas e100
-> blend 0.25 / 0.375 / 0.375
-> validacion de formato
-> submission.csv
```

Por defecto:

```python
RUN_HEAVY_STEPS = False
```

Con ese valor, el notebook usa artefactos ya generados. Para reproducir caches y
entrenamientos desde cero, cambiar el flag a `True` en un entorno preparado para
entrenamientos largos.

## Regularizacion

El modelo final no depende de una unica tecnica de regularizacion. Las tres
ramas usan `head_dropout=0.3`; las capas convolucionales usan `BatchNorm`; y el
dataset de entrenamiento aplica augmentation tipo SpecAugment sobre el log-mel
con corrimiento temporal, mascara temporal y mascara de frecuencia.

Las dos ramas temporales usan `AdamW`, por lo que aplican `weight_decay=1e-4`
por defecto. La rama `separable_headsep` usa `Adam`; en el trainer actual ese
optimizador no recibe `weight_decay`, por lo que no hay L2 efectiva en esa rama.
No se usa `block_dropout`, `early_stopping`, time reverse ni contraste aleatorio
en la corrida final.

## Artefacto final

```text
100. Entregable/submission.csv
```

Debe coincidir con:

```text
investigation/results/submissions/parallel100_20260702/e100_headsep25_globalmel375_f1024_375.csv
```

SHA-256 esperado:

```text
4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab
```
