# 03 - Entrenamiento

Objetivo: explicar el entrenamiento del modelo final, no todo el historial de
experimentos.

Notebook principal:

- `01_baselines_y_modelos.ipynb`

Modelo final oficial:

```text
0.25  * separable_headsep_e100_seed42
+ 0.375 * globalmel_sep_temporal_e100_seed42
+ 0.375 * sep_temporal_f1024_e100_seed42
```

Kaggle private LB: `0.67126`.

## Las tres ramas

| Rama | Entrada | Normalizacion | Arquitectura | Rol |
|---|---|---|---|---|
| `separable_headsep_e100_seed42` | log-mel `128 x 512` | por clip | CNN separable-residual + cabeza densa | patrones locales tiempo-frecuencia |
| `globalmel_sep_temporal_e100_seed42` | log-mel `128 x 512` | global por banda mel | CNN separable-residual + BiGRU | escala estable y evolucion temporal |
| `sep_temporal_f1024_e100_seed42` | log-mel `128 x 1024` | por clip | CNN separable-residual + BiGRU | contexto temporal largo |

## Por que 100 epocas

La ronda `parallel100_20260702` mostro que la misma idea simple seguia
mejorando al entrenar mas:

```text
3-way original        0.66649
3-way e100 equal      0.67055
3-way e100 weighted   0.67126
```

La reponderacion final da menos peso a la rama puramente convolucional y mas a
las dos ramas temporales.

## Regularizacion del modelo final

Las tres ramas usan `head_dropout=0.3`. En `separable_headsep` ese dropout esta
en la cabeza densa de 256 unidades; en `globalmel_sep_temporal` y
`sep_temporal_f1024` esta despues del pooling de la BiGRU temporal.

Durante entrenamiento se aplica augmentation sobre el log-mel solo en train:
desplazamiento temporal aleatorio, mascara temporal y mascara en frecuencia.
Las opciones de invertir el tiempo y cambiar contraste existen en el script,
pero en la corrida final quedaron apagadas (`0.0`).

La regularizacion por peso depende del optimizador: las dos ramas temporales
usan `AdamW`, por lo tanto toman el `weight_decay` por defecto `1e-4`; la rama
`separable_headsep` usa `Adam`, y en el trainer actual `Adam` no aplica
`weight_decay`. `block_dropout` y `early_stopping` existen como opciones, pero
en la corrida final estan desactivados.

## Que queda fuera de este notebook

Los experimentos que no forman parte del entrenamiento final se separan en:

- `04_experimentacion_camino/`
- `05_experimentacion_general/`
- `06_experimentacion_final/`
