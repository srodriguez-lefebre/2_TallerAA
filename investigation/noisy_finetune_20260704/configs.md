# Configuracion del experimento

## Config comun

| Parametro | Valor |
|---|---:|
| Epocas adicionales | 30 |
| Learning rate | 0.0001 |
| Scheduler | cosine |
| LR minimo | 0.000001 |
| Curated loss weight | 1.00 |
| Noisy loss weight | 0.30 |
| Batch composition | 50% curated / 50% noisy |
| Validation split | 20% curated, seed 42 |
| Pos weight | Calculado solo con curated train |
| Gaussian noise std | 0.015 |
| Time shift | hasta 32 frames |
| Time mask | ancho 8 a 48 frames |
| Frequency mask | ancho 4 a 16 bandas mel |

## Ramas

| Branch | Frames | Cache tag | Batch size | Arquitectura | Activacion | Optimizer | Head |
|---|---:|---|---:|---|---|---|---|
| `separable_headsep` | 512 | none | 24 | `separable_residual` | `relu` | `adam` | hidden 256, dropout 0.30 |
| `globalmel_sep_temporal` | 512 | `globalmel` | 24 | `separable_temporal_bigru` | `silu` | `adamw` | dropout 0.30 |
| `sep_temporal_f1024` | 1024 | none | 12 | `separable_temporal_bigru` | `silu` | `adamw` | dropout 0.30 |

## Decision de mezcla

Se mantiene el mismo peso de ensamble que dio el score `0.67126`:

```text
0.25  * separable_headsep
0.375 * globalmel_sep_temporal
0.375 * sep_temporal_f1024
```

La decision clave es que `train_noisy` no se agrega como si tuviera la misma
confiabilidad que `train_curated`: aparece en la mitad de cada batch para aportar
diversidad, pero cada fila noisy pesa 0.30 en la perdida.

