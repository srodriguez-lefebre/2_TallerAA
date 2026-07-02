# Resultados extra seleccionados

Esta carpeta resume los cuatro resultados que conviene conservar como candidatos
fuertes y defendibles. Todos salen de la misma linea experimental
`parallel100_20260702`: reentrenar los componentes del sistema simple durante
100 epocas y evaluar los CSV en Kaggle.

## Punto de partida

Antes de esta ronda, el sistema simple defendible era:

```text
1/3 * separable_headsep
+ 1/3 * globalmel_sep_temporal
+ 1/3 * sep_temporal_f1024
```

con private LB `0.66649`.

La ronda nueva mantiene la misma idea conceptual: modelos sobre espectrogramas
log-mel, sin usar `current` ni esconder el blend gigante anterior.

## Componentes base

| componente | descripcion corta | CSV |
| --- | --- | --- |
| `H` | `separable_headsep_e100_seed42`: CNN separable-residual con cabeza densa | `investigation/submissions/parallel100_20260702_separable_headsep_e100_seed42/small_logmel_cnn.csv` |
| `G42` | `globalmel_sep_temporal_e100_seed42`: rama temporal BiGRU con normalizacion global mel | `investigation/submissions/parallel100_20260702_globalmel_sep_temporal_e100_seed42/small_logmel_cnn.csv` |
| `G123` | misma rama `globalmel`, entrenada con seed 123 | `investigation/submissions/parallel100_20260702_globalmel_sep_temporal_e100_seed123/small_logmel_cnn.csv` |
| `F42` | `sep_temporal_f1024_e100_seed42`: rama temporal BiGRU con contexto largo 1024 frames | `investigation/submissions/parallel100_20260702_sep_temporal_f1024_e100_seed42/small_logmel_cnn.csv` |
| `F123` | misma rama `f1024`, entrenada con seed 123 | `investigation/submissions/parallel100_20260702_sep_temporal_f1024_e100_seed123/small_logmel_cnn.csv` |

## Los cuatro resultados elegidos

| archivo | formula | Kaggle private LB |
| --- | --- | ---: |
| `01_e100_equal_3way.md` | `1/3 H + 1/3 G42 + 1/3 F42` | `0.67055` |
| `02_e100_weighted_3way.md` | `0.25 H + 0.375 G42 + 0.375 F42` | `0.67126` |
| `03_e100_globalmel_bagged.md` | `0.25 H + 0.1875 G42 + 0.1875 G123 + 0.375 F42` | `0.67528` |
| `04_e100_globalmel_f1024_bagged.md` | `0.25 H + 0.1875 G42 + 0.1875 G123 + 0.1875 F42 + 0.1875 F123` | `0.67674` |

## Lectura general

1. Entrenar a 100 epocas la misma arquitectura simple ya mejora el resultado:
   `0.66649 -> 0.67055`.
2. Reponderar de forma gruesa mejora un poco mas: las ramas temporales pesan mas
   que `headsep`.
3. El bagging por semillas ayuda mucho cuando se aplica dentro de una rama
   conceptual. No es que `seed123` sea mejor sola; de hecho los reemplazos solos
   no mejoraron. El beneficio viene de promediar errores distintos.
4. El mejor candidato actual es el doble bagging temporal con private LB
   `0.67674`.

## Donde quedo registrado

- Resultados principales: `investigation/results/simple_defensible_blends_2026_07_01.md`
- Log tabular: `investigation/results/experiment_log.csv`
- CSVs generados: `investigation/results/submissions/parallel100_20260702/`
