# Resultado 4 - Bagging en globalmel y f1024

## Resumen

Este es el mejor resultado de la ronda. Mantiene el peso conceptual de tres
bloques, pero usa bagging por semillas en las dos ramas temporales.

Formula expandida:

```text
0.25   * separable_headsep_e100_seed42
+ 0.1875 * globalmel_sep_temporal_e100_seed42
+ 0.1875 * globalmel_sep_temporal_e100_seed123
+ 0.1875 * sep_temporal_f1024_e100_seed42
+ 0.1875 * sep_temporal_f1024_e100_seed123
```

Formula conceptual:

```text
0.25  * headsep
+ 0.375 * globalmel_bagged
+ 0.375 * f1024_bagged
```

Kaggle private LB:

```text
0.67674
```

## Artefacto

CSV:

```text
investigation/results/submissions/parallel100_20260702/e100_headsep25_globalmelbag375_f1024bag375.csv
```

SHA-256:

```text
7f645f4484980984bc3e7e7ea38940fa781bc1b8546579ad10f6e0c396999019
```

Shape:

```text
3361 x 81
```

## Interpretacion

Este resultado confirma que el bagging por semillas estaba aportando diversidad
real:

```text
bagging solo en globalmel:     0.67528
bagging en globalmel y f1024:  0.67674
mejora:                       +0.00146
```

Tambien se probo reemplazar `f1024_seed42` por `f1024_seed123`, y eso dio
`0.66735`. Igual que con `globalmel`, la seed nueva no era mejor por si sola. El
beneficio aparece al promediar semillas.

## Por que es defendible

Aunque la formula expandida tiene cinco CSV, conceptualmente no vuelve al blend
gigante anterior. Sigue teniendo tres bloques:

1. una rama CNN separable-residual (`headsep`);
2. una rama temporal con normalizacion global mel (`globalmel`);
3. una rama temporal con contexto largo de 1024 frames (`f1024`).

La diferencia es que los dos bloques temporales usan bagging interno de semillas
para reducir varianza.
