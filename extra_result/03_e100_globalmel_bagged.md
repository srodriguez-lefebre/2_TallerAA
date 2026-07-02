# Resultado 3 - Bagging en globalmel

## Resumen

Este resultado mantiene los tres bloques conceptuales, pero promedia dos semillas
de la rama `globalmel`.

Formula expandida:

```text
0.25   * separable_headsep_e100_seed42
+ 0.1875 * globalmel_sep_temporal_e100_seed42
+ 0.1875 * globalmel_sep_temporal_e100_seed123
+ 0.375  * sep_temporal_f1024_e100_seed42
```

Formula conceptual:

```text
0.25  * headsep
+ 0.375 * globalmel_bagged
+ 0.375 * f1024
```

Kaggle private LB:

```text
0.67528
```

## Que significa bagged

`Bagged` significa que se promedian dos modelos de la misma familia entrenados
con distinta semilla.

En este caso:

```text
globalmel_bagged =
  0.5 * globalmel_seed42
+ 0.5 * globalmel_seed123
```

La arquitectura y los datos son los mismos. Cambia la inicializacion y el orden
aleatorio del entrenamiento. Eso produce errores distintos; al promediarlos se
reduce varianza.

## Artefacto

CSV:

```text
investigation/results/submissions/parallel100_20260702/e100_headsep25_globalmel_bag375_f1024_375.csv
```

SHA-256:

```text
5458c70ebcef2f45836ec63979a34d3e9d3f9bacb875d2c5a2c45755cbf88964
```

Shape:

```text
3361 x 81
```

## Interpretacion

Este fue un salto importante:

```text
e100 reponderado sin bagging: 0.67126
bagging en globalmel:         0.67528
mejora:                       +0.00402
```

Tambien se probo reemplazar `globalmel_seed42` por `globalmel_seed123`, y eso dio
`0.67051`. Por tanto, la mejora no viene de que la seed 123 sea mejor sola, sino
de promediar ambas semillas.
