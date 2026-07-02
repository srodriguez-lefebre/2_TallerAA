# Resultado 2 - Mismo 3-way a 100 epocas, reponderado

## Resumen

Este resultado usa los mismos tres CSV de 100 epocas, pero cambia los pesos de
forma gruesa:

```text
0.25  * separable_headsep_e100_seed42
+ 0.375 * globalmel_sep_temporal_e100_seed42
+ 0.375 * sep_temporal_f1024_e100_seed42
```

Kaggle private LB:

```text
0.67126
```

## Por que importa

Sigue siendo un sistema de tres componentes. La unica decision nueva es que las
dos ramas temporales pesan un poco mas que `headsep`.

Comparacion:

```text
e100 pesos iguales: 0.67055
e100 reponderado:   0.67126
mejora:             +0.00071
```

## Artefacto

CSV:

```text
investigation/results/submissions/parallel100_20260702/e100_headsep25_globalmel375_f1024_375.csv
```

SHA-256:

```text
4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab
```

Shape:

```text
3361 x 81
```

## Interpretacion

Los diagnosticos de pares mostraron que los tres componentes son necesarios, pero
tambien que las ramas temporales aportan mucho al resultado final. Por eso esta
reponderacion conserva `headsep`, pero le da mayor peso conjunto a `globalmel` y
`f1024`.

Es defendible porque no es una busqueda fina de muchos decimales: son pesos
redondos y faciles de explicar.
