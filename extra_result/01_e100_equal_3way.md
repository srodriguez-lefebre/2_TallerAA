# Resultado 1 - Misma idea a 100 epocas

## Resumen

Este es el resultado que replica la idea del notebook/sistema simple, pero
entrenando los tres componentes durante 100 epocas.

```text
1/3 * separable_headsep_e100_seed42
+ 1/3 * globalmel_sep_temporal_e100_seed42
+ 1/3 * sep_temporal_f1024_e100_seed42
```

Kaggle private LB:

```text
0.67055
```

## Por que importa

Es el candidato mas limpio para explicar: son los mismos tres bloques del sistema
defendible, sin pesos especiales y sin componentes ocultos.

Comparacion directa:

```text
3-way original e40/e56: 0.66649
3-way a 100 epocas:    0.67055
mejora:                +0.00406
```

## Artefacto

CSV:

```text
investigation/results/submissions/parallel100_20260702/e100_headsep_globalmel_f1024_equal.csv
```

SHA-256:

```text
b68265422943a1484396dd4193746557b64203f5dd0baf3d2b085f5759422156
```

Shape:

```text
3361 x 81
```

## Interpretacion

El resultado sugiere que las tres ramas todavia se beneficiaban de mas
entrenamiento. No cambia la arquitectura ni agrega blending complejo; solo
extiende el entrenamiento a 100 epocas.

Para presentacion, esta es la version mas simple de defender.
