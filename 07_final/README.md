# 07 - Final

Objetivo: fijar la seleccion final de la entrega y validar que el CSV cumple el
formato de Kaggle.

Notebook principal:

- `01_seleccion_final.ipynb`

Validador:

- `validate_final_artifacts.py`
- `validate_final_config.py`

Modelo final oficial:

```text
0.25  * separable_headsep_e100_seed42
+ 0.375 * globalmel_sep_temporal_e100_seed42
+ 0.375 * sep_temporal_f1024_e100_seed42
```

Kaggle private LB: `0.67126`.

Artefacto final:

```text
07_final/submission.csv
100. Entregable/submission.csv
```

Ambos deben coincidir con:

```text
investigation/results/submissions/parallel100_20260702/e100_headsep25_globalmel375_f1024_375.csv
```

SHA-256 esperado:

```text
4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab
```

El candidato `0.67674` se documenta como mejor experimento con bagging, pero no
se selecciona como entrega porque usa bagging interno en dos ramas temporales.

## Validacion de configuracion

`validate_final_config.py` compara el pipeline de `100. Entregable` con los
metadatos reales de las tres corridas `parallel100_20260702_*_e100_seed42`.
Tambien recompone el blend `0.25 / 0.375 / 0.375` desde los tres CSV de
componentes y verifica que sea exactamente el mismo archivo final seleccionado.
