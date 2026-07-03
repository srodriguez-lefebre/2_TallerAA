# Reproduccion pesada del 100 - 2026-07-03

Se ejecuto `100. Entregable/00_pipeline_entregable.ipynb` con
`RUN_HEAVY_STEPS=True`.

La corrida completo las tres ramas:

- `separable_headsep_e100`: 100 epocas, CSV generado.
- `globalmel_e100`: 100 epocas, CSV generado.
- `f1024_e100`: 100 epocas, CSV generado.

El notebook guardado contiene las salidas de esa ejecucion pesada y no muestra
errores.

## Resultado reproducido

El blend generado por la corrida pesada se preservo localmente en:

```text
100. Entregable/outputs/reproduccion_20260703/submission_reproducida.csv
```

SHA-256:

```text
8e6e914924764fbd3cac5c3802b6861ebc25dcc1a84d41a4d769f2b0d306d6d2
```

El CSV tiene formato valido de Kaggle: `3361` filas, `81` columnas, mismo orden
de `fname` que `sample_submission.csv` y probabilidades en `[0, 1]`.

## Artefacto oficial restaurado

Luego de guardar la reproduccion, se restauro:

```text
100. Entregable/submission.csv
```

al CSV oficial scoreado en Kaggle:

```text
investigation/results/submissions/parallel100_20260702/e100_headsep25_globalmel375_f1024_375.csv
```

SHA-256 oficial:

```text
4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab
```

Motivo: la configuracion pesada se reproduce y valida, pero la nueva corrida no
es bit a bit identica al artefacto ya scoreado. Para la entrega se conserva el
CSV oficial con private LB `0.67126`.
