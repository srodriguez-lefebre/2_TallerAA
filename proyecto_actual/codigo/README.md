# Codigo de entrega

Notebook principal:

- `pipeline_final_taller_2.ipynb`

El notebook reconstruye el CSV final desde los artefactos oficiales del modelo
3-way e100 reponderado. Por defecto corre en modo liviano para validar formato,
componentes y blend sin reentrenar.

Para reproducir caches y entrenamiento desde cero, cambiar:

```python
RUN_HEAVY_STEPS = True
```

Ese modo usa los scripts de `investigation/scripts/`, entrena las tres ramas a
100 epocas y deja salidas bajo `proyecto_actual/codigo/outputs/`.

Artefacto generado por la corrida liviana:

- `submission.csv`

El CSV esperado tiene `3361` filas, `81` columnas y SHA-256:

```text
4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab
```
