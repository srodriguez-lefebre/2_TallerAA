# 100 - Entregable

Esta carpeta contiene el notebook final de entrega:

- `00_pipeline_entregable.ipynb`

El notebook documenta y deja codificado el recorrido completo:

```text
audios crudos
-> caches log-mel
-> entrenamiento de las 3 ramas finales
-> blend soft-voting
-> validacion de formato
-> CSV final
```

Por defecto el notebook esta en modo seguro:

```python
RUN_HEAVY_STEPS = False
```

Asi se puede revisar y ejecutar sin relanzar entrenamientos largos. Para
reproducir todo desde cero, cambiar ese flag a `True`.

La configuracion final defendible es:

```text
1/3 * separable_headsep
+ 1/3 * globalmel_sep_temporal
+ 1/3 * sep_temporal_f1024
```

Kaggle private LB verificado: `0.66649`.
