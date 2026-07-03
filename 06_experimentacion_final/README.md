# 06 - Experimentacion final

Objetivo: comparar solo las variantes hechas cuando el modelo final ya estaba
definido conceptualmente como un 3-way log-mel.

Notebook principal:

- `01_experimentos_modelo_final.ipynb`

La secuencia central es:

```text
3-way original                  0.66649
3-way e100 equal                0.67055
3-way e100 weighted             0.67126  <- final oficial
globalmel bagged                0.67528
globalmel + f1024 bagged        0.67674
```

Decision: la entrega oficial usa `0.67126` porque conserva tres componentes
fisicos, pesos simples y una explicacion directa. El `0.67674` queda como mejor
experimento con bagging, pero no como final oficial.
