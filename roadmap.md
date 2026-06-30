# Roadmap del Proyecto 2

## Objetivo

Construir una entrega final defendible para Freesound Audio Tagging 2019. La
meta no es copiar todo `investigation/`, sino promover las ideas que tienen:

- soporte en la consigna o en el material del curso;
- evidencia experimental clara;
- una explicacion tecnica razonable;
- una ruta de reproduccion o, si no existe completa, una limitacion documentada.

## Lectura de la consigna

El problema es de audio tagging multi-label. Cada audio puede tener una o mas de
80 clases. La metrica indicada por la consigna es `lwlrap`, que evalua el ranking
de clases positivas por muestra.

La consigna de audio justifica usar representaciones tiempo-frecuencia:

- STFT para senales no estacionarias;
- escala logaritmica de magnitudes;
- escala Mel;
- MFCC como alternativa clasica.

La consigna de Freesound agrega dos restricciones metodologicas importantes:

- hay subconjuntos de entrenamiento con fiabilidad distinta;
- la comparacion debe hacerse con `lwlrap`, no con accuracy.

## Estructura reducida

La estructura nueva reemplaza la proliferacion de carpetas numeradas del Taller 1
por cuatro bloques:

| Carpeta | Rol | Salida esperada |
|---|---|---|
| `01_analisis_datos/` | Entender datos, clases, splits y metrica | Notebook de EDA y conclusiones |
| `02_preprocesamiento/` | Preparar features/caches de audio | Notebook de log-mel y artefactos |
| `03_entrenamiento/` | Comparar modelos y mejoras defendibles | Notebooks de baseline, CNN y seleccion |
| `04_final/` | Armar el sistema final reproducible | Notebook/pipeline final y submission |

`investigation/` queda como laboratorio historico y fuente de artefactos. No se
borra ni se mezcla con la entrega ordenada.

`investigation/results/` es el registro vivo de experimentos nuevos. Antes de
promover cambios al proyecto final hay que revisar:

- `investigation/results/README.md`;
- `investigation/results/experiment_log.csv`;
- los detalles `investigation/results/<run_name>.md`;
- submissions candidatas en `investigation/results/submissions/`.

## Criterio para aceptar una mejora

Una mejora pasa a la entrega si cumple al menos una de estas condiciones:

1. La consigna la motiva directamente, por ejemplo log-mel o `lwlrap`.
2. El curso la cubre, por ejemplo CNN, dropout, BatchNorm, regularizacion,
   learning-rate scheduling, transfer learning, validacion o ensembles.
3. La evidencia local y/o Kaggle muestra una mejora clara contra un baseline
   comparable.
4. Aunque no mejore como modelo individual, agrega diversidad medible al blend.

Una mejora se documenta como descartada si:

- solo mejora un split local pero baja en leaderboard;
- depende de ajuste manual de leaderboard sin hipotesis tecnica;
- no tiene artefactos suficientes para reproducirla;
- mezcla datos ruidosos sin control y degrada validacion curated.

## Linea base defendible

1. Priors de clases: solo para validar formato de submission.
2. LogisticRegression One-vs-Rest sobre estadisticas log-mel: primer modelo que
   escucha audio real.
3. Regularizacion de LogisticRegression: mejora razonable por desbalance, pocas
   muestras limpias y muchas clases.
4. CNN sobre imagenes log-mel: preserva estructura tiempo-frecuencia.
5. Mejoras de entrenamiento neural: inicializacion He, BatchNorm, dropout,
   scheduler por plateau, early stopping y cabeza densa.
6. Full-train y seed ensemble: usar todo curated al final y reducir varianza.
7. Arquitecturas diversas y transfer: separable/residual y ResNet congelada solo
   si aportan al blend y se explican como diversidad.
8. Blend final: promedio ponderado de familias con errores distintos.

## Material del curso que respalda decisiones

- `TAA2026_clase12y13y14`: seleccion de hiperparametros, holdout/CV,
  regularizacion, BatchNorm, early stopping, transfer learning y learning rate.
- `TAA_clase16y18`: CNNs, arquitecturas profundas, data augmentation, dropout,
  ResNet, Xception/separable convolutions y transfer learning.
- Libro de Geron: multilabel classification, regularizacion, ensembles, CNNs y
  modelos preentrenados.

## Como trabajar

Cada notebook debe cerrar con:

- que pregunta responde;
- que baseline usa;
- que evidencia produce;
- si la decision queda `keep`, `discard`, `blend-only` o `needs-rerun`;
- comandos o artefactos necesarios para reproducir.

Ademas, cada notebook de entrenamiento o final debe indicar si uso evidencia de
`investigation/results/` y que decision tomo sobre esas corridas.

No hacer commits automaticamente. No publicar en el Kaggle del curso.
