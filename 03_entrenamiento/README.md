# 03 - Entrenamiento

Objetivo: seleccionar modelos y mejoras con evidencia, no por prueba aleatoria.

Familias promovidas desde `investigation/`:

- priors solo como smoke test de submission;
- LogisticRegression One-vs-Rest sobre estadisticas log-mel;
- LogisticRegression regularizada (`C` bajo) si mejora contra baseline;
- CNN sobre imagenes log-mel con salida sigmoid y BCE;
- mejoras de entrenamiento neural: inicializacion He, BatchNorm, dropout,
  scheduler por plateau, early stopping y cabeza densa;
- full-train y seed ensemble para configuraciones ya elegidas;
- separable/residual y ResNet congelada solo como diversidad de blend.

No promover sin advertencia:

- calibracion por priors que rompa ranking;
- noisy data concatenado sin control;
- augmentations que bajan validacion;
- micro-ajustes de pesos elegidos solo por leaderboard.

Notebook:

- `01_baselines_y_modelos.ipynb`

Artefactos de decision:

- `decision_matrix.csv`: tabla legible por notebooks con evidencia, soporte del
  curso y decision.
- `training_results.csv`: corridas frescas e historicas con `valid_lwlrap`,
  artefactos y decision.
- `runs/`: artefactos livianos generados por corridas frescas de la entrega.

Documentacion humana consolidada:

- `decisiones_config_y_proceso.md` en la raiz explica las decisiones de proceso,
  configuracion y descarte.
- `investigation/results/` conserva el detalle vivo de experimentos nuevos.

## Registro vivo

Antes de agregar o promover una mejora nueva, revisar `investigation/results/`.
Esa carpeta concentra los experimentos que se vayan escribiendo a partir de
ahora:

- `experiment_log.csv`: tabla de corridas;
- `<run_name>.md`: detalle, comando, resultado y conclusion;
- `submissions/`: candidatos de submission.
