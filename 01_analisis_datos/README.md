# 01 - Analisis de datos

Objetivo: entender el problema antes de entrenar.

Este bloque cubre:

- consigna y metrica `lwlrap`;
- estructura de `train_curated.csv`, `train_noisy.csv` y `sample_submission.csv`;
- cantidad de clases y distribucion de etiquetas;
- audios disponibles, duraciones y archivos conocidos como problematicos;
- riesgos metodologicos: multi-label, desbalance, noisy labels y leaderboard.

Notebook principal:

- `01_analisis_datos.ipynb`

Artefactos generados:

- `results/dataset_split_summary.csv`
- `results/labels_per_audio_summary.csv`
- `results/label_count_table.csv`
- `results/label_cumulative_share.csv`
- `results/duration_summary.csv`
- `results/known_bad_files.csv`
- `results/figures/`: graficas usadas por el notebook.

La salida esperada es una lista corta de decisiones: que datos usar para
validacion, que metrica reportar y que riesgos no se deben esconder en el
informe.
