# 02 - Preprocesamiento

Objetivo: convertir WAVs en representaciones reutilizables y justificadas.

Representaciones aceptadas para la entrega:

- estadisticas de log-mel para baseline tabular;
- imagenes log-mel de tamano fijo para CNNs;
- MFCC solo como comparacion si se documenta contra log-mel.

El material de consigna justifica STFT, log magnitudes, escala Mel y MFCC. El
curso justifica normalizacion, augmentations controladas y separar transformacion
de entrenamiento/validacion.

Notebooks:

- `00_C00_base_audio.ipynb`: base canonica de datos, labels, clases y archivos
  problematicos.
- `01_P00_logmel_stats_basic.ipynb`: stats log-mel basicas para baseline
  tabular.
- `02_P01_logmel_image_512.ipynb`: imagen log-mel `128 x 512` para CNN.
- `03_P02_globalmel_y_f1024.ipynb`: normalizacion global mel y ventana temporal
  `1024`, ideas que luego pasan a las ramas finales.
- `10_p00vsp01vsp02.ipynb`: comparacion consolidada de las variantes y su
  impacto en private LB. La seleccion final vigente se decide despues, en
  `06_experimentacion_final/` y `07_final/`.

Orden recomendado: correr `00`, luego `01`, `02`, `03` y finalmente `10`.

Artefactos generados por los notebooks:

- `results/C00_base_audio_summary.csv`
- `results/C00_curated_label_counts.csv`
- `results/C00_known_bad_files.csv`
- `results/P00_logmel_stats_basic_summary.csv`
- `results/P01_logmel_image_512_summary.csv`
- `results/P02_globalmel_f1024_summary.csv`
- `results/P02_globalmel_f1024_evidence.csv`
- `results/10_p00vsp01vsp02_comparison.csv`

Nota: las graficas de los notebooks usan `matplotlib`. En la venv local quedo
instalado para poder ejecutar y guardar los outputs.

Scripts historicos reutilizados desde `investigation/`:

- `scripts/build_feature_cache.py`
- `scripts/build_logmel_image_cache.py`
- `scripts/fat2019/features.py`
- `scripts/fat2019/spectrogram_images.py`
