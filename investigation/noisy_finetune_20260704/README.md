# Noisy fine-tune desde ensamble E100

Experimento aislado para continuar el entrenamiento de las tres ramas del ensamble final
`3-way e100 reponderado 0.67126`, agregando `train_noisy` con menor peso y
augmentations de espectrograma.

## Hipotesis

El modelo final aprendio una representacion fuerte desde `train_curated`. La idea
es usar ese estado como punto de partida y agregar variedad desde `train_noisy`
sin dejar que las etiquetas ruidosas dominen el entrenamiento:

- batches 50% curated y 50% noisy;
- perdida noisy con peso 0.30;
- learning rate menor (`1e-4`) y scheduler coseno;
- 30 epocas adicionales;
- augmentations: time shift, time mask, frequency mask y gaussian noise.

## Ramas fuente

| Rama | Checkpoint fuente | Peso ensamble |
|---|---|---:|
| `separable_headsep` | `investigation/models/parallel100_20260702_separable_headsep_e100_seed42/small_logmel_cnn_best.pt` | 0.250 |
| `globalmel_sep_temporal` | `investigation/models/parallel100_20260702_globalmel_sep_temporal_e100_seed42/small_logmel_cnn_best.pt` | 0.375 |
| `sep_temporal_f1024` | `investigation/models/parallel100_20260702_sep_temporal_f1024_e100_seed42/small_logmel_cnn_best.pt` | 0.375 |

## Ejecucion

Chequeo estatico:

```bash
.venv/bin/python investigation/noisy_finetune_20260704/scripts/validate_experiment.py --static
```

Corrida completa:

```bash
bash investigation/noisy_finetune_20260704/scripts/run_experiment.sh
```

Validacion final:

```bash
.venv/bin/python investigation/noisy_finetune_20260704/scripts/validate_experiment.py
```

## Salidas

Las salidas pesadas quedan bajo `investigation/noisy_finetune_20260704/`:

- `source_states/`: copia de checkpoints/metadatos fuente y manifest con hashes.
- `data/*_f1024_x.npy`: caches memmap ignorados por Git para poder entrenar f1024 sin OOM.
- `runs/<branch>/models/`: checkpoint `best` y `final`.
- `runs/<branch>/submissions/`: submission del componente fine-tuned.
- `runs/<branch>/experiments/`: historia, metricas y predicciones de validacion.
- `runs/ensemble/submission.csv`: blend con los pesos originales.
- `results.md`: resumen local, hash de submission y estado Kaggle.

## Nota sobre metrica local

Los checkpoints fuente fueron entrenados con `full_train=True`, por lo que ya
vieron todo curated. Para medir si el fine-tuning degrada o mejora, este
experimento separa un 20% curated fijo durante la continuacion y reporta la
diferencia contra el checkpoint inicial en ese mismo split. Esta metrica sirve
como control de degradacion, no como estimacion completamente independiente.

La submission del experimento usa los checkpoints `final` tras 30 epocas, no el
`best` local, porque el objetivo era evaluar la continuacion completa del
entrenamiento. El `best` queda guardado solo como control.
