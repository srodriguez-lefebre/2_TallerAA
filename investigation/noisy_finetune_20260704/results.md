# Noisy fine-tune results

Generated: 2026-07-04T09:03:37.324017+00:00

## Local validation

| Branch | Weight | Baseline lwlrap | Best lwlrap | Final lwlrap | Best epoch | Submission ckpt |
|---|---:|---:|---:|---:|---:|---|
| separable_headsep | 0.250 | 0.999569 | 0.999569 | 0.995336 | 0 | final |
| globalmel_sep_temporal | 0.375 | 0.998965 | 0.998965 | 0.995154 | 0 | final |
| sep_temporal_f1024 | 0.375 | 0.998850 | 0.998850 | 0.995542 | 0 | final |

Ensemble validation lwlrap: `0.998993`

## Submission

- Path: `/home/santig14/fing/taa/2_TallerAA/investigation/noisy_finetune_20260704/runs/ensemble/submission.csv`
- SHA256: `b29288caa6e7b37b29e830a29655decc7c6bc8110ca3a40b828f4dd2f5fabdcc`
- Rows: `3361`
- Labels: `80`

## Kaggle

El submit directo a Kaggle falla con `400 Bad Request` porque la competencia es
code-only. Se subio el CSV con el flujo dataset privado + kernel de copia:

```bash
.venv/bin/python investigation/scripts/kaggle_code_submission.py \
  --csv investigation/noisy_finetune_20260704/runs/ensemble/submission.csv \
  --slug fat19-noisy-finetune-20260704 \
  --title "FAT19 noisy finetune 20260704" \
  --message "noisy_finetune_20260704 final 3-way"
```

Resultado Kaggle:

| Date | Description | Status | Public | Private |
|---|---|---|---:|---:|
| 2026-07-04 13:42:32 | noisy_finetune_20260704 final 3-way | COMPLETE | 0.00000 | 0.68122 |

## Kaggle del curso

Referencia tomada del leaderboard de `TAA 2026 - Freesound Audio Tagging`.
Corresponde a una variante del dataset del curso, por lo que se debe reportar
separada del Kaggle publico `freesound-audio-tagging-2019` y no compararla como
si fuera exactamente la misma competencia.

| Leaderboard | Team | Entries | Score | Last |
|---|---|---:|---:|---|
| Public | Santiago Rodriguez | 1 | 0.64307 | 26m |

Lectura: el mismo CSV candidato tambien queda como referencia fuerte en el
Kaggle del curso, con score publico `0.64307`. Para la defensa conviene mostrar
ambos valores: `0.68122` como score privado en Freesound 2019 y `0.64307` como
referencia del curso sobre su variante de datos.
