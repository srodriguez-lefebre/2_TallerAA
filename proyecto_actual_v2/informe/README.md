# Informe

Informe de la entrega v2. El foco narrativo es el noisy fine-tune final:

- partida desde el ensamble `3-way e100`;
- continuacion con `train_curated + train_noisy`;
- batches 50/50;
- menor peso para noisy;
- scheduler coseno y learning rate bajo;
- score final `0.68122` en Freesound 2019.

Compilar desde esta carpeta:

```bash
pdflatex informe.tex
pdflatex informe.tex
```

