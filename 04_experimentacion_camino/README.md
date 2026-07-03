# 04 - Experimentacion camino

Objetivo: resumir el camino que llevo desde baselines simples hasta modelos
fuertes de audio, sin mezclar esta historia con la decision final.

Notebook principal:

- `01_camino_modelos_fuertes.ipynb`

Esta carpeta responde:

- que se probo para pasar de priors y modelos tabulares a CNNs log-mel;
- que mejoras hicieron crecer el sistema;
- por que `separable_headsep`, `globalmel_sep_temporal` y
  `sep_temporal_f1024` quedaron como bloques relevantes;
- que resultados historicos sirven como soporte, aunque no sean la entrega.

Conclusion esperada: el proyecto llega a modelos fuertes al preservar estructura
tiempo-frecuencia con log-mel, usar CNNs separables/residuales y agregar ramas
temporales con normalizacion o contexto distinto.
