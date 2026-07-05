# Proyecto actual v2 - Freesound Audio Tagging

Nueva version del paquete de entrega centrada en el experimento
`noisy_finetune_20260704`.

El punto final ya no es el ensamble `3-way e100` con score privado `0.67126`,
sino la continuacion controlada con `train_noisy`, que obtuvo:

- Freesound Audio Tagging 2019 private score: `0.68122`.
- Kaggle del curso public score: `0.64307`.

## Estructura

- `codigo/`: notebook ejecutado, CSV final y metadatos.
- `informe/`: informe LaTeX y PDF.
- `presentacion/`: presentacion HTML.
- `validar_entrega.py`: chequeo integral del paquete.

El notebook principal de `codigo/` contiene el pipeline reproducible desde datos
crudos y usa helpers locales en `codigo/pipeline_src/`.

Tambien se incluye `codigo/pipeline_final_taller_2_v2_largo.ipynb`, una version
extendida que embebe todo el codigo fuente dentro del propio notebook. Esa
version sirve para defensa o auditoria cuando se quiere mostrar el pipeline sin
abrir archivos auxiliares.

`investigation/` queda como historial de trabajo, no como fuente necesaria para
ejecutar el pipeline final.
