# Taller AA - Proyecto 2

Proyecto academico para Freesound Audio Tagging 2019.

El laboratorio historico vive en `investigation/`. La entrega nueva se organiza
en menos etapas, con notebooks y decisiones justificadas por la consigna, el
material del curso y la evidencia de los experimentos previos. La superficie
final de presentacion esta en `100. Entregable/`.

## Estructura de trabajo

- `01_analisis_datos/`: consigna, metrica, splits, clases, datos disponibles y
  primeros chequeos de audios.
- `02_preprocesamiento/`: representaciones de audio defendibles: STFT,
  log-mel, MFCC y caches reutilizables.
- `03_entrenamiento/`: baselines, CNNs, busqueda controlada de mejoras,
  validacion y comparacion contra `investigation/`.
- `04_final/`: auditoria de seleccion, pesos de blend elegidos y validacion de
  submission.
- `100. Entregable/`: notebook unico de entrega, con modo seguro para
  reconstruir el CSV final sin reentrenar modelos largos.
- `estado_entrega.md`: auditoria corta de requisitos, evidencia y validaciones
  ejecutadas.
- `decisiones_config_y_proceso.md`: explicacion consolidada de cada decision de
  configuracion/proceso, con evidencia y estado.

## Regla de alcance

Solo se publican submissions en el desafio publico de Kaggle Freesound Audio
Tagging 2019. El proyecto de Kaggle del curso queda fuera de esta estructura.

## Fuentes

- Consigna: `docs/proyecto_2_audio.pdf` y `docs/proyecto_2_freesound.pdf`.
- Curso: PDFs en `course/`.
- Evidencia experimental previa: `investigation/` y `docs/impact_analysis/`.
