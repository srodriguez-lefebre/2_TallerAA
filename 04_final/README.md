# 04 - Final

Objetivo: reunir la mejor configuracion defendible en un pipeline final.

El pipeline final debe:

- usar solo artefactos seleccionados por evidencia;
- dejar clara la diferencia entre validacion local, full-train y Kaggle;
- validar formato de `submission.csv`;
- registrar pesos del blend y razon de cada rama;
- documentar limitaciones de reproducibilidad si alguna rama viene de
  `investigation/` sin comando exacto.

Notebook principal:

- `01_pipeline_final.ipynb`

Artefactos de seleccion:

- `final_pipeline_manifest.csv`: componentes, rol, evidencia, artefacto y
  estado.
- `submission_candidates.csv`: candidates finales con hashes, score disponible y
  decision.
- `final_pipeline_metadata.json`: metadata de la submission seleccionada,
  evidencia Kaggle y criterio de seleccion.
- `final_selection.md`: narrativa humana corta de la seleccion final.
- `submission.csv`: CSV final elegido y validado contra `sample_submission.csv`.
- `validate_final_artifacts.py`: validacion por CLI de manifest y submissions
  candidatas.

Validacion rapida:

```bash
python 04_final/validate_final_artifacts.py
```

Fuentes de verdad:

- `submission.csv` es el archivo a entregar.
- `final_pipeline_metadata.json` guarda score, fecha, descripcion y SHA del
  candidato seleccionado.
- `submission_candidates.csv` lista alternativas y deja claro que fue superado.
- `final_selection.md` explica la seleccion en texto humano.
