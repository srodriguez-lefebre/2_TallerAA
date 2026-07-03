# Reorganizacion Entrega Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** reorganizar la entrega del Proyecto 2 para que las carpetas principales expliquen el modelo final `3-way e100 reponderado` y separen claramente las distintas capas de experimentacion.

**Architecture:** las carpetas `01` y `02` conservan el analisis y el preprocesamiento base. Las carpetas `03` a `07` se reconstruyen alrededor del modelo final oficial y las tres categorias de experimentacion. `100. Entregable` queda como pipeline final preparado, pero no se ejecuta en este entorno.

**Tech Stack:** Jupyter notebooks, Python, pandas, matplotlib, scripts existentes de `investigation/`, CSVs de submissions y validaciones por SHA/formato.

---

## Decision Final Fijada

El modelo final oficial para la defensa es:

```text
0.25  * separable_headsep_e100_seed42
+ 0.375 * globalmel_sep_temporal_e100_seed42
+ 0.375 * sep_temporal_f1024_e100_seed42
```

Score Kaggle private:

```text
0.67126
```

Artefacto esperado:

```text
investigation/results/submissions/parallel100_20260702/e100_headsep25_globalmel375_f1024_375.csv
```

SHA-256 esperado:

```text
4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab
```

El candidato `0.67674` queda documentado como mejor experimento con bagging, pero no como final oficial.

---

## Estructura Objetivo

El paquete de entrega debe quedar entendible copiando estas 8 carpetas:

```text
01_analisis_datos/
02_preprocesamiento/
03_entrenamiento/
04_experimentacion_camino/
05_experimentacion_general/
06_experimentacion_final/
07_final/
100. Entregable/
```

Cada carpeta debe tener:

- `README.md` con rol de la carpeta, notebook principal, entradas, salidas y conclusion.
- notebook principal con markdowns explicativos.
- tablas o figuras livianas cuando ayuden a entender la decision.
- referencias claras a `investigation/` cuando el artefacto pesado no se copie.

---

## Politica De Ejecucion De Notebooks

Todos los notebooks de `01` a `07` deben quedar ejecutados y guardados con outputs.

Excepcion explicita:

```text
100. Entregable/00_pipeline_entregable.ipynb
```

Ese notebook no se ejecuta ahora. Debe quedar preparado para ejecutarse luego en un entorno mas seguro.

---

## Validacion De Cierre

El trabajo solo se puede marcar como terminado si pasan estos puntos.

### 1. Estructura

- [ ] Existen las 8 carpetas objetivo.
- [ ] Cada carpeta tiene `README.md`.
- [ ] Cada carpeta de `01` a `07` tiene al menos un notebook principal.
- [ ] `100. Entregable` tiene notebook final de pipeline.

Comando sugerido:

```bash
find 01_analisis_datos 02_preprocesamiento 03_entrenamiento \
  04_experimentacion_camino 05_experimentacion_general \
  06_experimentacion_final 07_final "100. Entregable" \
  -maxdepth 1 -type f | sort
```

### 2. Consistencia Del Modelo Final

- [ ] `README.md` declara que la entrega final vive en `100. Entregable/`.
- [ ] `estado_entrega.md` declara `0.67126` como final oficial.
- [ ] `07_final/README.md` y su notebook declaran la misma formula.
- [ ] `100. Entregable/README.md` y su notebook declaran la misma formula.
- [ ] `0.67674` aparece solo como experimento con bagging, no como final oficial.
- [ ] `0.66649` aparece solo como referencia previa, no como final oficial.

Comandos sugeridos:

```bash
grep -RIn "0.67126" README.md estado_entrega.md 07_final "100. Entregable"
grep -RIn "0.67674" README.md estado_entrega.md 06_experimentacion_final 07_final "100. Entregable"
grep -RIn "0.66649" README.md estado_entrega.md 06_experimentacion_final 07_final "100. Entregable"
```

### 3. Artefacto Final

- [ ] `07_final/submission.csv` existe o se documenta explicitamente si se mantiene solo en `100. Entregable/submission.csv`.
- [ ] `100. Entregable/submission.csv` coincide con el CSV elegido de `parallel100_20260702`.
- [ ] El SHA del CSV final es `4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab`.
- [ ] El CSV final tiene `3361` filas y `81` columnas.
- [ ] Las columnas coinciden con `data/sample_submission.csv`.
- [ ] Las probabilidades estan en `[0, 1]`.

Comandos sugeridos:

```bash
sha256sum "100. Entregable/submission.csv" \
  investigation/results/submissions/parallel100_20260702/e100_headsep25_globalmel375_f1024_375.csv
```

```bash
python 07_final/validate_final_artifacts.py
```

### 4. Notebooks Ejecutados

- [ ] Notebooks de `01` a `07` tienen `execution_count` en todas sus celdas de codigo.
- [ ] Notebooks de `01` a `07` tienen outputs guardados cuando corresponde.
- [ ] `100. Entregable/00_pipeline_entregable.ipynb` puede quedar sin outputs.

Comando sugerido:

```bash
python scripts/check_delivery_notebooks_executed.py \
  --include 01_analisis_datos 02_preprocesamiento 03_entrenamiento \
            04_experimentacion_camino 05_experimentacion_general \
            06_experimentacion_final 07_final \
  --exclude "100. Entregable"
```

Si el script no existe al momento de implementar, crearlo como parte de la reorganizacion.

### 5. Markdown Explicativo

Cada notebook principal debe explicar:

- [ ] objetivo del notebook;
- [ ] entradas usadas;
- [ ] transformaciones o entrenamiento realizado;
- [ ] salidas generadas;
- [ ] conclusion de la etapa;
- [ ] como se conecta con el modelo final.

Temas que deben quedar repartidos desde `resumen_modelos_audio_taa.md`:

- [ ] `.wav -> mono -> 44.1 kHz`;
- [ ] MelSpectrogram y escala log/dB;
- [ ] crop/pad temporal a `512` o `1024` frames;
- [ ] normalizacion por clip;
- [ ] normalizacion global por banda mel;
- [ ] CNN separable residual;
- [ ] BiGRU temporal;
- [ ] soft voting multietiqueta;
- [ ] pesos finales `0.25 / 0.375 / 0.375`.

### 6. Separacion De Experimentacion

- [ ] `04_experimentacion_camino` explica como se llego a modelos fuertes.
- [ ] `05_experimentacion_general` concentra pruebas laterales, descartes y resultados que no entran en la narrativa principal.
- [ ] `06_experimentacion_final` compara solo variantes alrededor del modelo final:

```text
3-way original:                  0.66649
3-way e100 equal:                0.67055
3-way e100 weighted:             0.67126
globalmel bagged:                0.67528
globalmel + f1024 bagged:        0.67674
```

- [ ] `06_experimentacion_final` justifica por que se elige `0.67126` aunque `0.67674` tenga mejor score.

### 7. Entregable Preparado

`100. Entregable` debe contener el pipeline final preparado:

```text
wav
-> mono
-> 44.1 kHz
-> log-mel
-> crop/pad 512 o 1024
-> normalizacion clip/global-mel
-> predicciones de las tres ramas e100
-> blend 0.25 / 0.375 / 0.375
-> validacion
-> submission.csv
```

Validacion esperada para esta etapa:

- [ ] el notebook referencia los artefactos e100 correctos;
- [ ] el notebook no se ejecuta durante esta reorganizacion;
- [ ] el README explica como ejecutarlo despues en entorno seguro.

---

## Plan De Implementacion

### Task 1: Inventario Y Respaldo Conceptual

**Files:**
- Read: `README.md`
- Read: `estado_entrega.md`
- Read: `extra_result/README.md`
- Read: `extra_result/*.md`
- Read: `100. Entregable/componentes/*.md`
- Read: `07_final/submission.csv`
- Read: `investigation/results/submissions/parallel100_20260702/`
- Read: `investigation/results/experiment_log.csv`

- [ ] Verificar que el CSV `e100_headsep25_globalmel375_f1024_375.csv` existe.
- [ ] Verificar su SHA contra `4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab`.
- [ ] Registrar en notas de trabajo que `0.67126` es final oficial.
- [ ] Registrar que `0.67674` queda como experimento con bagging.

### Task 2: Crear Estructura Nueva

**Files:**
- Create: `04_experimentacion_camino/README.md`
- Create: `04_experimentacion_camino/01_camino_modelos_fuertes.ipynb`
- Create: `05_experimentacion_general/README.md`
- Create: `05_experimentacion_general/01_experimentos_generales.ipynb`
- Create: `06_experimentacion_final/README.md`
- Create: `06_experimentacion_final/01_experimentos_modelo_final.ipynb`
- Create: `07_final/README.md`
- Create: `07_final/01_seleccion_final.ipynb`
- Create: `07_final/validate_final_artifacts.py`

- [ ] Crear carpetas nuevas.
- [ ] Crear READMEs con objetivo, notebook principal y conclusion esperada.
- [ ] Crear notebooks con markdown inicial y celdas de carga de tablas/resumen.

### Task 3: Recentrar Entrenamiento

**Files:**
- Modify: `03_entrenamiento/README.md`
- Modify: `03_entrenamiento/01_baselines_y_modelos.ipynb`

- [ ] Reescribir la narrativa para enfocarla en las tres ramas finales.
- [ ] Explicar configuracion e100 y pesos del sistema final.
- [ ] Mover referencias largas a experimentos laterales hacia `04`, `05` o `06`.
- [ ] Ejecutar y guardar el notebook.

### Task 4: Separar Experimentacion

**Files:**
- Modify/Create notebooks de `04_experimentacion_camino/`, `05_experimentacion_general/`, `06_experimentacion_final/`

- [ ] `04` resume el camino hasta modelos fuertes.
- [ ] `05` resume pruebas descartadas o laterales.
- [ ] `06` compara variantes finales y justifica `0.67126`.
- [ ] Ejecutar y guardar notebooks `04`, `05`, `06`.

### Task 5: Rehacer Final

**Files:**
- Create/Modify: `07_final/README.md`
- Create/Modify: `07_final/01_seleccion_final.ipynb`
- Create/Modify: `07_final/validate_final_artifacts.py`
- Copy/Generate: `07_final/submission.csv` if final policy decides to keep a local copy.

- [ ] Declarar formula final.
- [ ] Declarar score `0.67126`.
- [ ] Validar SHA, shape, columnas y rango de probabilidades.
- [ ] Ejecutar y guardar notebook `07`.
- [ ] Ejecutar `python 07_final/validate_final_artifacts.py`.

### Task 6: Preparar 100 Entregable Sin Ejecutarlo

**Files:**
- Modify: `100. Entregable/README.md`
- Modify: `100. Entregable/00_pipeline_entregable.ipynb`
- Modify/Copy: `100. Entregable/submission.csv`

- [ ] Actualizar formula y score a `0.67126`.
- [ ] Actualizar referencias a artefactos e100.
- [ ] Incluir pipeline completo de procesamiento.
- [ ] No ejecutar el notebook.
- [ ] Copiar o generar `submission.csv` desde el CSV final elegido, verificando SHA.

### Task 7: Actualizar Documentacion Raiz

**Files:**
- Modify: `README.md`
- Modify: `estado_entrega.md`
- Modify: `decisiones_config_y_proceso.md` if still part of the delivery surface.

- [ ] Actualizar estructura de 8 carpetas.
- [ ] Actualizar modelo final y score.
- [ ] Explicar rol de las tres carpetas de experimentacion.
- [ ] Explicar excepcion de ejecucion para `100. Entregable`.

### Task 8: Validacion Final

**Files:**
- Create if missing: `scripts/check_delivery_notebooks_executed.py`
- Run: `python scripts/check_delivery_notebooks_executed.py ...`
- Run: `python 07_final/validate_final_artifacts.py`

- [ ] Verificar estructura.
- [ ] Verificar notebooks ejecutados de `01` a `07`.
- [ ] Verificar que `100. Entregable` no se haya ejecutado por accidente si la politica final pide dejarlo sin outputs.
- [ ] Verificar SHA final.
- [ ] Buscar contradicciones de score/final.
- [ ] Revisar `git status --short` y reportar archivos cambiados.

---

## Criterio De No Terminacion

No se considera terminado si ocurre cualquiera de estas condiciones:

- algun notebook de `01` a `07` queda sin ejecutar;
- `100. Entregable` se ejecuta en este entorno por accidente;
- `README.md`, `estado_entrega.md`, `07_final` y `100. Entregable` no coinciden en modelo final;
- `0.67674` queda presentado como final oficial;
- `0.66649` queda presentado como final vigente;
- falta README en alguna de las 8 carpetas;
- el CSV final no coincide con SHA `4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab`.

---

## Execution Handoff

Plan saved in `docs/plans/2026-07-02-reorganizacion-entrega-validacion.md`.

Preferred execution mode for this repo: inline execution in the current workspace, with checkpoints after each group of folders. No `git add`, `git commit`, `git push` or pull request unless explicitly requested.
