# Codigo

`pipeline_final_taller_2_v2.ipynb` es el notebook de entrega para la version v2.
`pipeline_final_taller_2_v2_largo.ipynb` es la version extendida con todo el
codigo fuente embebido dentro del notebook.

El notebook esta organizado como pipeline autocontenido:

1. valida la estructura de datos crudos en `data/`;
2. define la configuracion final de las tres ramas;
3. construye los comandos para generar log-mel desde `.wav`;
4. entrena las ramas curated e100;
5. continua con fine-tuning curated + noisy;
6. recrea el ensamble y valida `submission.csv`;
7. escribe `pipeline_final_taller_2_v2_metadata.json`.

Para no relanzar horas de entrenamiento al abrir el notebook,
`RUN_FULL_PIPELINE` queda en `False`. Cambiarlo a `True` ejecuta el flujo completo
desde datos crudos hasta el CSV final. Los helpers usados por el notebook viven
en `pipeline_src/`, dentro de esta misma carpeta de codigo.

La version `*_largo` no depende de `pipeline_src/` en tiempo de ejecucion: guarda
las fuentes completas en una celda `EMBEDDED_FILES` y las materializa en
`work/_embedded_pipeline_src` al ejecutarse. Sirve para defensa o auditoria cuando
conviene que el notebook contenga absolutamente todo.

El CSV entregable es `submission.csv`, con SHA256:

```text
b29288caa6e7b37b29e830a29655decc7c6bc8110ca3a40b828f4dd2f5fabdcc
```
