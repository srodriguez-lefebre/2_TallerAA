# Proyecto actual - Entrega Taller 2

Esta carpeta concentra los artefactos pensados para entregar y defender el
Proyecto 2 de Freesound Audio Tagging.

## Estructura

- `codigo/`: notebook ejecutable del pipeline final.
- `informe/`: informe LaTeX del proyecto.
- `presentacion/`: presentacion HTML para la defensa.

La entrega se basa en el modelo final oficial:

```text
0.25  * separable_headsep_e100_seed42
+ 0.375 * globalmel_sep_temporal_e100_seed42
+ 0.375 * sep_temporal_f1024_e100_seed42
```

Score privado de Kaggle: `0.67126`.

El experimento `0.67674` se conserva como mejor score exploratorio con bagging,
pero no se selecciona como entrega principal.
