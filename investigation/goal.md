# Goal: Freesound Audio Tagging 2019

## Objetivo

Investigar, probar ideas y obtener un lwlrap ≥ 0.3 (idealmente ≥ 0.4) en el leaderboard de Kaggle. El foco es experimentación rápida, no entregables académicos.

---

## Qué ES este goal

- Probar ideas, medir resultados, iterar
- Script de submission reutilizable para postular rápido
- Registro simple: idea → qué se probó → score obtenido
- Investigación web de soluciones existentes para inspirarse
- Todo en Python (scripts), no notebooks obligatorios

## Qué NO ES este goal

- No hay informe, presentación, ni pipeline final
- No hay una estructura rígida de fases obligatorias
- No hay validación cruzada obligatoria — se usa lo que sirva para medir rápido

---

## Estructura de trabajo

```
proyecto 2/
├── data/                    # datos extraídos del zip
├── scripts/                 # scripts de preprocesamiento, entrenamiento, etc.
│   └── submit.py            # script de postulación rápida y monótona
├── experiments/             # una carpeta o archivo por idea probada
│   └── resultados.md        # registro: idea → prueba → score
├── research/                # hallazgos de soluciones existentes
└── models/                  # modelos guardados
```

## Registro de experimentos

Formato simple en `resultados.md`:

```
| # | Idea | Qué se hizo | lwlrap valid | lwlrap LB | Notas |
|---|------|-------------|-------------|-----------|-------|
| 1 | Baseline CNN mel | CNN 3 capas sobre mel 128x128, solo curated | 0.25 | — | primera prueba |
| 2 | + noisy pretrain | pretrain en noisy, finetune curated | 0.35 | 0.32 | mejora clara |
```

## Script de submission (`submit.py`)

Responsabilidades:
- Recibe un modelo entrenado (o path a pesos) y genera `submission.csv`
- Carga audios de test, extrae features, predice, formatea CSV
- Validación básica del formato (80 columnas, nombres correctos, probabilidades)
- Listo para postular: `python submit.py --model models/exp_03.pt`

## Piezas necesarias para arrancar

1. **Extraer el zip** que ya está en el repo con los datos
2. **Script de features de audio** (mel spectrograms u otro) reutilizable
3. **Script de submit** reutilizable
4. **Primer modelo baseline** para tener un número de referencia

## Ideas a investigar y probar (no necesariamente en orden)

- CNN sobre mel spectrograms (baseline)
- Pretrained CNNs (ResNet, EfficientNet) sobre spectrograms como imágenes
- Pretrain en noisy + finetune en curated (patrón ganador)
- Data augmentation: mixup, SpecAugment, time shift, pitch shift
- Attention mechanisms
- Modelos que ya existan públicos adaptados
- Cualquier otra cosa que aparezca en la investigación web

## Criterio de "terminado"

| Aspecto | Mínimo | Bueno |
|---|---|---|
| **lwlrap en LB** | ≥ 0.30 | ≥ 0.40 |
| **Ideas probadas** | ≥ 3 con resultados registrados | ≥ 6 |
| **Script de submission** | Funcional y reutilizable | — |
| **Investigación** | Revisión de soluciones top | Técnicas de ahí probadas |
| **Registro** | Tabla con idea/prueba/score | — |

## Contexto técnico clave

- **Métrica**: lwlrap (ya implementada en `proyecto 2/a.ipynb`)
- **Dataset**: curated (4970 clips limpios) + noisy (19815 clips ruidosos), 80 clases, WAV 44.1kHz mono
- **Submission**: CSV con `fname` + probabilidad por cada una de las 80 clases
- **Leaderboard referencia**: 1ro = 0.76, oro ≈ 0.74, plata ≈ 0.72
- **Archivos corruptos conocidos**: `f76181c4.wav, 77b925c2.wav, 6a1f682a.wav, c7db12aa.wav, 7752cc8a.wav` (labels incorrectos), `1d44b0bd.wav` (sin señal)
- **Datos del zip**: `train_curated.csv`, `train_noisy.csv`, `sample_submission.csv`, carpetas de WAVs
- **1er lugar usó**: CNN con attention + SpecAugment + Mixup + ensemble
- **Patrón ganador común**: pretrain en noisy → finetune en curado
