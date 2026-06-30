# Impact drivers - Freesound Audio Tagging 2019

Este documento explica los cambios que mas movieron el resultado del sistema.
Complementa el registro cronologico de `experiments/resultados.md` y el analisis
tabular de `experiments/impact_analysis/`.

El objetivo no es listar todos los experimentos. El objetivo es responder:

- que cambio produjo cada salto grande;
- contra que baseline se mide;
- por que ese cambio tenia sentido tecnico;
- que evidencia lo confirmo;
- que limitaciones o falsos positivos dejo.

## Como leer los impactos

Los deltas de leaderboard usan el `privateScore` de Kaggle, porque ese fue el
score que se uso para comparar submissions finales. Cuando se menciona
`valid_lwlrap`, es la validacion local disponible para seleccionar candidatos,
normalmente un holdout random 80/20 sobre `train_curated.csv` con seed 42.

No todos los deltas locales son comparables entre si. Los modelos `full-train`
no tienen holdout porque usan todo curated. En esos casos el impacto se mide por
leaderboard o por el experimento de holdout que eligio la configuracion.

Tambien hay que separar dos tipos de impacto:

- **impacto individual**: el modelo por si solo mejora;
- **impacto de ensemble**: el modelo no es el mejor individualmente, pero suma
  porque sus errores son distintos.

Esta distincion es clave para entender por que modelos debiles, como ResNet50
ImageNet congelada, terminaron aportando al blend final.

## Resumen de saltos confirmados

| Rank | Cambio | Baseline | Resultado | Delta private LB | Lectura corta |
|------|--------|----------|-----------|------------------|---------------|
| 1 | Priors -> sklearn log-mel | #0 `0.03749` | #1 `0.32714` | `+0.28965` | Primer modelo que usa audio real |
| 2 | Sklearn stats -> CNN log-mel | #5 `0.37607` | #13 `0.52257` | `+0.14650` | El espectrograma como imagen captura estructura temporal/frecuencial |
| 3 | Fashion-MNIST transfer local | #30 `0.58612` | #44 `0.64665` | `+0.06053` | Mejor optimizacion, cabeza densa y diversidad neural |
| 4 | Regularizacion sklearn C=0.01 | #1 `0.32714` | #5 `0.37607` | `+0.04893` | Menos sobreajuste en un baseline barato |
| 5 | Blend CNN + sklearn | #13 `0.52257` | #17 `0.56682` | `+0.04425` | Errores complementarios entre stats y CNN |
| 6 | Full-train + dos seeds CNN | #17 `0.56682` | #30 `0.58612` | `+0.01930` | Mas datos para entrenar y menor varianza |
| 7 | Gatos/perros transfer local | #44 `0.64665` | #51 `0.65289` | `+0.00624` | Mas diversidad, pero con retornos marginales |

## 1. Priors -> sklearn log-mel

**Experimentos:** #0 -> #1  
**Categoria:** features  
**Private LB:** `0.03749 -> 0.32714`  
**Delta:** `+0.28965`

El experimento #0 era un smoke-test: para cada clase escribia una probabilidad
constante basada en la frecuencia de esa clase en train. Eso valida el formato
de la submission y el flujo de Kaggle, pero no escucha el audio. Por eso su
score era muy bajo.

El experimento #1 fue el primer modelo util. Para cada WAV se extrajeron
estadisticas del log-mel por banda: media, desviacion, maximo, percentil 75,
duracion, RMS y zero-crossing rate. Luego se entreno una LogisticRegression
One-vs-Rest con `class_weight="balanced"`.

El salto fue grande porque el sistema paso de predecir solo popularidad de
clases a usar evidencia acustica. Aunque las features eran simples, ya
codificaban energia por bandas, textura espectral, duracion y actividad del
waveform. Eso alcanza para separar clases obvias como ladridos, instrumentos,
golpes, habla o agua mucho mejor que un prior global.

La eleccion de LogisticRegression tambien fue razonable para este punto:
entrena rapido, tolera features densas, permite multilabel con One-vs-Rest y da
probabilidades directamente compatibles con el CSV de Kaggle.

**Por que fue importante:** establecio el primer baseline competitivo y
reutilizable. Tambien dejo funcionando la extraccion de features, la metrica
local y el pipeline de submission.

**Limitacion:** las estadisticas colapsan el eje temporal. Dos audios con eventos
en orden distinto pueden parecer similares si sus resumenes por banda coinciden.
Ese techo explica por que el siguiente salto vino al preservar la forma 2D del
espectrograma.

## 2. Sklearn stats -> CNN log-mel

**Experimentos:** #5 -> #13  
**Categoria:** arquitectura / representacion  
**Private LB:** `0.37607 -> 0.52257`  
**Delta:** `+0.14650`

El experimento #5 era el mejor baseline estadistico: LogisticRegression con
`C=0.01` sobre log-mel stats. El #13 cambio la representacion y el modelo:
cacheo espectrogramas log-mel `128x512` como imagenes y entreno una CNN chica
multilabel.

Este fue el mayor salto tecnico aislado despues del baseline inicial. La razon
principal es que la CNN conserva la estructura tiempo-frecuencia. En audio, no
solo importa cuanta energia hay en una banda; importa cuando aparece, cuanto
dura, si se repite, si sube o baja en frecuencia y si convive con otros eventos.
Una CNN puede detectar patrones locales en el espectrograma, mientras que las
estadisticas globales pierden esa informacion.

El entrenamiento tambien estaba mejor alineado con el problema: una salida
sigmoid por clase, `BCEWithLogitsLoss`, pesos positivos por desbalance y
augmentations tipo SpecAugment simples. Eso encaja mejor con una tarea
multilabel que un clasificador lineal sobre resumenes globales.

**Por que fue importante:** confirmo que el proyecto debia moverse hacia
espectrogramas como imagen. A partir de este punto, las mejoras grandes vinieron
de arquitectura, optimizacion y ensembles neuronales, no de agregar estadisticas
tabulares.

**Limitacion:** aumento mucho el costo de entrenamiento y creo dependencia de
GPU/PyTorch. Tambien hizo mas importante guardar metadata, checkpoints y
submissions para poder reproducir variantes.

## 3. Fashion-MNIST transfer local

**Experimentos:** #30 -> #44  
**Categoria:** optimizacion/training + arquitectura + blend  
**Private LB:** `0.58612 -> 0.64665`  
**Delta:** `+0.06053`

Esta etapa tomo ideas del notebook de Fashion-MNIST y las adapto al clasificador
de espectrogramas. No fue un solo cambio: fue una secuencia de mejoras de
optimizacion, cabeza del modelo y diversidad en ensemble.

Los cambios relevantes fueron:

- inicializacion He para capas convolucionales y capas densas relevantes;
- `ReduceLROnPlateau` para bajar el learning rate cuando la validacion se
  estancaba;
- cabeza densa de 256 unidades con BatchNorm, activacion y dropout;
- variantes con ReLU y una version literal Adam/ReLU;
- full-train con dos semillas y blend fino contra sklearn.

La evidencia local mostro que la CNN anterior no estaba limitada solo por datos,
sino tambien por optimizacion y capacidad de la cabeza. El experimento #31 subio
la validacion local de `0.6528` a `0.7528` con He + plateau. El #34 llevo la
cabeza densa 256 a `0.7838`, y el ensemble local head256/ReLU llego a `0.7946`.
Luego esas configuraciones se reentrenaron full curated y se mezclaron hasta el
mejor #44.

**Por que funciono:** la CNN ya tenia una buena representacion, pero necesitaba
entrenar de forma mas estable y tener una cabeza con capacidad suficiente para
combinar los canales finales. El scheduler por plateau ayudo a no quedarse con
un LR demasiado alto; la cabeza 256 agrego no linealidad util; ReLU/literal
aportaron modelos parecidos pero no identicos.

**Por que fue importante:** fue el ultimo salto grande del proyecto. Paso de una
buena familia CNN inicial a un sistema neural bastante mas fuerte y estable.

**Limitacion:** no todo lo que mejoro localmente mejoro en LB. Algunas variantes
valieron mas como diversidad que como modelo individual. Por eso esta etapa debe
verse como mejora acumulada de training + arquitectura + blend, no como un unico
parametro magico.

## 4. Regularizacion sklearn C=0.01

**Experimentos:** #1 -> #5  
**Categoria:** optimizacion/training  
**Private LB:** `0.32714 -> 0.37607`  
**Delta:** `+0.04893`

Despues del primer sklearn, se barrio la regularizacion de LogisticRegression.
El mejor punto fue `C=0.01`. En scikit-learn, bajar `C` aumenta la
regularizacion. El resultado fue una mejora consistente desde #1 hasta #5.

Esto funciono porque el baseline tenia relativamente pocas filas curated y
muchas clases, con fuerte desbalance. Un modelo lineal con regularizacion debil
puede ajustar demasiado el split local o magnificar ruido de clases raras. Al
bajar `C`, las fronteras se suavizan y las probabilidades quedan menos
extremas.

La mejora fue barata: no requirio GPU, no cambio features y mantuvo el mismo
pipeline. Por eso fue una buena inversion temprana.

**Por que fue importante:** dejo un baseline tabular fuerte que siguio aportando
en blends incluso despues de que la CNN fuera mucho mejor individualmente.

**Limitacion:** seguir afinando `C` no dio saltos relevantes. #12 mejoro apenas
algunas metricas locales/OOF, pero no mejoro el leaderboard. Eso marco que el
techo del enfoque tabular ya estaba cerca.

## 5. Blend CNN + sklearn

**Experimentos:** #13 -> #17  
**Categoria:** blend  
**Private LB:** `0.52257 -> 0.56682`  
**Delta:** `+0.04425`

El primer blend fuerte mezclo la CNN log-mel con el sklearn `C=0.01`. Aunque la
CNN era claramente mejor individualmente, el sklearn seguia agregando senal.
El mejor peso subido en esa etapa fue aproximadamente 65% CNN y 35% sklearn.

La razon tecnica es complementariedad. La CNN aprende patrones locales del
espectrograma. El sklearn usa estadisticas globales por banda, duracion, RMS y
ZCR. Esas features son mas simples, pero capturan propiedades del clip completo
que una CNN chica puede no ponderar igual.

La evidencia posterior de correlaciones refuerza esta lectura: `sklearn C=0.01`
y `head256` tuvieron correlacion Pearson `0.3865` en holdout, muy baja para dos
modelos del mismo problema. Baja correlacion significa que sus errores no son
los mismos, y eso es exactamente lo que un promedio necesita para mejorar.

**Por que fue importante:** mostro que el camino no era elegir "sklearn o CNN",
sino mantener familias distintas y mezclarlas. Ese patron se repitio en las
etapas posteriores.

**Limitacion:** los pesos optimos locales y de LB no siempre coincidieron. La
validacion local favorecia algunos blends que luego no generalizaban igual. Por
eso los pesos de blend terminaron siendo una mezcla de evidencia local y
feedback de Kaggle.

## 6. Full-train + dos seeds CNN

**Experimentos:** #17 -> #30  
**Categoria:** full-train + seed ensemble  
**Private LB:** `0.56682 -> 0.58612`  
**Delta:** `+0.01930`

Una vez elegida una CNN razonable con holdout, se reentreno usando todas las
filas curated. Eso elimina la validacion local durante el entrenamiento, pero
permite que el modelo use los ejemplos que antes quedaban reservados.

Tambien se promediaron dos semillas. En redes neuronales, cambiar seed cambia
inicializacion, orden de batches y detalles de entrenamiento. Dos modelos con
arquitectura identica pueden terminar en soluciones parecidas pero no iguales.
Promediarlos suele reducir varianza y mejorar calibracion.

En este proyecto, el full-train individual subio la rama CNN, y el promedio de
dos seeds mejoro el blend con sklearn. El ajuste fino de pesos llevo al #30.

**Por que funciono:** el dataset curated era chico. Usar todas las filas aumenta
la informacion disponible para la red. El ensemble de seeds estabiliza
predicciones y reduce errores accidentales de una corrida particular.

**Limitacion:** no todo seed extra ayudo. El tercer seed probado en #29 bajo el
LB frente al blend de dos seeds. Eso sugiere que agregar modelos similares tiene
retornos decrecientes si no aportan diversidad real.

## 7. Gatos/perros transfer local

**Experimentos:** #44 -> #51  
**Categoria:** arquitectura + transferencia + blend  
**Private LB:** `0.64665 -> 0.65289`  
**Delta acumulado:** `+0.00624`  
**Delta final contra #48:** `+0.00051`

Esta etapa tomo ideas del notebook de gatos/perros: convoluciones separables,
residuales con proyeccion, global average pooling, cabeza densa compacta,
augmentations evaluadas individualmente y transferencia con ResNet50 ImageNet
congelada.

Hubo tres aportes principales:

- **separable residual:** subio fuerte la validacion local individual
  (`0.80215`) y agrego una arquitectura distinta;
- **ResNet50 congelada:** fue debil individualmente (`0.66976` local), pero
  tuvo baja correlacion con otras ramas, por lo que sumo diversidad;
- **headsep:** combino separable residual con cabeza densa 256 y logro el mejor
  modelo individual local (`0.80781`).

El punto importante es que esta etapa no produjo otro salto grande como
Fashion-MNIST. El mejor blend #48 ya habia llevado el sistema a `0.65238`. El
experimento final #51 mejoro a `0.65289`, o sea `+0.00051` incremental contra el
mejor catsdogs previo. El `+0.00624` es correcto solo si se mide toda la etapa
contra el cierre anterior #44.

**Por que funciono:** las nuevas ramas aportaron diversidad al ensemble. ResNet50
no entendia el audio tan bien como las CNNs entrenadas para log-mel, pero sus
errores eran distintos. El modelo headsep mejoro localmente y entro con peso
moderado en el blend final.

**Limitacion:** fue una mejora marginal. A esa altura el sistema ya estaba en
zona de retornos decrecientes. Seguir moviendo pesos de modelos parecidos
probablemente aporte poco comparado con incorporar una fuente nueva de
informacion, como pretraining real en AudioSet, noisy data controlado, MixUp
sistematico u OOF mas fuerte para calibrar blends.

## Cambios que explican falsos positivos

Algunos experimentos mejoraron una metrica local o parecian razonables, pero no
movieron el leaderboard en la direccion correcta. Conviene tenerlos presentes
para no repetirlos.

| Experimentos | Que paso | Lectura |
|--------------|----------|---------|
| #7, #8 | Features extended y blend local mejoraron validacion pero bajaron LB | Sobreajuste al split local |
| #9 | Calibracion por priors bajo mucho el LB | Altero el ranking de clases de forma destructiva |
| #10 | Noisy directo bajo fuerte la validacion curated | El ruido no se puede concatenar sin control |
| #22 | 1024 frames con checkpoint entrenado en 512 bajo validacion | Cambiar inferencia sin entrenar para ese formato no alcanzo |
| #29 | Tercer seed CNN no mejoro | Mas modelos similares no garantizan diversidad |
| #46 | Reversion temporal y contraste bajaron head256 | Augmentations visuales no siempre preservan semantica de audio |
| #50 | Headsep conservador bajo contra #48 | Mejor local no implica mejor blend si el peso no esta bien ubicado |

## Lectura final

Los mayores impactos vinieron de tres decisiones:

1. **Cambiar la representacion:** de priors a features log-mel, y luego de
   stats globales a espectrogramas 2D.
2. **Mejorar el entrenamiento neural:** inicializacion, scheduler, cabeza densa,
   full-train y seeds.
3. **Mantener diversidad en ensembles:** sklearn, CNNs con variantes de cabeza,
   separable residual y ResNet50 aportaron por errores distintos.

Despues de `0.64665`, las mejoras fueron reales pero chicas. Para obtener otro
salto grande, la evidencia del repo apunta a probar informacion nueva o un
entrenamiento mas alineado con soluciones top: pretraining AudioSet, noisy data
curado, MixUp/SpecAugment mas sistematico, pseudo-labeling u OOF para calibrar
blends sin depender tanto del leaderboard.
