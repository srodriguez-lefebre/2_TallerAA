# Componente 2: globalmel_sep_temporal

## Que es

`globalmel_sep_temporal` es un modelo de clasificacion de audio basado en
espectrogramas log-mel, con dos ideas principales:

1. normalizacion global por banda mel;
2. modelado temporal con CNN separable-residual + BiGRU.

Flujo:

```text
audio .wav
-> espectrograma log-mel 128 x 512
-> normalizacion global por banda mel
-> CNN separable-residual
-> secuencia temporal de features
-> BiGRU
-> pooling temporal
-> 80 probabilidades
```

Artefacto usado:

```text
investigation/submissions/parallel100_20260702_globalmel_sep_temporal_e100_seed42/small_logmel_cnn.csv
```

## Por que existe esta rama

`separable_headsep` es una CNN fuerte, pero usa pooling global y no modela
explicitamente el orden temporal. Esta rama agrega una cabeza recurrente BiGRU
para mirar la evolucion del sonido en el tiempo.

Ademas cambia el preprocesamiento: usa normalizacion global por banda mel. Esa
fue una de las mejoras mas claras de la investigacion.

## Que es normalizacion global por banda mel

Un espectrograma log-mel tiene 128 bandas mel. Cada banda puede tener una escala
distinta: algunas bandas suelen tener mas energia, otras menos.

La normalizacion global calcula, para cada banda mel:

```text
media_global[mel_band]
desvio_global[mel_band]
```

usando todos los audios de `train_curated`.

Despues normaliza cada valor asi:

```text
valor_normalizado = (valor - media_global_de_la_banda) / desvio_global_de_la_banda
```

La diferencia contra normalizar cada clip por separado es importante:

- normalizacion por clip: cada audio se reescala usando sus propias
  estadisticas;
- normalizacion global: todos los audios quedan en una escala comun aprendida
  del conjunto de entrenamiento.

## Por que ayuda la normalizacion global

Ayuda porque hace mas comparable la entrada de distintos clips.

Sin normalizacion global, dos clips pueden tener rangos de energia muy
distintos. Eso puede hacer que el modelo aprenda diferencias de escala que no
son necesariamente relevantes para la clase.

Con normalizacion global:

- cada banda mel queda centrada de forma consistente;
- se reduce variabilidad artificial;
- el entrenamiento suele ser mas estable;
- el modelo puede concentrarse mas en patrones que en escalas absolutas.

Es una idea clasica de preprocesamiento: centrar y escalar features.

## Entrada del modelo

La entrada es:

```text
128 bandas mel x 512 frames
```

Pero con tag de cache:

```text
cache_tag: globalmel
```

Eso indica que el cache fue creado con:

```text
--normalization global-mel
```

## CNN separable-residual

La primera parte del modelo es muy parecida a `separable_headsep`. Usa bloques
separables residuales:

```text
Conv inicial
Bloque residual separable 64 -> 128
Bloque residual separable 128 -> 256
Bloque residual separable 256 -> 384
Depthwise separable 384 -> 512
```

Pero hay una diferencia clave: esta rama no termina con pooling global 2D. En
vez de colapsar toda la dimension temporal, conserva una secuencia de features
para que la BiGRU pueda procesarla.

## Que es una BiGRU

Una GRU es una red recurrente. Sirve para procesar secuencias. En vez de mirar
todos los frames como una imagen estatica, procesa la informacion paso a paso.

En este problema, la secuencia son los frames temporales del espectrograma.

Una BiGRU es una GRU bidireccional:

```text
GRU forward:  mira de izquierda a derecha
GRU backward: mira de derecha a izquierda
```

Eso permite que cada representacion temporal tenga informacion del pasado y del
futuro dentro del clip.

Por que sirve en audio:

- algunos sonidos tienen evolucion temporal;
- importa el orden de eventos;
- un patron puede depender de como empieza y como termina;
- una CNN pura puede perder parte de esa informacion al hacer pooling.

## Como se convierte la CNN en secuencia

La CNN produce un tensor con forma conceptual:

```text
batch x canales x frecuencia_reducida x tiempo_reducido
```

Para pasarlo a la BiGRU, se promedia sobre frecuencia:

```text
mean sobre frecuencia
-> batch x canales x tiempo_reducido
```

Luego se transpone para que cada paso temporal tenga un vector de features:

```text
batch x tiempo_reducido x canales
```

Eso es lo que recibe la GRU.

## Que es pooling temporal en esta rama

La BiGRU devuelve una salida por cada paso temporal. Para clasificar el clip
completo hay que resumir toda la secuencia.

Esta rama usa dos resumenes:

```text
mean pooling temporal
max pooling temporal
```

Mean pooling responde: "cual fue la activacion promedio a lo largo del clip".

Max pooling responde: "cual fue la activacion mas fuerte en algun momento del
clip".

Se concatenan ambos:

```text
[mean temporal, max temporal]
```

Esto es util porque algunas clases dependen de eventos sostenidos y otras de
eventos breves pero intensos.

## Configuracion de entrenamiento

Configuracion registrada:

```text
architecture: separable_temporal_bigru
n_mels: 128
frames: 512
cache_tag: globalmel
activation: silu
initializer: he_normal
head_dropout: 0.3
optimizer: adamw
scheduler: multistep
lr_milestones: 25,39
epochs: 100
seed: 42
full_train: true
```

## Diferencia contra separable_headsep

Comparacion:

```text
separable_headsep:
  log-mel 512
  CNN separable-residual
  pooling global 2D
  cabeza densa

globalmel_sep_temporal:
  log-mel 512 con normalizacion global por banda mel
  CNN separable-residual
  BiGRU temporal
  pooling temporal mean + max
```

La diferencia fuerte es:

- mejor normalizacion;
- modelado explicito de secuencia temporal.

## Rol en el blend final

En el blend final:

```text
0.375 * globalmel_sep_temporal_e100_seed42
```

Su rol es aportar una vista temporal con entrada mejor estandarizada. Complementa
a `separable_headsep` porque no se limita a una representacion global de imagen:
conserva el eje temporal y lo procesa con BiGRU.

Resumen corto para defensa:

```text
globalmel_sep_temporal = CNN separable-residual + BiGRU sobre log-mel 512,
con normalizacion global por banda mel. Aporta estabilidad de preprocesamiento y
modelado temporal.
```
