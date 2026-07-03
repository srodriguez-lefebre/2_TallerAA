# Componente 3: sep_temporal_f1024

## Que es

`sep_temporal_f1024` es una rama temporal CNN + BiGRU que usa una ventana de
entrada mas larga:

```text
audio .wav
-> espectrograma log-mel 128 x 1024
-> CNN separable-residual
-> secuencia temporal de features
-> BiGRU
-> pooling temporal
-> 80 probabilidades
```

Artefacto usado:

```text
investigation/submissions/parallel100_20260702_sep_temporal_f1024_e100_seed42/small_logmel_cnn.csv
```

## Idea principal

La diferencia principal de esta rama es el contexto temporal.

Las ramas de 512 frames ven:

```text
128 bandas mel x 512 frames
```

Esta rama ve:

```text
128 bandas mel x 1024 frames
```

Eso significa que recibe una ventana temporal mas larga del audio antes de
clasificar.

## Por que puede ayudar una ventana mas larga

No todos los sonidos se identifican con un fragmento corto. Algunas clases
pueden necesitar mas contexto:

- eventos que tienen inicio, desarrollo y final;
- sonidos repetidos;
- texturas largas;
- ambientes sostenidos;
- patrones que aparecen tarde en el clip.

Con 512 frames, el modelo puede perder parte de esa informacion si el evento es
largo o aparece fuera de la zona mas representativa. Con 1024 frames, la rama
tiene mas oportunidad de ver la estructura completa.

## Relacion con secuencias temporales

En audio, el tiempo importa. Un espectrograma no es solo una imagen cualquiera:
el eje horizontal tiene orden. El frame 10 ocurre antes que el frame 100.

Por eso una rama BiGRU tiene sentido. La BiGRU procesa una secuencia de
representaciones temporales y puede aprender patrones como:

```text
primero aparece energia baja
despues sube una banda
despues aparece una textura repetida
```

Una CNN con pooling global puede detectar que ciertos patrones existen, pero
puede perder parte del orden. La BiGRU intenta conservar esa informacion.

## Arquitectura general

La arquitectura es:

```text
Input: 1 x 128 x 1024

Conv inicial:
  1 -> 64 canales

Bloque residual separable:
  64 -> 128

Bloque residual separable:
  128 -> 256

Bloque residual separable:
  256 -> 384

Depthwise separable final:
  384 -> 512

BiGRU temporal

Pooling temporal:
  mean + max

Linear final:
  -> 80 clases
```

## Que comparte con globalmel_sep_temporal

Comparte la misma familia de arquitectura:

```text
CNN separable-residual + BiGRU
```

Eso significa:

- usa convoluciones separables para extraer patrones locales;
- usa conexiones residuales para entrenar bloques profundos;
- conserva el eje temporal;
- usa BiGRU para procesar la secuencia;
- resume la salida temporal con mean y max pooling.

## Que cambia respecto a globalmel_sep_temporal

La diferencia es:

```text
globalmel_sep_temporal:
  512 frames
  normalizacion global por banda mel

sep_temporal_f1024:
  1024 frames
  normalizacion por clip/base
```

Entonces no son duplicados. Una rama aporta normalizacion global; la otra aporta
mayor longitud temporal.

## Que es el costo de usar 1024 frames

Usar mas frames no es gratis:

- aumenta memoria;
- baja el batch size posible;
- tarda mas en entrenar;
- puede introducir mas ruido si partes largas del clip no son relevantes.

Por eso esta rama se usa como complemento, no como unico modelo. El blend deja
que aporte cuando su contexto largo ayuda, pero no obliga a depender solo de esa
vista.

## Configuracion de entrenamiento

Configuracion registrada:

```text
architecture: separable_temporal_bigru
n_mels: 128
frames: 1024
cache_tag: none
activation: silu
initializer: he_normal
head_dropout: 0.3
optimizer: adamw
scheduler: multistep
lr_milestones: 19,25
epochs: 100
batch_size: 12
seed: 42
full_train: true
```

El batch size es menor que en ramas de 512 frames porque la entrada ocupa mas
memoria.

## Por que entra al blend final

En el blend final:

```text
0.375 * sep_temporal_f1024_e100_seed42
```

Su rol es aportar una escala temporal distinta. Esto mejora la diversidad del
ensemble porque no esta mirando exactamente la misma ventana que las ramas de
512 frames.

## Diferencia contra separable_headsep

```text
separable_headsep:
  CNN sobre log-mel 512
  pooling global
  cabeza densa

sep_temporal_f1024:
  CNN sobre log-mel 1024
  BiGRU temporal
  pooling temporal mean + max
```

La primera es mas "visual/convolucional". La segunda es mas "secuencial".

## Diferencia contra globalmel_sep_temporal

```text
globalmel_sep_temporal:
  mejor normalizacion
  contexto 512

sep_temporal_f1024:
  contexto mas largo
  normalizacion base
```

Ambas son temporales, pero su hipotesis de mejora es diferente.

## Resumen para defensa

```text
sep_temporal_f1024 = CNN separable-residual + BiGRU sobre log-mel con ventana de
1024 frames. Aporta contexto temporal largo y complementa las ramas de 512
frames.
```
