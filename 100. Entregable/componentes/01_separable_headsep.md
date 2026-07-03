# Componente 1: separable_headsep

## Que es

`separable_headsep` es una red neuronal convolucional que clasifica audio a
partir de espectrogramas log-mel. No recibe el audio crudo directamente. Primero
el audio se transforma a una imagen tiempo-frecuencia:

```text
audio .wav
-> waveform mono
-> espectrograma mel
-> escala log/dB
-> imagen log-mel 128 x 512
-> CNN separable-residual
-> cabeza densa
-> 80 probabilidades, una por clase
```

En el blend final cumple el rol de componente convolucional fuerte. Es el modelo
que mira el espectrograma mas como una imagen y aprende patrones locales:
bandas, golpes, texturas, armonicos, ruido, ataques cortos y formas
tiempo-frecuencia.

Artefacto usado:

```text
investigation/submissions/parallel100_20260702_separable_headsep_e100_seed42/small_logmel_cnn.csv
```

## Entrada: log-mel 128 x 512

La entrada es una matriz:

```text
128 bandas mel x 512 frames temporales
```

Cada fila representa una banda de frecuencia en escala mel. La escala mel
aproxima mejor la percepcion humana que una escala lineal de frecuencia. Cada
columna representa un instante temporal. Entonces el espectrograma puede
interpretarse como una imagen donde:

- eje vertical: frecuencia;
- eje horizontal: tiempo;
- valor del pixel: energia logaritmica en esa zona.

Usar log-mel es estandar en clasificacion de audio porque comprime la senal
cruda en una representacion mas informativa y mas facil de procesar con CNNs.

## Que es una convolucion en este contexto

Una convolucion aplica filtros pequenos sobre la imagen. En audio log-mel, esos
filtros aprenden patrones locales como:

- energia concentrada en cierta banda;
- cambios bruscos en el tiempo;
- bandas horizontales sostenidas;
- patrones verticales de ataque;
- texturas de ruido.

La ventaja de una CNN es que el mismo filtro se reutiliza en muchas posiciones.
Eso reduce parametros y permite detectar un patron aunque aparezca en otro
momento del audio.

## Que significa "separable"

Este componente usa convoluciones separables en profundidad. Una convolucion
normal mezcla dos cosas al mismo tiempo:

1. mira vecinos espaciales, por ejemplo un parche 3x3;
2. mezcla canales.

Una convolucion separable separa esas dos operaciones:

```text
depthwise convolution
-> mira el patron espacial dentro de cada canal por separado

pointwise convolution 1x1
-> mezcla informacion entre canales
```

En codigo, el bloque es:

```text
Conv2d 3x3 con groups=in_channels
Conv2d 1x1
BatchNorm
ReLU
```

Por que sirve:

- baja la cantidad de parametros;
- baja costo computacional;
- obliga al modelo a aprender primero patrones locales y despues combinarlos;
- esta inspirado en familias como Xception/MobileNet.

En este proyecto esto es defendible porque queremos una CNN fuerte pero no
excesivamente grande para el tamano del dataset curated.

## Que significa "residual"

Un bloque residual no solo transforma la entrada. Tambien conserva un camino
directo desde la entrada hacia la salida.

La idea general es:

```text
salida = transformacion_convolucional(x) + proyeccion_residual(x)
```

En este modelo, cada bloque residual tiene:

```text
rama principal:
  DepthwiseSeparableConv
  DepthwiseSeparableConv
  MaxPool

rama residual:
  Conv2d 1x1 con stride 2
  BatchNorm

salida:
  ReLU(rama principal + rama residual)
```

La rama residual usa `Conv2d 1x1` porque las dimensiones cambian: el bloque
reduce resolucion con stride/pooling y aumenta canales. La proyeccion 1x1 adapta
la entrada para que pueda sumarse con la salida de la rama principal.

Por que sirve:

- facilita el flujo del gradiente durante el entrenamiento;
- reduce el riesgo de que capas profundas degraden el aprendizaje;
- permite que el bloque aprenda una correccion sobre la entrada en vez de
  reconstruir todo desde cero.

En palabras simples: el modelo puede decir "conservo parte de lo que ya sabia y
agrego una transformacion nueva".

## Arquitectura concreta

La arquitectura usada en `separable_headsep` es:

```text
Input: 1 x 128 x 512

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

AdaptiveAvgPool2d(1, 1)

Head densa:
  Flatten
  Linear 512 -> 256
  BatchNorm1d
  ReLU
  Dropout 0.3
  Linear 256 -> 80
```

## Que es pooling global

El pooling global resume cada mapa de activacion completo en un solo numero.

Antes del pooling, la CNN tiene algo como:

```text
512 canales x frecuencia_reducida x tiempo_reducido
```

Con `AdaptiveAvgPool2d((1, 1))`, cada canal se promedia sobre toda su dimension
de frecuencia y tiempo:

```text
512 canales x H x W
-> 512 canales x 1 x 1
```

Es decir, por cada filtro aprendido, el modelo se queda con una medida global de
cuanto aparecio ese patron en el clip.

Por que sirve:

- permite producir un vector de tamano fijo;
- reduce parametros antes de la cabeza final;
- hace que el modelo dependa menos de la posicion exacta del evento;
- resume "si aparecio este patron" mas que "donde aparecio exactamente".

Limitacion:

- al promediar globalmente, se pierde parte del orden temporal fino.

Por eso este componente se complementa con las ramas BiGRU del blend.

## Que es la cabeza densa de 256

Despues del pooling global queda un vector de 512 valores. La cabeza densa lo
transforma en predicciones:

```text
512 features
-> capa densa de 256
-> BatchNorm
-> ReLU
-> Dropout
-> capa final de 80 salidas
```

La capa de 256 permite combinar patrones aprendidos por la CNN antes de decidir
las clases. Por ejemplo, una clase puede depender de varios patrones al mismo
tiempo: ataque corto, energia en ciertas bandas y textura sostenida.

El dropout reduce sobreajuste apagando aleatoriamente parte de la representacion
durante entrenamiento.

## Configuracion de entrenamiento

Configuracion registrada:

```text
architecture: separable_residual
n_mels: 128
frames: 512
activation: relu
initializer: he_normal
head_hidden: 256
head_dropout: 0.3
optimizer: adam
scheduler: multistep
epochs: 100
seed: 42
full_train: true
```

Se entreno con todo `train_curated` para generar predicciones sobre el test.

## Rol en el blend final

En el blend final:

```text
0.25 * separable_headsep_e100_seed42
```

Su rol es aportar una vista convolucional fuerte. Es distinto a los otros dos
componentes porque:

- no usa BiGRU;
- resume con pooling global;
- se concentra en patrones espaciales del espectrograma;
- usa una cabeza densa para combinar features.

Resumen corto para defensa:

```text
separable_headsep = CNN separable-residual sobre log-mel, con pooling global y
cabeza densa de 256. Aporta el componente convolucional fuerte del ensemble.
```
