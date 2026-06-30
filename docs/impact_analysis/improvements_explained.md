# Que hace cada mejora importante

Este documento explica las mejoras principales del proyecto en lenguaje mas
natural, manteniendo el significado tecnico. Para ver deltas, baselines y
ranking completo, usar `docs/impact_analysis/impact_drivers.md`.

La idea central: cada mejora cambio una de estas cosas:

- **que informacion mira el modelo**;
- **como aprende de esa informacion**;
- **como se combinan modelos que se equivocan distinto**.

## 1. Priors -> sklearn log-mel

Antes de esta mejora, el sistema no escuchaba el audio. El baseline por priors
solo miraba cuantas veces aparecia cada clase en train y usaba esa frecuencia
como probabilidad para todos los audios de test.

Ejemplo: si `Bark` aparecia relativamente mucho en train, el baseline le daba
una probabilidad parecida a todos los clips, aunque el audio no tuviera ningun
ladrido.

El cambio a sklearn log-mel hizo que el modelo empezara a mirar una huella
resumida del audio:

- energia por bandas de frecuencia;
- promedio, desviacion, maximo y percentiles del log-mel;
- duracion del clip;
- volumen aproximado (`RMS`);
- cambios rapidos de signo en la onda (`zero-crossing rate`).

Despues, una LogisticRegression One-vs-Rest aprendia una regla por clase usando
esas features.

En terminos simples: se paso de **adivinar por popularidad** a **clasificar con
un resumen acustico del sonido**.

Por eso fue el salto numerico mas grande. No porque sklearn fuera una solucion
sofisticada, sino porque fue el primer modelo que uso informacion real del WAV.

## 2. Sklearn stats -> CNN sobre espectrogramas log-mel

El sklearn usaba resumenes globales del audio. Eso sirve, pero pierde el orden
temporal. Si un sonido aparece al principio, al medio o al final, muchas de esas
features pueden quedar parecidas.

La CNN sobre log-mel cambio eso. En vez de darle al modelo una lista de numeros,
se le dio una imagen del sonido:

- eje horizontal: tiempo;
- eje vertical: frecuencia;
- intensidad: energia en esa frecuencia y momento.

Con esa representacion, la CNN puede detectar formas locales:

- golpes cortos;
- sonidos repetidos;
- ruido constante;
- habla;
- instrumentos;
- agua;
- barridos de frecuencia;
- patrones que suben o bajan en frecuencia.

En terminos simples: se paso de **leer un resumen del audio** a **ver el dibujo
tiempo-frecuencia completo del sonido**.

Esta fue la mejora tecnica mas importante del proyecto. Definio que el camino
competitivo era tratar el audio como espectrograma y entrenar CNNs, no quedarse
solo con estadisticas globales.

## 3. Regularizacion sklearn con C=0.01

La LogisticRegression tiene un parametro `C`. En scikit-learn, bajar `C`
aumenta la regularizacion.

Regularizar significa obligar al modelo a ser mas simple y menos extremo. Eso
ayuda cuando:

- hay pocos datos limpios;
- hay muchas clases;
- algunas clases aparecen poco;
- hay ruido o correlaciones accidentales en el split de entrenamiento.

Con `C` mas alto, el modelo puede confiar demasiado en patrones que aparecen en
train pero no generalizan. Con `C=0.01`, las decisiones quedan mas suaves.

En terminos simples: esta mejora hizo que el sklearn fuera **menos memorioso y
mas conservador**.

Fue una mejora barata porque no cambio features, no necesito GPU y mantuvo el
mismo pipeline. Tambien dejo un baseline tabular fuerte que despues siguio
aportando en blends.

## 4. Blend CNN + sklearn

La CNN y sklearn miran el mismo audio, pero desde puntos de vista distintos.

La CNN mira patrones locales en el espectrograma. Es buena para formas
tiempo-frecuencia: golpes, repeticiones, texturas, voz, instrumentos o sonidos
que evolucionan.

El sklearn mira resumenes globales. Es mas simple, pero captura propiedades del
clip completo: duracion, energia general, distribucion por bandas y actividad
del waveform.

Aunque la CNN era mejor modelo individual, el sklearn no era inutil. A veces
acertaba donde la CNN dudaba, porque sus errores eran distintos.

El blend promedio sus probabilidades. Eso mejora cuando los modelos tienen
senales complementarias. En el repo, la correlacion entre `sklearn C=0.01` y
una rama CNN fuerte era baja, lo que confirma esa complementariedad.

En terminos simples: fue como **preguntarle a dos modelos con criterios
distintos y combinar sus opiniones**.

Esta idea se volvio una regla del proyecto: no alcanza con buscar el mejor
modelo individual; tambien importa guardar modelos diversos para ensemble.

## 5. Full-train + dos seeds CNN

Durante la busqueda de modelos, se separaba una parte de curated para validar.
Eso permite medir `valid_lwlrap`, pero deja menos datos para entrenar.

Cuando ya se eligio una configuracion, se hizo full-train: entrenar con todo el
curated. Eso le da al modelo mas ejemplos y puede mejorar el modelo final.

Ademas, se entreno la misma arquitectura con distintas semillas. Cambiar la seed
cambia detalles del entrenamiento:

- pesos iniciales;
- orden de batches;
- augmentations aplicadas;
- pequenas decisiones numericas del optimizador.

Dos modelos con la misma arquitectura terminan parecidos, pero no identicos.
Promediarlos reduce la varianza de una corrida individual.

En terminos simples: esta mejora **uso mas ejemplos y redujo la suerte de una
sola corrida**.

El efecto fue positivo, pero con retornos decrecientes. Un tercer seed no mejoro
el resultado, lo que muestra que agregar modelos parecidos no siempre agrega
diversidad util.

## 6. Mejoras estilo Fashion-MNIST

Esta etapa no cambio tanto que informacion veia la CNN. La mejora principal fue
hacer que la CNN aprendiera mejor.

Los cambios mas importantes fueron:

- **He initialization:** inicia los pesos en una escala mas adecuada para redes
  con activaciones tipo ReLU. Esto ayuda a que los gradientes circulen mejor al
  principio del entrenamiento.
- **ReduceLROnPlateau:** baja el learning rate cuando la validacion deja de
  mejorar. Eso permite avanzar rapido al principio y ajustar mas fino despues.
- **Head denso 256:** agrega una capa final con mas capacidad antes de predecir
  las 80 clases. La CNN extrae patrones; esta cabeza aprende a combinarlos mejor.
- **ReLU y variante literal:** generan modelos parecidos, pero no iguales. Eso
  aporta diversidad para el ensemble.
- **Full-train y blends finos:** reentrenan las mejores configuraciones con todo
  curated y ajustan pesos entre modelos.

En terminos simples: esta etapa hizo que la CNN **aprendiera de forma mas
estable y combinara mejor los patrones que detectaba**.

Fue el segundo gran salto del proyecto despues de adoptar CNNs sobre log-mel.
La representacion ya era buena; faltaba mejorar optimizacion, capacidad de la
cabeza y diversidad entre variantes.

## 7. Gatos/perros: separable residual, ResNet50 y headsep

Esta etapa agrego arquitecturas nuevas inspiradas en otro notebook.

### Separable residual

Usa convoluciones separables y conexiones residuales.

Una convolucion separable divide el trabajo en dos partes: primero procesa cada
canal por separado y luego mezcla canales. Eso puede dar mas capacidad con menos
costo que convoluciones normales.

Las conexiones residuales permiten que una capa aprenda una correccion sobre lo
que ya venia entrando. Eso ayuda a entrenar redes mas profundas o expresivas.

En terminos simples: el modelo pudo **mirar patrones mas complejos sin crecer
tanto en costo**.

### ResNet50 congelada

La ResNet50 estaba preentrenada en ImageNet, o sea en imagenes comunes, no en
audio. Se uso sobre espectrogramas convertidos a imagen RGB.

Como modelo individual no fue muy fuerte. Tiene sentido: no fue entrenada para
audio. Pero veia los espectrogramas con filtros distintos a las CNNs propias, y
eso produjo errores diferentes.

En terminos simples: ResNet50 no era el mejor "oyente", pero aportaba **otro
punto de vista visual sobre el espectrograma**.

### Headsep

Headsep combino la arquitectura separable residual con una cabeza densa de 256
unidades. La arquitectura extraia patrones y la cabeza aprendia a combinarlos
mejor para las 80 clases.

Fue el mejor modelo individual local de esa etapa, pero su aporte final al LB
fue moderado porque se parecia bastante a otras ramas CNN.

En terminos simples: headsep fue una **CNN mas fuerte localmente**, pero en el
blend final aporto poco porque no era completamente distinta.

## Resumen mental

La historia del proyecto se puede entender asi:

1. **Priors:** no escucha el audio, solo usa popularidad de clases.
2. **Sklearn log-mel:** escucha el audio como resumen estadistico.
3. **CNN log-mel:** ve el sonido como imagen tiempo-frecuencia.
4. **Regularizacion:** hace el baseline sklearn menos memorioso.
5. **Blends:** combina modelos que se equivocan distinto.
6. **Full-train + seeds:** usa mas datos y reduce varianza.
7. **Fashion-MNIST:** mejora como aprende la CNN.
8. **Gatos/perros:** agrega arquitecturas y puntos de vista nuevos para el
   ensemble.

La mejora mas importante tecnicamente fue pasar a **CNN sobre espectrogramas
log-mel**. La mejora que mas refinamiento competitivo aporto despues fue la
etapa **Fashion-MNIST**, porque hizo que esa CNN aprendiera bastante mejor.
