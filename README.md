# Cap_3_Lab
Laboratorio del Capítulo 3

# Reporte: Laboratorio 5, Computación Heterogénea y GPU

EL5859 Computación Heterogénea, II Semestre 2026.
Integrantes: Gonzalo Alpízar Salas, Gerson Adrián Cordero Zúñiga, Nicole Irina Corrales Rodríguez, Keilin Loáisiga Téllez

## Contenido

1. [Descripción del programa base](#1-descripción-del-programa-base)
2. [Diagrama de flujo](#2-diagrama-de-flujo)
3. [Explicación de cada función](#3-explicación-de-cada-función)
4. [Resultados del perfilado CPU](#4-resultados-del-perfilado-cpu)
5. [Justificación de las funciones portadas a GPU](#5-justificación-de-las-funciones-portadas-a-gpu)
6. [Diseño de los kernels CUDA](#6-diseño-de-los-kernels-cuda)
7. [Estrategia de validación CPU vs GPU](#7-estrategia-de-validación-cpu-vs-gpu)
8. [Resultados de cada optimización](#8-resultados-de-cada-optimización)
9. [Capturas de Nsight / nvprof](#9-capturas-de-nsight--nvprof)
10. [Análisis de rendimiento](#10-análisis-de-rendimiento)
11. [Conclusiones](#11-conclusiones)
12. [Instrucciones de compilación y ejecución](#12-instrucciones-de-compilación-y-ejecución)
13. [Declaración de uso de IA](#13-declaración-de-uso-de-ia)

---

## 1. Descripción del programa base

El programa simula la caída de una gota sobre un estanque y genera un video de 30 s
en escala de grises. La superficie del agua es una malla de 640 × 640 alturas
(`float`). En cada paso de tiempo se calcula cómo cambia la altura de cada celda a
partir de sus vecinas y de su movimiento anterior. Después cada estado se dibuja
con iluminación simple para que parezca agua y se escribe como un cuadro del video.

### Modelo físico

Lo que se aproxima es la **ecuación de onda 2D amortiguada**:

```
∂²h/∂t² = c² ∇²h − γ ∂h/∂t
```

- `h(x, y, t)` es la altura del agua.
- `c` (`wave_speed` = 0.45) es la velocidad con la que viaja la onda.
- `γ` (`damping`) es la pérdida de energía, que hace que las ondas se vayan apagando.

Se resuelve con diferencias finitas explícitas (esquema tipo Verlet), tomando
`dt = dx = 1`:

```
laplaciano = izquierda + derecha + arriba + abajo − 4·centro
h_next = 2·h_cur − h_prev + c²·laplaciano − amortiguamiento·(h_cur − h_prev)
```

La condición inicial es una gaussiana de radio 18 celdas en el centro del estanque,
con una velocidad inicial para que la onda salga hacia afuera. En los bordes hay una
capa absorbente de 32 píxeles que aumenta el amortiguamiento para que las olas no
reboten contra el marco. Para que el esquema sea estable en 2D se necesita
`c ≤ 1/√2 ≈ 0.707`, y con `c = 0.45` se cumple.

### Línea base CPU (Ejercicio A)

| Métrica | Valor |
|---|---|
| Resolución | 640 × 640 px |
| Duración del video | 30 s |
| FPS | 30 |
| Pasos simulados | 900 (1 paso por cuadro) |
| Tiempo total (promedio de 4 corridas) | 13.29 s (σ ≈ 0.05 s) |
| Rendimiento (promedio) | 67.7 pasos/s |
| Uso de CPU | 99 % (un solo hilo) |
| Memoria máxima | ≈ 73 MB |
| Entorno | x86_64, gcc 13.3.0, Meson 1.3.2, OpenCV 4.6.0, `--buildtype=release` |

Las 4 corridas dieron 13.367, 13.283, 13.263 y 13.257 s. La primera fue la más lenta,
posiblemente porque el sistema aún no tenía el programa en caché. El tiempo medido
por el programa incluye la simulación, el renderizado y la escritura del video, pero
no la reserva de memoria ni `add_drop`. Por eso **13.29 s** es la referencia para
calcular los speedups.

## 2. Diagrama de flujo

## 3. Explicación de cada función

El código fuente comentado está en [`src/main.cpp`](src/main.cpp).

**`Config`.** Estructura que junta todos los parámetros: tamaño de la malla, duración
y fps, las constantes de la física, el tamaño de la gota y la ruta del video. Se crea
una sola vez en `main` como `const` y las demás funciones la reciben, así no hay
números escritos a mano en el código.

**`index_of`.** Convierte una posición `(x, y)` en el índice del vector. La malla está
guardada por filas, por lo que las celdas de una misma fila quedan seguidas en memoria.

**`add_drop`.** Crea la perturbación inicial. Suma una gaussiana en el centro a
`current` y le resta una parte a `previous`; como la velocidad es `current − previous`,
la gota arranca con velocidad y no solo con altura. Se ejecuta una vez antes del ciclo.

**`border_absorption`.** Devuelve el amortiguamiento de cada celda según su distancia
al borde. En el centro es el valor base, y en los 32 px cercanos al borde crece de
forma cuadrática hasta `damping + edge_damping`. Es una capa absorbente: la onda se
apaga antes de llegar al marco y no rebota.

**`simulate_step`.** Hace avanzar la simulación un paso. Con `previous` y `current`
calcula `next`, usando el laplaciano de 5 puntos y restando el amortiguamiento por la
velocidad. Cada celda depende solo de valores del paso anterior, por lo que todas se
pueden calcular a la vez. El borde de la malla queda en 0.

**`render_frame`.** Convierte la malla de alturas en la imagen de un cuadro. Por cada
píxel calcula la pendiente de la superficie con los vecinos, de ahí una normal, y con
ella la luz difusa y un brillo especular en las crestas. Eso se combina con la altura
para obtener el gris. Cada píxel es independiente de los demás.

**`main`.** Reserva las tres mallas, pone la gota, abre el video y ejecuta el ciclo:
por cada cuadro avanza la simulación, renderiza y escribe al video. Al terminar
calcula el tiempo y los pasos por segundo.

## 4. Resultados del perfilado CPU

> Pendiente (Ejercicio C).

## 5. Justificación de las funciones portadas a GPU

> Pendiente (Ejercicio D).

## 6. Diseño de los kernels CUDA

> Pendiente (Ejercicio D).

## 7. Estrategia de validación CPU vs GPU

> Pendiente (Ejercicio E).

## 8. Resultados de cada optimización

| Versión | Tiempo total | Pasos/s | Speedup | Error máximo |
|---|---|---|---|---|
| CPU base | 13.29 s | 67.7 | 1.00x | referencia |
| GPU base | | | | |
| Optimización 1 | | | | |
| Optimización 2 | | | | |
| Optimización 3 | | | | |

> Pendiente (Ejercicios D y F). Para cada optimización: qué problema resuelve, qué
> cambió en el código, qué evidencia de perfilado la justifica, cuánto mejoró o
> empeoró y qué impacto tuvo en la precisión.

## 9. Capturas de Nsight / nvprof

> Pendiente (Ejercicio F).

## 10. Análisis de rendimiento

> Pendiente.

## 11. Conclusiones

> Pendiente.

## 12. Instrucciones de compilación y ejecución

```bash
meson setup build --buildtype=release
meson compile -C build
./build/drop_simulation
```

El programa genera `output/drop_simulation.mp4` e imprime el tiempo, los pasos
simulados y el rendimiento. Para medir el tiempo y la memoria de la línea base:

```bash
/usr/bin/time -v ./build/drop_simulation 2>&1 | tee output/run_cpu_base.log
```

Para revisar los parámetros del video generado:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,duration,nb_frames \
  -of default=nw=1 output/drop_simulation.mp4
```

Requisitos: g++, Meson, Ninja y OpenCV 4 (`libopencv-dev`).

## 13. Declaración de uso de IA


