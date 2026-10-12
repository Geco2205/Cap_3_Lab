#pragma once

struct GpuState;

// Reservar memoria GPU e inicializar datos
GpuState* gpu_create(
    const float* previous,
    const float* current,
    int width,
    int height,
    float wave_speed,
    float damping,
    float edge_damping
);

// Ejecutar un paso de simulacion en GPU
void gpu_step(GpuState* state);

// Renderizar un fotograma en GPU
void gpu_render(
    GpuState* state,
    unsigned char* host_bgr
);

// Reservar buffer host pinned para un fotograma BGR
unsigned char* gpu_alloc_pinned(int width, int height);

// Liberar buffer host pinned
void gpu_free_pinned(unsigned char* ptr) noexcept;

// Encolar render + copia D2H asincrona en el stream; registra el evento del slot
void gpu_render_async(
    GpuState* state,
    unsigned char* host_pinned,
    int slot
);

// Esperar a que termine el fotograma del slot
void gpu_wait(GpuState* state, int slot);

// Liberar los recursos de GPU
void gpu_destroy(GpuState* state) noexcept;
