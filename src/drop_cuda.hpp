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

// Liberar los recursos de GPU
void gpu_destroy(GpuState* state) noexcept;
