#include "drop_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>


// Estado de la simulacion en GPU
struct GpuState {
    float* previous = nullptr;
    float* current = nullptr;
    float* next = nullptr;

    unsigned char* frame_bgr = nullptr;

    int width = 0;
    int height = 0;

    float wave_speed = 0.0f;
    float damping = 0.0f;
    float edge_damping = 0.0f;

    cudaStream_t stream = nullptr;
    cudaEvent_t frame_done[2] = {nullptr, nullptr};
};


namespace {

// Comprobar errores de CUDA
void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(result)
        );
    }
}


// Calcular la absorcion en los bordes
__device__ float border_absorption_device(
    int x,
    int y,
    int width,
    int height,
    float damping,
    float edge_damping
) {
    const int band = 32;

    int dist = x < y ? x : y;

    const int right = width - 1 - x;
    const int bottom = height - 1 - y;

    if (right < dist) dist = right;
    if (bottom < dist) dist = bottom;

    if (dist >= band) {
        return damping;
    }

    const float t =
        1.0f - static_cast<float>(dist) /
        static_cast<float>(band);

    return damping + edge_damping * t * t;
}


// Kernel CUDA: actualizacion de la ecuacion de onda
__global__ void simulate_step_kernel(
    const float* previous,
    const float* current,
    float* next,
    int width,
    int height,
    float wave_speed,
    float damping,
    float edge_damping
) {
    // Coordenadas del hilo
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Verificar limites de la malla
    if (x >= width || y >= height) {
        return;
    }

    const int idx = y * width + x;

    // Condiciones de borde
    if (x == 0 || y == 0 ||
        x == width - 1 || y == height - 1) {

        next[idx] = 0.0f;
        return;
    }

    // Laplaciano discreto
    const float laplacian =
        current[idx - 1] +
        current[idx + 1] +
        current[idx - width] +
        current[idx + width] -
        4.0f * current[idx];

    // Velocidad local
    const float velocity =
        current[idx] - previous[idx];

    // Amortiguamiento
    const float local_damping =
        border_absorption_device(
            x, y, width, height,
            damping, edge_damping
        );

    const float c2 = wave_speed * wave_speed;

    // Actualizacion de la onda
    next[idx] =
        2.0f * current[idx] -
        previous[idx] +
        c2 * laplacian -
        local_damping * velocity;
}


// Kernel CUDA: renderizado de cada fotograma
__global__ void render_frame_kernel(
    const float* height_map,
    unsigned char* bgr,
    int width,
    int height,
    float lx,
    float ly,
    float lz
) {
    // Coordenadas del hilo
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // Coordenadas de los vecinos
    const int xm = x > 0 ? x - 1 : 0;
    const int xp = x + 1 < width ? x + 1 : width - 1;

    const int ym = y > 0 ? y - 1 : 0;
    const int yp = y + 1 < height ? y + 1 : height - 1;

    const int idx = y * width + x;

    // Pendientes de la superficie
    const float dx =
        height_map[y * width + xm] -
        height_map[y * width + xp];

    const float dy =
        height_map[ym * width + x] -
        height_map[yp * width + x];

    // Vector normal
    const float nx = 2.8f * dx;
    const float ny = 2.8f * dy;
    const float nz = 1.0f;

    const float norm = sqrtf(
        nx * nx + ny * ny + nz * nz
    );

    // Iluminacion difusa
    float diffuse =
        (nx * lx + ny * ly + nz * lz) / norm;

    if (diffuse < 0.0f) {
        diffuse = 0.0f;
    }

    // Altura de la onda
    float wave =
        0.5f + 1.8f * height_map[idx];

    if (wave < 0.0f) wave = 0.0f;
    if (wave > 1.0f) wave = 1.0f;

    // Componente especular
    const float specular =
        powf(diffuse, 24.0f);

    // Intensidad del pixel
    float intensity =
        35.0f + 120.0f * wave;

    intensity *=
        0.60f + 0.65f * diffuse;

    intensity +=
        130.0f * specular;

    // Limitar intensidad entre 0 y 255
    if (intensity < 0.0f) intensity = 0.0f;
    if (intensity > 255.0f) intensity = 255.0f;

    const unsigned char gray =
        static_cast<unsigned char>(intensity);

    // Imagen BGR en escala de grises
    bgr[3 * idx + 0] = gray;
    bgr[3 * idx + 1] = gray;
    bgr[3 * idx + 2] = gray;
}

} // namespace


// Reservar memoria GPU e inicializar datos
GpuState* gpu_create(
    const float* previous,
    const float* current,
    int width,
    int height,
    float wave_speed,
    float damping,
    float edge_damping
) {
    GpuState* state = new GpuState;

    state->width = width;
    state->height = height;

    state->wave_speed = wave_speed;
    state->damping = damping;
    state->edge_damping = edge_damping;

    // Cantidad de elementos y bytes
    const std::size_t cells =
        static_cast<std::size_t>(width) * height;

    const std::size_t mesh_bytes =
        cells * sizeof(float);

    try {
        // Reservar memoria GPU para las tres mallas
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&state->previous),
                mesh_bytes
            ),
            "cudaMalloc previous"
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&state->current),
                mesh_bytes
            ),
            "cudaMalloc current"
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&state->next),
                mesh_bytes
            ),
            "cudaMalloc next"
        );

        // Reservar memoria GPU para imagen BGR
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&state->frame_bgr),
                3 * cells
            ),
            "cudaMalloc frame"
        );

        // Copiar estado anterior CPU -> GPU
        check_cuda(
            cudaMemcpy(
                state->previous,
                previous,
                mesh_bytes,
                cudaMemcpyHostToDevice
            ),
            "H2D previous"
        );

        // Copiar estado actual CPU -> GPU
        check_cuda(
            cudaMemcpy(
                state->current,
                current,
                mesh_bytes,
                cudaMemcpyHostToDevice
            ),
            "H2D current"
        );

        // Stream y eventos (uno por buffer pinned)
        check_cuda(cudaStreamCreate(&state->stream), "cudaStreamCreate");

        for (int i = 0; i < 2; ++i) {
            check_cuda(
                cudaEventCreateWithFlags(
                    &state->frame_done[i],
                    cudaEventDisableTiming
                ),
                "cudaEventCreate"
            );
        }

    } catch (...) {
        gpu_destroy(state);
        throw;
    }

    return state;
}


// Ejecutar un paso de simulacion en GPU
void gpu_step(GpuState* state) {

    // Configuracion de bloques e hilos
    const dim3 block(32, 8);

    const dim3 grid(
        (state->width + 31) / 32,
        (state->height + 7) / 8
    );

    // Lanzamiento del kernel de simulacion
    simulate_step_kernel<<<grid, block, 0, state->stream>>>(
        state->previous,
        state->current,
        state->next,
        state->width,
        state->height,
        state->wave_speed,
        state->damping,
        state->edge_damping
    );

    check_cuda(
        cudaGetLastError(),
        "simulate_step_kernel"
    );

    // Intercambiar punteros sin copiar arreglos
    std::swap(state->previous, state->current);
    std::swap(state->current, state->next);
}


// Renderizar en GPU y copiar imagen a CPU
void gpu_render(
    GpuState* state,
    unsigned char* host_bgr
) {
    // Configuracion de bloques e hilos
    const dim3 block(32, 8);

    const dim3 grid(
        (state->width + 31) / 32,
        (state->height + 7) / 8
    );

    // Vector de iluminacion normalizado
    const float light_norm = std::sqrt(
        0.35f * 0.35f +
        0.55f * 0.55f +
        0.76f * 0.76f
    );

    const float lx = -0.35f / light_norm;
    const float ly = -0.55f / light_norm;
    const float lz = 0.76f / light_norm;

    // Lanzamiento del kernel de renderizado
    render_frame_kernel<<<grid, block>>>(
        state->current,
        state->frame_bgr,
        state->width,
        state->height,
        lx,
        ly,
        lz
    );

    check_cuda(
        cudaGetLastError(),
        "render_frame_kernel"
    );

    // Tamano del fotograma BGR
    const std::size_t bytes =
        3ULL * state->width * state->height;

    // Copiar el fotograma GPU -> CPU
    check_cuda(
        cudaMemcpy(
            host_bgr,
            state->frame_bgr,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "D2H frame"
    );
}

// Reservar buffer host pinned (page-locked) para un fotograma BGR
unsigned char* gpu_alloc_pinned(int width, int height) {
    void* ptr = nullptr;

    check_cuda(
        cudaHostAlloc(
            &ptr,
            3ULL * width * height,
            cudaHostAllocDefault
        ),
        "cudaHostAlloc"
    );

    return static_cast<unsigned char*>(ptr);
}


// Liberar buffer host pinned
void gpu_free_pinned(unsigned char* ptr) noexcept {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}


// Renderizar y copiar D2H de forma asincrona en el stream
void gpu_render_async(
    GpuState* state,
    unsigned char* host_pinned,
    int slot
) {
    const dim3 block(32, 8);

    const dim3 grid(
        (state->width + 31) / 32,
        (state->height + 7) / 8
    );

    const float light_norm = std::sqrt(
        0.35f * 0.35f +
        0.55f * 0.55f +
        0.76f * 0.76f
    );

    const float lx = -0.35f / light_norm;
    const float ly = -0.55f / light_norm;
    const float lz = 0.76f / light_norm;

    render_frame_kernel<<<grid, block, 0, state->stream>>>(
        state->current,
        state->frame_bgr,
        state->width,
        state->height,
        lx,
        ly,
        lz
    );

    check_cuda(
        cudaGetLastError(),
        "render_frame_kernel"
    );

    const std::size_t bytes =
        3ULL * state->width * state->height;

    check_cuda(
        cudaMemcpyAsync(
            host_pinned,
            state->frame_bgr,
            bytes,
            cudaMemcpyDeviceToHost,
            state->stream
        ),
        "D2H async frame"
    );

    // Marca el fin de la copia de este fotograma
    check_cuda(
        cudaEventRecord(state->frame_done[slot], state->stream),
        "cudaEventRecord"
    );
}


// Esperar a que el fotograma del slot este en el host
void gpu_wait(GpuState* state, int slot) {
    check_cuda(
        cudaEventSynchronize(state->frame_done[slot]),
        "cudaEventSynchronize"
    );
}

// Liberar recursos de GPU
void gpu_destroy(GpuState* state) noexcept {

    if (!state) {
        return;
    }

    if (state->previous) {
        cudaFree(state->previous);
    }

    if (state->current) {
        cudaFree(state->current);
    }

    if (state->next) {
        cudaFree(state->next);
    }

    if (state->frame_bgr) {
        cudaFree(state->frame_bgr);
    }

    for (int i = 0; i < 2; ++i) {
        if (state->frame_done[i]) {
            cudaEventDestroy(state->frame_done[i]);
        }
    }

    if (state->stream) {
        cudaStreamDestroy(state->stream);
    }

    delete state;
}
