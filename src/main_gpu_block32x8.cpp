#include <algorithm>
#include <chrono>
#include <cmath>
#include <experimental/filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <memory>
#include "drop_cuda.hpp"

namespace {

struct Config {
    int width = 640;
    int height = 640;
    double seconds = 30.0;
    int fps = 30;
    int steps_per_frame = 1;
    float wave_speed = 0.45f;
    float damping = 0.006f;
    float edge_damping = 0.035f;
    float drop_radius = 18.0f;
    float drop_strength = 1.0f;
    std::string output = "output/drop_simulation_gpu_block32x8.mp4";
};

int index_of(int x, int y, int width) {
    return y * width + x;
}

void add_drop(std::vector<float>& current, std::vector<float>& previous, const Config& cfg) {
    const float cx = 0.5f * static_cast<float>(cfg.width - 1);
    const float cy = 0.5f * static_cast<float>(cfg.height - 1);
    const float sigma2 = cfg.drop_radius * cfg.drop_radius;

    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            const float r2 = dx * dx + dy * dy;
            const float pulse = cfg.drop_strength * std::exp(-r2 / (2.0f * sigma2));
            const int idx = index_of(x, y, cfg.width);
            current[idx] += pulse;
            previous[idx] -= 0.35f * pulse;
        }
    }
}
} //namespace

int main() {
    try {
        const Config cfg;

        const int total_frames =
            static_cast<int>(std::round(cfg.seconds * cfg.fps));

        const std::size_t cells =
            static_cast<std::size_t>(cfg.width) * cfg.height;

        // Crear directorio de salida
        const std::experimental::filesystem::path output_path(
            cfg.output
        );

        if (output_path.has_parent_path()) {
            std::experimental::filesystem::create_directories(
                output_path.parent_path()
            );
        }

        // Inicializacion en CPU
        std::vector<float> previous(cells, 0.0f);
        std::vector<float> current(cells, 0.0f);

        add_drop(current, previous, cfg);

        // Preparar escritura del video
        cv::VideoWriter writer(
            cfg.output,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            static_cast<double>(cfg.fps),
            cv::Size(cfg.width, cfg.height)
        );

        if (!writer.isOpened()) {
            throw std::runtime_error(
                "No se pudo abrir el archivo de salida"
            );
        }

        // Iniciar medicion del tiempo
        const auto start = std::chrono::steady_clock::now();

        // Reservar memoria GPU y copiar estados iniciales
        std::unique_ptr<GpuState, decltype(&gpu_destroy)> gpu(
            gpu_create(
                previous.data(),
                current.data(),
                cfg.width,
                cfg.height,
                cfg.wave_speed,
                cfg.damping,
                cfg.edge_damping
            ),
            &gpu_destroy
        );

        // Ciclo principal de simulacion
        for (int frame = 0; frame < total_frames; ++frame) {

            // Actualizar ecuacion de onda en GPU
            for (int step = 0;
                 step < cfg.steps_per_frame;
                 ++step) {

                gpu_step(gpu.get());
            }

            // Preparar imagen en CPU
            cv::Mat image(
                cfg.height,
                cfg.width,
                CV_8UC3
            );

            // Renderizar en GPU y copiar imagen a CPU
            gpu_render(
                gpu.get(),
                image.ptr<unsigned char>()
            );

            // Agregar texto al fotograma
            cv::putText(
                image,
                "GPU float32 | frame " + std::to_string(frame),
                cv::Point(18, 32),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(235, 235, 235),
                1,
                cv::LINE_AA
            );

            // Escribir el video mediante OpenCV
            writer.write(image);

            if (frame % std::max(1, total_frames / 10) == 0) {
                std::cout << "Frame " << frame
                          << " / " << total_frames << '\n';
            }
        }

        // Finalizar medicion
        const auto end = std::chrono::steady_clock::now();

        const double elapsed =
            std::chrono::duration<double>(
                end - start
            ).count();

        const double simulated_steps =
            static_cast<double>(total_frames) *
            cfg.steps_per_frame;

        std::cout << "Video generado: " << cfg.output << '\n';
        std::cout << "Tiempo GPU: " << elapsed << " s\n";

        std::cout << "Pasos simulados: "
                  << simulated_steps << '\n';

        std::cout << "Rendimiento: "
                  << simulated_steps / elapsed
                  << " pasos/s\n";

    } catch (const std::exception& e) {

        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
