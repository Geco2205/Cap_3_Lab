#include <algorithm>
#include <chrono>
#include <cmath>
#include <experimental/filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Config guarda todos los parámetros de la simulación juntos: tamaño de la malla,
// duración y fps del video, las constantes de la física (velocidad de la onda y
// amortiguamientos), el tamaño de la gota y el archivo de salida. En main se crea
// una sola vez como const y todas las funciones la reciben, así ningún valor
// queda escrito a mano dentro del código y se puede cambiar el experimento desde
// un solo lugar. Con los valores por defecto salen 30 s * 30 fps = 900 cuadros, y
// como steps_per_frame es 1, también 900 pasos de simulación.
struct Config {
    int width = 640;
    int height = 640;
    double seconds = 30.0;
    int fps = 30;
    int steps_per_frame = 1;       // pasos de simulación por cada cuadro del video
    float wave_speed = 0.45f;      // velocidad de la onda; en 2D tiene que ser <= 1/sqrt(2) para que sea estable
    float damping = 0.006f;        // pérdida de energía en todo el estanque
    float edge_damping = 0.035f;   // amortiguamiento extra que se suma cerca del borde
    float drop_radius = 18.0f;     // tamaño de la gota (sigma de la gaussiana)
    float drop_strength = 1.0f;    // altura de la gota en el centro
    std::string output = "output/drop_simulation.mp4";
};

// Convierte una posición (x, y) de la malla al índice del vector. La malla se
// guarda por filas, entonces las celdas de una misma fila quedan seguidas en
// memoria. Todas las demás funciones la usan para leer y escribir alturas.
int index_of(int x, int y, int width) {
    return y * width + x;
}

// Esta función crea la perturbación inicial, o sea la gota. Es el punto de partida
// de toda la simulación: sin ella el agua está quieta y nunca pasa nada. La gota
// es una campana gaussiana centrada en el estanque. Se suma a current (la altura
// en este instante) y una parte se le resta a previous. Como la velocidad se
// calcula como current - previous, la gota no empieza quieta sino con velocidad,
// y eso hace que la onda salga hacia afuera. Se ejecuta una sola vez, antes del
// ciclo principal, y no toca las celdas del borde de la malla.
void add_drop(std::vector<float>& current, std::vector<float>& previous,
              const Config& cfg) {
    const float cx = 0.5f * static_cast<float>(cfg.width - 1);
    const float cy = 0.5f * static_cast<float>(cfg.height - 1);
    const float sigma2 = cfg.drop_radius * cfg.drop_radius;

    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            const float r2 = dx * dx + dy * dy;
            const float pulse =
                cfg.drop_strength * std::exp(-r2 / (2.0f * sigma2));
            const int idx = index_of(x, y, cfg.width);
            current[idx] += pulse;
            previous[idx] -= 0.35f * pulse;
        }
    }
}

// Calcula cuánto amortiguamiento le toca a cada celda según qué tan cerca está
// del borde. Sirve para que las olas no reboten contra el marco de la imagen: sin
// esto, al llegar al borde se reflejarían y el video se vería como una piscina
// pequeña. Lo que hace es una capa absorbente. En el centro devuelve solo el
// amortiguamiento base, pero en los 32 píxeles más cercanos al borde le suma un
// término que crece de forma cuadrática hasta edge_damping. Así la onda va
// perdiendo energía poco a poco antes de llegar al borde. La usa simulate_step
// en cada celda de cada paso.
float border_absorption(int x, int y, const Config& cfg) {
    constexpr int band = 32;
    const int dist = std::min({x, y, cfg.width - 1 - x, cfg.height - 1 - y});
    if (dist >= band) {
        return cfg.damping;
    }

    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return cfg.damping + cfg.edge_damping * t * t;
}

// Esta es la función que hace avanzar la simulación un paso de tiempo, o sea
// donde se actualiza la ecuación de onda. Recibe la malla del paso anterior
// (previous) y la del paso actual (current), y escribe la del siguiente paso en
// next. Para cada celda calcula el laplaciano con sus 4 vecinas (la diferencia
// entre el valor de la celda y el promedio de lo que la rodea), que es lo que
// jala la altura hacia sus vecinas y hace que la onda se propague. Luego le resta
// el amortiguamiento multiplicado por la velocidad. Cada celda solo lee valores
// viejos y escribe en su propia posición de next, así que todas se pueden
// calcular al mismo tiempo, y por eso la queremos pasar a CUDA con un hilo por
// celda. El borde queda siempre en 0 porque next se llena de ceros antes y el
// ciclo no lo toca.
void simulate_step(const std::vector<float>& previous,
                   const std::vector<float>& current, std::vector<float>& next,
                   const Config& cfg) {
    const float c2 = cfg.wave_speed * cfg.wave_speed;

    std::fill(next.begin(), next.end(), 0.0f);

    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const int idx = index_of(x, y, cfg.width);
            const float laplacian =
                current[idx - 1] + current[idx + 1] + current[idx - cfg.width] +
                current[idx + cfg.width] - 4.0f * current[idx];
            const float velocity = current[idx] - previous[idx];
            const float local_damping = border_absorption(x, y, cfg);

            next[idx] = 2.0f * current[idx] - previous[idx] + c2 * laplacian -
                        local_damping * velocity;
        }
    }
}

// Convierte la malla de alturas en la imagen de un cuadro del video. La
// simulación solo produce números, y esta función es la que los vuelve algo que
// se pueda ver. Para cada píxel mira las alturas de los vecinos para saber qué
// tan inclinada está la superficie, con eso saca una normal, y con la normal
// calcula cuánta luz recibe (luz difusa) y un brillo fuerte en las crestas
// (especular). Después lo combina con la altura del punto para obtener el gris.
// Al final escribe encima el número de cuadro. Cada píxel se calcula sin
// depender de los demás, así que también se puede paralelizar. Además hace una
// normalización y un pow por píxel, entonces probablemente pese bastante.
cv::Mat render_frame(const std::vector<float>& height, const Config& cfg,
                     int frame_number) {
    cv::Mat image(cfg.height, cfg.width, CV_8UC3);

    const cv::Vec3f light_dir = cv::normalize(cv::Vec3f(-0.35f, -0.55f, 0.76f));

    for (int y = 0; y < cfg.height; ++y) {
        for (int x = 0; x < cfg.width; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(cfg.width - 1, x + 1);
            const int ym = std::max(0, y - 1);
            const int yp = std::min(cfg.height - 1, y + 1);

            const float dx = height[index_of(xm, y, cfg.width)] -
                             height[index_of(xp, y, cfg.width)];
            const float dy = height[index_of(x, ym, cfg.width)] -
                             height[index_of(x, yp, cfg.width)];
            const cv::Vec3f normal =
                cv::normalize(cv::Vec3f(2.8f * dx, 2.8f * dy, 1.0f));

            const float diffuse = std::max(0.0f, normal.dot(light_dir));
            const float wave = std::clamp(
                0.5f + 1.8f * height[index_of(x, y, cfg.width)], 0.0f, 1.0f);
            const float specular = std::pow(std::max(0.0f, diffuse), 24.0f);

            float intensity = 35.0f + 120.0f * wave;
            intensity *= 0.60f + 0.65f * diffuse;
            intensity += 130.0f * specular;

            const auto gray =
                static_cast<unsigned char>(std::clamp(intensity, 0.0f, 255.0f));
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(gray, gray, gray);
        }
    }

    cv::putText(image, "CPU float32 | frame " + std::to_string(frame_number),
                cv::Point(18, 32), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(235, 235, 235), 1, cv::LINE_AA);

    return image;
}

}  // namespace

// Función principal: arma todo y controla el ciclo de simulación. Primero crea
// la carpeta de salida, reserva las tres mallas que se necesitan (previous,
// current y next, que son los instantes t-1, t y t+1), pone la gota con add_drop
// y abre el escritor de video. Después empieza el ciclo principal, que por cada
// cuadro hace tres cosas: avanza la simulación con simulate_step (y rota los
// buffers con swap para no copiar datos), dibuja el estado con render_frame y
// escribe esa imagen en el video. Mientras tanto va imprimiendo el avance cada
// 10 %. Al terminar calcula las métricas: tiempo, pasos simulados y pasos por
// segundo. El tiempo incluye la simulación, el render y la escritura del video,
// pero no incluye la reserva de memoria ni add_drop.
int main() {
    try {
        const Config cfg;
        const int total_frames =
            static_cast<int>(std::round(cfg.seconds * cfg.fps));
        const std::size_t cells = static_cast<std::size_t>(cfg.width) *
                                  static_cast<std::size_t>(cfg.height);

        std::experimental::filesystem::path output_path(cfg.output);
        if (output_path.has_parent_path()) {
            std::experimental::filesystem::create_directories(
                output_path.parent_path());
        }

        std::vector<float> previous(cells, 0.0f);
        std::vector<float> current(cells, 0.0f);
        std::vector<float> next(cells, 0.0f);

        add_drop(current, previous, cfg);

        cv::VideoWriter writer(
            cfg.output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            static_cast<double>(cfg.fps), cv::Size(cfg.width, cfg.height));

        if (!writer.isOpened()) {
            throw std::runtime_error("No se pudo abrir el archivo de salida: " +
                                     cfg.output);
        }

        const auto start = std::chrono::steady_clock::now();
        double time_simulation = 0.0;
        double time_render = 0.0;
        double time_video = 0.0;

        for (int frame = 0; frame < total_frames; ++frame) {
            for (int step = 0; step < cfg.steps_per_frame; ++step) {
                const auto t0 = std::chrono::steady_clock::now();

                simulate_step(previous, current, next, cfg);

                const auto t1 = std::chrono::steady_clock::now();
                time_simulation +=
                    std::chrono::duration<double>(t1 - t0).count();

                previous.swap(current);
                current.swap(next);
            }

            const auto t2 = std::chrono::steady_clock::now();

            cv::Mat image = render_frame(current, cfg, frame);

            const auto t3 = std::chrono::steady_clock::now();

            time_render += std::chrono::duration<double>(t3 - t2).count();

            const auto t4 = std::chrono::steady_clock::now();

            writer.write(image);

            const auto t5 = std::chrono::steady_clock::now();

            time_video += std::chrono::duration<double>(t5 - t4).count();

            if (frame % std::max(1, total_frames / 10) == 0) {
                std::cout << "Frame " << frame << " / " << total_frames << '\n';
            }
        }

        const auto end = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double>(end - start).count();
        const double simulated_steps =
            static_cast<double>(total_frames) * cfg.steps_per_frame;

        std::cout << "Video generado: " << cfg.output << '\n';
        std::cout << "Tiempo: " << elapsed << " s\n";
        std::cout << "Pasos simulados: " << simulated_steps << '\n';
        std::cout << "Rendimiento: " << simulated_steps / elapsed
                  << " pasos/s\n";

        std::cout << "\n===== PERFILADO CPU =====\n";

        std::cout << "Simulacion: " << time_simulation << " s ("
                  << 100.0 * time_simulation / elapsed << "%)\n";

        std::cout << "Renderizado: " << time_render << " s ("
                  << 100.0 * time_render / elapsed << "%)\n";

        std::cout << "Escritura video: " << time_video << " s ("
                  << 100.0 * time_video / elapsed << "%)\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}