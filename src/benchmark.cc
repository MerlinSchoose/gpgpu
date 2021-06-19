#include <vector>
#include <benchmark/benchmark.h>

#include "cpu_lbp.hh"
#include "gpu_lbp.hh"
#include "utils.hh"

std::string inputfilename = "../data/barcode-00-01.jpg";

void BM_Rendering_cpu(benchmark::State& st)
{
    cv::Mat image = cv::imread(inputfilename, cv::IMREAD_GRAYSCALE);

    for (auto _ : st)
        cpu_lbp(image);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(),
            benchmark::Counter::kIsRate);
}

void BM_Rendering_gpu(benchmark::State& st)
{
    cv::Mat image = cv::imread(inputfilename, cv::IMREAD_GRAYSCALE);

    size_t rows = ((image.cols + TILE_SIZE - 1) / TILE_SIZE)
        * ((image.rows + TILE_SIZE - 1) / TILE_SIZE);
    size_t cols = HISTO_SIZE;

    unsigned char *histos_buffer = (unsigned char *)
        malloc(rows * cols * sizeof(unsigned char));

    for (auto _ : st)
        gpu_lbp(mat_to_bytes(image), image.cols, image.rows, histos_buffer);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(),
            benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Rendering_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Rendering_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_MAIN();
