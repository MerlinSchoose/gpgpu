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

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

/*void BM_Rendering_gpu(benchmark::State& st)
{
    int stride = width * kRGBASize;
    std::vector<char> data(height * stride);

    for (auto _ : st)
        render(data.data(), width, height, stride, niteration);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}*/

BENCHMARK(BM_Rendering_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

/*BENCHMARK(BM_Rendering_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();*/

BENCHMARK_MAIN();
