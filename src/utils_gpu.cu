#include "utils_gpu.cuh"

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
    cudaError_t err = cudaGetLastError();
    std::cerr << msg << " (" << fname << ", line: " << line << ")\n";
    std::cerr << "Error " << cudaGetErrorName(err) << ": "
        << cudaGetErrorString(err) << "\n";
    std::exit(1);
}

__device__ unsigned char get(unsigned char patch[TILE_SIZE][TILE_SIZE], int i,
        int j)
{
    return (i < 0 || j < 0 || i >= TILE_SIZE || j >= TILE_SIZE)
        ? 0 : patch[j][i];
}

__device__ unsigned char get_texton(unsigned char patch[TILE_SIZE][TILE_SIZE],
        int i, int j)
{
    unsigned char value = patch[j][i];

    unsigned char texton = get(patch, i - 1, j - 1) >= value;
    texton <<= 1;

    texton |= get(patch, i - 1, j) >= value;
    texton <<= 1;

    texton |= get(patch, i - 1, j + 1) >= value;
    texton <<= 1;

    texton |= get(patch, i, j + 1) >= value;
    texton <<= 1;

    texton |= get(patch, i + 1, j + 1) >= value;
    texton <<= 1;

    texton |= get(patch, i + 1, j) >= value;
    texton <<= 1;

    texton |= get(patch, i + 1, j - 1) >= value;
    texton <<= 1;

    texton |= get(patch, i, j - 1) >= value;

    return texton;
}

