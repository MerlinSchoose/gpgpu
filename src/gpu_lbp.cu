// #include <spdlog/spdlog.h>

#include "gpu_lbp.hh"

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

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

__global__ void kernel(unsigned char* image, int width, int height,
        size_t pitch, unsigned char *histos_buffer, size_t pitch_buffer) {
    __shared__ unsigned char tile[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned char textonz[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned histo[HISTO_SIZE];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (size_t i = 0; i < HISTO_SIZE; ++i)
            histo[i] = 0;
    }

__global__ void kernel(unsigned char* buffer, int width, int height,
        size_t pitch) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height || x % 16 != 0 || y % 16 != 0)
        return;

    unsigned char color = (x + y) % 255;

    for(int i = x; i < x + TILE_SIZE; ++i)
    {
        for(int j = y; j < y + TILE_SIZE; ++j)
        {
            unsigned char *lineptr = (unsigned char *) (buffer + j * pitch);
            lineptr[i] = color;
        }
    }
}

void gpu_lbp(unsigned char *image, int image_cols, int image_rows,
        unsigned char *image_buffer) {
    cudaError_t error = cudaSuccess;

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((image_cols + threads.x - 1) / threads.x,
            (image_rows + threads.y - 1) / threads.y);

    size_t pitch_image;
    size_t pitch_histos;
    unsigned char *dev_image;
    unsigned char *dev_histos;

    error = cudaMallocPitch(&dev_histos, &pitch_histos,
            HISTO_SIZE, blocks.x * blocks.y * sizeof(unsigned char));

    if (error)
        abortError("Fail buffer allocation");

    error = cudaMallocPitch(&dev_image, &pitch_image,
            image_cols * sizeof(unsigned char), image_rows);

    if (error)
        abortError("Fail buffer allocation");

    error = cudaMemcpy2D(dev_image, pitch_image, image,
            image_cols * sizeof(unsigned char),
            image_cols * sizeof(unsigned char), image_rows,
            cudaMemcpyHostToDevice);

    if (error)
        abortError("Fail buffer copy to device");

    kernel<<<blocks, threads>>>(dev_image, image_cols, image_rows, pitch_image,
            dev_histos, pitch_histos);

    if (cudaPeekAtLastError())
        abortError("Computation Error");

    error = cudaMemcpy2D(histos_buffer, HISTO_SIZE, dev_histos,
            pitch_histos, HISTO_SIZE,
            blocks.x * blocks.y * sizeof(unsigned char),
            cudaMemcpyDeviceToHost);

    if (error)
        abortError("Unable to copy buffer back to memory");

    error = cudaFree(dev_image);
    error = cudaFree(dev_histos);

    if (error)
        abortError("Unable to free memory");
}
