// #include <spdlog/spdlog.h>

#include "gpu_lbp.hh"

#define THREADS_SIZE 32
#define TILE_SIZE 16

/* [[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__) */

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

    size_t pitch;
    error = cudaMallocPitch(&image, &pitch, image_cols * sizeof(unsigned char),
            image_rows);

    /* if (error)
        abortError("Fail buffer allocation"); */

    dim3 threads(THREADS_SIZE, THREADS_SIZE);
    dim3 blocks((image_cols + threads.x - 1) / threads.x,
            (image_rows + threads.y - 1) / threads.y);

    kernel<<<blocks, threads>>>(image, image_cols, image_rows, pitch);

    /* if (cudaPeekAtLastError())
        abortError("Computation Error"); */

    error = cudaMemcpy2D(image_buffer, image_cols * sizeof(unsigned char),
            image, pitch, image_cols * sizeof(unsigned char), image_rows,
            cudaMemcpyDeviceToHost);

    /* if (error)
        abortError("Unable to copy buffer back to memory"); */

    error = cudaFree(image);

    /* if (error)
        abortError("Unable to free memory"); */
}
