#include <iostream>
#include <cassert>

#include "gpu_lbp.hh"

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
    cudaError_t err = cudaGetLastError();
    std::cerr << msg << " (" << fname << ", line: " << line << ")\n";
    std::cerr << "Error " << cudaGetErrorName(err) << ": "
        << cudaGetErrorString(err) << "\n";
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
    __shared__ unsigned textonz[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned histo[HISTO_SIZE];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (size_t i = 0; i < HISTO_SIZE; ++i)
            histo[i] = 0;
    }

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    tile[threadIdx.x][threadIdx.y] = image[x + y * pitch];
    __syncthreads();

    assert(threadIdx.x + threadIdx.y * blockDim.x < TILE_SIZE * TILE_SIZE);
    textonz[threadIdx.x + threadIdx.y * blockDim.x] = get_texton(tile,
            threadIdx.y, threadIdx.x);
    __syncthreads();

    atomicInc(&(histo[textonz[threadIdx.x + threadIdx.y * blockDim.x]]), 256);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        unsigned char *lineptr = histos_buffer + (blockIdx.x + blockIdx.y
                * ((width) / blockDim.x)) * pitch_buffer;

        for (size_t i = 0; i < HISTO_SIZE; ++i)
            lineptr[i] = (unsigned char) histo[i];
    }
}

void gpu_lbp(unsigned char *image, int image_cols, int image_rows,
        unsigned char *histos_buffer) {
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
    cudaDeviceSynchronize();

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
