#pragma once
#include <iostream>
#include <cassert>

#define HISTO_SIZE 256
#define TILE_SIZE 16



#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line);


__device__ unsigned char get(unsigned char patch[TILE_SIZE][TILE_SIZE], int i,
        int j);

__device__ unsigned char get_texton(unsigned char patch[TILE_SIZE][TILE_SIZE],
        int i, int j);
