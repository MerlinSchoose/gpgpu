#pragma once

unsigned char *render_gpu(unsigned width, unsigned height, unsigned char *histos,
                          size_t histos_pitch, float *centers, unsigned char *rand_colors);