#include <cuda_runtime.h>

void __global__ memCoalesce(const float *a, const float *b, float *c, int size);

void __global__ sharedMem(const float *a, const float *b, float *c, int size);

void __global__ tiling2D(const float *a, const float *b, float *c, int size);

void __global__ tiling2D_V2(const float *a, const float *b, float *c, int size);
