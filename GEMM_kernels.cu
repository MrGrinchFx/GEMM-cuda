
#include "GEMM_kernels.cuh"
#include <cuda_runtime.h>
void __global__ naiveKernel(const float *a, const float *b, float *c, int M,
                            int N, int K) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int row = idx / N;
  int col = idx % N;
  float result = 0.0f;
  if (idx < (M * N)) {
    for (int i = 0; i < K; i++) {
      result += a[row * K + i] * b[N * i + col];
    }

    c[row * N + col] = result;
  }
}

void __global__ memCoalesce(const float *a, const float *b, float *c,
                            int size) {
  // TODO
}

void __global__ sharedMem(const float *a, const float *b, float *c, int size) {
  // TODO
}

void __global__ tiling2D(const float *a, const float *b, float *c, int size) {
  // TODO
}

void __global__ tiling2D_V2(const float *a, const float *b, float *c,
                            int size) {
  // TODO
}
