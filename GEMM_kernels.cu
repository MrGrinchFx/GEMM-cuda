
#include "GEMM_kernels.cuh"
#include <cuda_runtime.h>
void __global__ naiveKernel(const float *a, const float *b, float *c, int M,
                            int N, int K) {
  int cRow = blockDim.x * blockIdx.x + threadIdx.x;
  int cCol = blockDim.y * blockIdx.y + threadIdx.y;

  float result = 0.0f;
  if (cRow < M && cCol < N) {
    for (int i = 0; i < K; i++) {
      result += a[cRow * K + i] * b[N * i + cCol];
    }

    c[cRow * N + cCol] = result;
  }
}

void __global__ eqCheck(const float *truth, const float *test, int rows,
                        int cols, int *mismatchFlag) {
  int cRow = blockDim.x * blockIdx.x + threadIdx.x;
  int cCol = blockDim.y * blockIdx.y + threadIdx.y;

  if (cCol < cols && cRow < rows) {
    if (fabs(truth[cRow * cols + cCol] - test[cRow * cols + cCol]) > 0.01f) {
      atomicExch(mismatchFlag, 1);
    }
  }
}

// void __global__ memCoalesce(const float *a, const float *b, float *c,
//                             int size) {
//   // TODO
// }

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
