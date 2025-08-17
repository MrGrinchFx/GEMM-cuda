
#include "GEMM_kernels.cuh"
#include <cuda_runtime.h>

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

void __global__ naiveKernel(const float *a, const float *b, float *c, int M,
                            int N, int K) {
  int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
  int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

  float result = 0.0f;
  if (xIdx < M && yIdx < N) {
    for (int i = 0; i < K; i++) {
      result += a[xIdx * K + i] * b[N * i + yIdx];
    }

    c[xIdx * N + yIdx] = result;
  }
}

void __global__ memCoalesce(const float *a, const float *b, float *c, int M,
                            int N, int K) {
  // instead of the other method, we do this so that threads in the same warp
  // access the same row in A and compute with different columns of B.
  int yIdx = blockDim.x * blockIdx.x + threadIdx.x;
  int xIdx = blockDim.y * blockIdx.y + threadIdx.y;

  float result = 0.0f;
  if (xIdx < M && yIdx < N) {
    for (int i = 0; i < K; i++) {
      result += a[xIdx * K + i] * b[N * i + yIdx];
    }

    c[xIdx * N + yIdx] = result;
  }
}

void __global__ sharedMem(const float *a, const float *b, float *c, int M,
                          int N, int K) {
  const int tileSize = 32;
  __shared__ float sA[tileSize * tileSize];
  __shared__ float sB[tileSize * tileSize];
  // to coallesce memory accesses
  int yChunkIdx = blockIdx.x;
  int xChunkIdx = blockIdx.y;

  // get to starting position
  a += yChunkIdx * K * tileSize;
  b += xChunkIdx * tileSize;
  c += yChunkIdx * N * tileSize + xChunkIdx * tileSize;

  int innerCol = threadIdx.x % tileSize;
  int innerRow = threadIdx.x / tileSize;
  float temp = 0.0;
  for (int blkIdx = 0; blkIdx < K; blkIdx += tileSize) {
    // copy from main matrix to sharedMem
    sA[innerRow * tileSize + innerCol] = a[innerRow * K + innerCol];
    sB[innerRow * tileSize + innerCol] = b[innerRow * N + innerCol];
    // wait for shared mem to be populated before continuing.
    __syncthreads();

    a += tileSize;
    b += tileSize * N;

    // perform matrix multiplication from sharedMem
    for (int innerIdx = 0; innerIdx < tileSize; ++innerIdx) {
      temp += sA[innerRow * tileSize + innerIdx] *
              sB[innerIdx * tileSize + innerCol];
    }
    // wait for all threads to finish before moving on
    __syncthreads();
  }
  // load from shared to output matrix
  c[innerRow * N + innerCol] += temp;
}

void __global__ tiling2D(const float *a, const float *b, float *c, int size) {
  // TODO
}

void __global__ tiling2D_V2(const float *a, const float *b, float *c,
                            int size) {
  // TODO
}
