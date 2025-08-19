#pragma once

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
