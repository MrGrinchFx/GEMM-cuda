#pragma once

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
