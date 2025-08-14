#include "GEMM.cuh"

#include <iostream>
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#define BLOCK_SIZE 256;

void initialize_matrices(int M, int K, int N, float *&h_A, float *&h_B,
                         float *&h_C) {
  h_A = new float[M * K];
  h_B = new float[K * N];
  h_C = new float[M * N];

  for (int i = 0; i < M * K; ++i) {
    h_A[i] = static_cast<float>(i % 100) * 0.1f;
  }

  for (int i = 0; i < K * N; ++i) {
    h_B[i] = static_cast<float>(i % 120) * -0.01f;
  }

  for (int i = 0; i < M * N; ++i) {
    h_C[i] = 0.0f;
  }
}
void print_matrix(const float *matrix, int row, int col) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::cout << matrix[row * i + j] << " ";
    }
    std::cout << "\n";
  }
}
void gemm_cpu_reference(int M, int K, int N, const float *h_A, const float *h_B,
                        float *h_C_ref) {

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float dot_product = 0.0f;

      for (int k = 0; k < K; ++k) {
        float a_val = h_A[i * K + k];

        float b_val = h_B[k * N + j];

        dot_product += a_val * b_val;
      }

      h_C_ref[i * N + j] = dot_product;
    }
  }
}

int main() {
  const int M = 256;
  const int K = 512;
  const int N = 128;

  float *h_A, *h_B, *h_C;

  initialize_matrices(M, K, N, h_A, h_B, h_C);
  float *output_ref = new float[M * N];

  gemm_cpu_reference(M, K, N, h_A, h_B, output_ref);

  GEMM *gemm = new GEMM();
  gemm->eq_check(output_ref, output_ref, M, N);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}
