// #include "GEMM.cu"
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
  // Allocate memory for the matrices on the host
  h_A = new float[M * K];
  h_B = new float[K * N];
  h_C = new float[M * N];

  // --- Initialize matrices A and B with a predictable arithmetic sequence ---

  for (int i = 0; i < M * K; ++i) {
    // Simple pattern: 0.0, 0.1, 0.2, ...
    h_A[i] = static_cast<float>(i % 100) * 0.1f;
  }

  for (int i = 0; i < K * N; ++i) {
    // A different simple pattern to avoid all matrices being identical
    h_B[i] = static_cast<float>(i % 120) * -0.01f;
  }

  // --- Initialize matrix C to all zeros ---
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
  std::cout << "Calculating reference GEMM result on CPU..." << std::endl;

  // Iterate over each row of matrix C (and A)
  for (int i = 0; i < M; ++i) {
    // Iterate over each column of matrix C (and B)
    for (int j = 0; j < N; ++j) {
      float dot_product = 0.0f;

      // Compute the dot product of row i from A and column j from B
      for (int k = 0; k < K; ++k) {
        // Get A(i, k)
        float a_val = h_A[i * K + k];

        // Get B(k, j)
        float b_val = h_B[k * N + j];

        dot_product += a_val * b_val;
      }

      // Store the result in C(i, j)
      h_C_ref[i * N + j] = dot_product;
    }
  }
  std::cout << "CPU reference calculation complete." << std::endl;
}

int main() {
  // --- Define matrix dimensions ---
  // Using sizes that are often multiples of 16 or 32 is good for GPU testing
  const int M = 256;
  const int K = 512;
  const int N = 128;

  // --- Host pointers for the matrices ---
  float *h_A, *h_B, *h_C;

  // --- Generate the matrices ---
  initialize_matrices(M, K, N, h_A, h_B, h_C);
  float *output_ref = new float[M * N];
  // --- Verification (Optional) ---
  // Print a few elements to confirm they are initialized
  gemm_cpu_reference(M, K, N, h_A, h_B, output_ref);
  print_matrix(output_ref, M, N);
  // --- Cleanup ---
  // Don't forget to free the allocated memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}
