#include "GEMM.cuh"
#include "utils.cuh"
#include <err.h>
#include <format>
#include <fstream>
#include <iostream>

void write_matrix(const float *matrix, int rows, int cols,
                  std::string fileName) {
  std::ofstream file(fileName);

  if (!file.is_open()) {
    std::cerr << "Error: file '" << fileName << "' could not be opened."
              << std::endl;
    return;
  }

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      file << matrix[i * cols + j] << " ";
    }
    file << "\n";
  }
  file.close();
}

void read_file(float *matrix, int rows, int cols, std::string file_name) {
  std::ifstream file(file_name);

  if (!file.is_open()) {
    std::cerr << "Error: Could not open file '" << file_name << "'"
              << std::endl;
    return;
  }
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (!(file >> matrix[i * cols + j])) {
        std::cerr << "Error: Unexpected end of file or invalid data."
                  << std::endl;
        // You can handle this error more gracefully, e.g., by returning a
        // status code.
        return;
      }
    }
  }
  file.close();
}

void initialize_matrices(int M, int K, int N, float *&h_A, float *&h_B,
                         float *&h_C) {

  for (int i = 0; i < (M * K); ++i) {
    h_A[i] = static_cast<float>(i % 100) * 0.1f;
  }

  for (int i = 0; i < (K * N); ++i) {
    h_B[i] = static_cast<float>(i % 120) * -0.01f;
  }

  for (int i = 0; i < (M * N); ++i) {
    h_C[i] = 0.0f;
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
  const int M = 4096;
  const int K = 2048;
  const int N = 4096;

  float *h_A, *h_B, *h_C;
  h_A = new float[M * K];
  h_B = new float[K * N];
  h_C = new float[M * N];
  float *output_ref = new float[M * N];

  // ONLY DO THIS ON FIRST RUN
  initialize_matrices(M, K, N, h_A, h_B, h_C);
  write_matrix(h_A, M, K, "A_matrix.txt");
  write_matrix(h_B, K, N, "B_matrix.txt");
  gemm_cpu_reference(M, K, N, h_A, h_B, output_ref);
  write_matrix(output_ref, M, N, "C_matrix.txt");

  read_file(h_A, M, K, "A_matrix.txt");
  read_file(h_B, K, N, "B_matrix.txt");
  read_file(output_ref, M, N, "C_matrix.txt");
  // create a GEMM object
  GEMM *gemm = new GEMM(h_A, h_B, h_C, output_ref, M, N, K, BLOCK_SIZE);
  gemm->run_tests();

  delete gemm;
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] output_ref;
  return 0;
}
