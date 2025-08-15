#include "GEMM.cuh"
#include "utils.cuh"
#include <err.h>
#include <format>
#include <fstream>
#include <iostream>

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

int main() {
  const int M = 4096;
  const int K = 2048;
  const int N = 4096;

  float *h_A, *h_B, *h_C;
  h_A = new float[M * K];
  h_B = new float[K * N];
  h_C = new float[M * N];
  float *output_ref = new float[M * N];

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
