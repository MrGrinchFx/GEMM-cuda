#include "GEMM.cuh"
#include "GEMM_kernels.cuh"
#include "utils.cuh"

GEMM::~GEMM() {}

void GEMM::print_matrix(const float *matrix, int row, int col) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::cout << matrix[col * i + j] << " ";
    }
    std::cout << "\n";
  }
}
void GEMM::run_tests() {
  // run helper functions to launch kernels
  naive_kernel(this->a, this->b, this->c, this->M, this->N, this->K,
               this->block_size);
  eq_check(this->c, this->ref, this->M, this->N);
  // mem_coalesce_kernel(this->a, this->b, this->c, this->M, this->N, this->K,
  //                     this->block_size);
  // shared_mem_kernel(this->a, this->b, this->c, this->M, this->N, this->K,
  //                   this->block_size);
  // tiling_kernel(this->a, this->b, this->c, this->M, this->N, this->K,
  //               this->block_size);
  // tiling_kernel_v2(this->a, this->b, this->c, this->M, this->N, this->K,
  //                  this->block_size);
}

void GEMM::eq_check(const float *truth, const float *test, int row, int col) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (abs(truth[col * i + j] - test[col * i + j]) > 0.01) {
        std::cout << "THEY DO NOT MATCH\n";
        return;
      }
    }
  }
  std::cout << "YAYYY they match!\n";
}

void GEMM::naive_kernel(const float *a, const float *b, float *c, int M, int N,
                        int K, int block_size) {
  float *d_a;
  float *d_b;
  float *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, sizeof(float) * M * K));
  CUDA_CHECK(cudaMalloc(&d_b, sizeof(float) * N * K));
  CUDA_CHECK(cudaMalloc(&d_c, sizeof(float) * M * N));

  CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(float) * N * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, sizeof(float) * M * N));
  int grid_size = (M * N + block_size - 1) / block_size;
  // each thread will be responsible for a single output cell
  naiveKernel<<<grid_size, block_size>>>(d_a, d_b, d_c, M, N, K);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}

void GEMM::mem_coalesce_kernel(const float *a, const float *b, float *c, int M,
                               int N, int K, int block_size) {
  // TODO
}

void GEMM::shared_mem_kernel(const float *a, const float *b, float *c, int M,
                             int N, int K, int block_size) {
  // TODO
}

void GEMM::tiling_kernel(const float *a, const float *b, float *c, int M, int N,
                         int K, int block_size) {
  // TODO
}

void GEMM::tiling_kernel_v2(const float *a, const float *b, float *c, int M,
                            int N, int K, int block_size) {
  // TODO
}
