#include "GEMM.cuh"
#include "kernels/GEMM_kernels.cuh"
#include "utils.cuh"
#include <format>
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
  eq_check(this->c, this->ref, this->M, this->N, "Naive Kernel");
  mem_coalesce_kernel(this->a, this->b, this->c, this->M, this->N, this->K);
  eq_check(this->c, this->ref, this->M, this->N, "Coalesced Kernel");
  shared_mem_kernel(this->a, this->b, this->c, this->M, this->N, this->K);
  eq_check(this->c, this->ref, this->M, this->N, "Shared Mem Kernel");

  tiling_kernel(this->a, this->b, this->c, this->M, this->N, this->K);
  eq_check(this->c, this->ref, this->M, this->N, "Thread Tiling Kernel");
  //  tiling_kernel_v2(this->a, this->b, this->c, this->M, this->N,
  //  this->K);
}

void GEMM::eq_check(const float *truth, const float *test, int row, int col,
                    std::string kernelName) {
  float *d_truth;
  float *d_test;
  int *d_mismatch_flag;
  CUDA_CHECK(cudaMalloc(&d_truth, sizeof(float) * row * col));
  CUDA_CHECK(cudaMalloc(&d_test, sizeof(float) * row * col));
  CUDA_CHECK(cudaMalloc(&d_mismatch_flag, sizeof(int)));

  int mismatch_flag = 0;
  CUDA_CHECK(cudaMemcpy(d_mismatch_flag, &mismatch_flag, sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_truth, truth, sizeof(float) * row * col,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_test, test, sizeof(float) * row * col,
                        cudaMemcpyHostToDevice));

  dim3 block_dim{32, 32, 1};
  dim3 grid_dim{(row + block_dim.x - 1) / block_dim.x,
                (col + block_dim.y - 1) / block_dim.y, 1};
  eqCheck<<<grid_dim, block_dim>>>(d_truth, d_test, row, col, d_mismatch_flag);

  CUDA_CHECK(cudaMemcpy(&mismatch_flag, d_mismatch_flag, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  if (mismatch_flag == 1) {
    std::cout << std::format("{} is not correct\n", kernelName);
  } else {
    std::cout << std::format("YAYYY! {} is correct\n", kernelName);
  }
  CUDA_CHECK(cudaFree(d_mismatch_flag));
  CUDA_CHECK(cudaFree(d_truth));
  CUDA_CHECK(cudaFree(d_test));
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
  dim3 block_dim{32, 32, 1};
  dim3 grid_dim{(M + block_dim.x - 1) / block_dim.x,
                (N + block_dim.y - 1) / block_dim.y, 1};
  // each thread will be responsible for a single output cell
  naiveKernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N, K);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}

void GEMM::mem_coalesce_kernel(const float *a, const float *b, float *c, int M,
                               int N, int K) {
  float *d_a, *d_b, *d_c;

  CUDA_CHECK(cudaMalloc(&d_a, sizeof(float) * M * K));
  CUDA_CHECK(cudaMalloc(&d_b, sizeof(float) * N * K));
  CUDA_CHECK(cudaMalloc(&d_c, sizeof(float) * M * N));

  CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(float) * N * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, sizeof(float) * M * N));
  dim3 block_dim{32, 32, 1};
  dim3 grid_dim{(M + block_dim.x - 1) / block_dim.x,
                (N + block_dim.y - 1) / block_dim.y, 1};
  memCoalesce<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N, K);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}

void GEMM::shared_mem_kernel(const float *a, const float *b, float *c, int M,
                             int N, int K) {
  float *d_a, *d_b, *d_c;
  unsigned int tile_size = 32;

  CUDA_CHECK(cudaMalloc(&d_a, sizeof(float) * M * K));
  CUDA_CHECK(cudaMalloc(&d_b, sizeof(float) * N * K));
  CUDA_CHECK(cudaMalloc(&d_c, sizeof(float) * M * N));

  CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(float) * N * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, sizeof(float) * M * N));
  dim3 block_dim{tile_size, tile_size, 1};
  dim3 grid_dim{(M + tile_size) / tile_size, (N + tile_size - 1) / tile_size,
                1};

  sharedMem<32><<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}

void GEMM::tiling_kernel(const float *a, const float *b, float *c, int M, int N,
                         int K) {
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, sizeof(float) * M * K));
  CUDA_CHECK(cudaMalloc(&d_b, sizeof(float) * N * K));
  CUDA_CHECK(cudaMalloc(&d_c, sizeof(float) * M * N));

  CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(float) * N * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, sizeof(float) * M * N));

  const int block_n = 128;
  const int block_m = 128;
  const int block_k = 16;

  const int thread_n = 8;
  const int thread_m = 8;

  // there will be a line of threads that are responsible for a small block. so
  // we only need a 1D block of threads.
  dim3 block_dim(16, 16, 1);
  // grid will be made up of a grid of blocks that are responsible for a portion
  // of C.

  dim3 grid_dim{(static_cast<unsigned int>(N) + block_n - 1) / block_n,
                (static_cast<unsigned int>(M) + block_m - 1) / block_m, 1};
  threadTiling<block_m, block_n, block_k, thread_m, thread_n>
      <<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
}

void GEMM::tiling_kernel_v2(const float *a, const float *b, float *c, int M,
                            int N, int K) {
  // TODO
}
