#pragma once

class GEMM {
public:
  // utility functions
  GEMM(const float *a, const float *b, float *c, const float *ref, int M, int N,
       int K, const int block_size)
      : a(a), b(b), c(c), M(M), N(N), K(K), block_size(block_size), ref(ref) {}
  ~GEMM();
  void eq_check(const float *truth, const float *test, int row, int col);
  void run_tests();
  void print_matrix(const float *matrix, int row, int col);
  // implementations prototypes
  // NAIVE
  void naive_kernel(const float *a, const float *b, float *c, int M, int N,
                    int K, int block_size);
  // COALESCE
  void mem_coalesce_kernel(const float *a, const float *b, float *c, int M,
                           int N, int K, int block_size);
  // SHARED_MEM
  void shared_mem_kernel(const float *a, const float *b, float *c, int M, int N,
                         int K, int block_size);
  // TILING
  void tiling_kernel(const float *a, const float *b, float *c, int M, int N,
                     int K, int block_size);
  // INCREASING ARITHMETIC INTENSITY IN 2D TILING
  void tiling_kernel_v2(const float *a, const float *b, float *c, int M, int N,
                        int K, int block_size);

private:
  // M is A's row and N is B's Col and K is A's col as well as B's rows
  int M, N, K;
  const float *a;
  const float *b;
  float *c;
  const float *ref;
  const int block_size;
};
