#pragma once

class GEMM {
public:
  // utility functions
  GEMM(float *a, float *b, float *c, int size, int rows, int cols)
      : a(a), b(b), c(c), size(size), rows(rows), cols(cols) {}
  void eq_check(const float *truth, const float *test);
  void run_tests(int kernel);

  // implementations prototypes
  // NAIVE
  void naive_kernel();
  // COALESCE
  void mem_coalesce_kernel();
  // SHARED_MEM
  void shared_mem_kernel();
  // TILING
  void tiling_kernel();
  // INCREASING ARITHMETIC INTENSITY IN 2D TILING
  void tiling_kernel_v2();

private:
  const float *a;
  const float *b;
  float *c;
  int size;
  int cols;
  int rows;
};
