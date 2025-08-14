#pragma once

class GEMM {
public:
  // utility functions
  void eq_check(const float *truth, const float *test, int row, int col);
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
};
