#include "GEMM.cuh"
#include "GEMM_kernels.cuh"
#include <iostream>
void GEMM::run_tests(int kernel) {
  switch (kernel) {

  case 0:
    // naive kernel
    naive_kernel();
    break;

  case 1:
    // coalesce
    mem_coalesce_kernel();
    break;

  case 2:
    // shared memory
    shared_mem_kernel();
    break;

  case 3:
    // Tiling
    tiling_kernel();
    break;

  case 4:
    // Tiling V2
    tiling_kernel_v2();
    break;
  }
}

void GEMM::eq_check(const float *truth, const float *test, int row, int col) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (truth[row * i + j] != test[row * i + j]) {
        printf("THEY DO NOT MATCH\n");
      }
    }
  }

  printf("YAYYY\n");
}

void GEMM::naive_kernel() {
  // TODO
}

void GEMM::mem_coalesce_kernel() {
  // TODO
}

void GEMM::shared_mem_kernel() {
  // TODO
}

void GEMM::tiling_kernel() {
  // TODO
}

void GEMM::tiling_kernel_v2() {
  // TODO
}
