

#include <cuda_runtime.h>

void __global__ eqCheck(const float *truth, const float *test, int rows,
                        int cols, int *mismatchFlag) {
  int cRow = blockDim.x * blockIdx.x + threadIdx.x;
  int cCol = blockDim.y * blockIdx.y + threadIdx.y;

  if (cCol < cols && cRow < rows) {
    if (fabs(truth[cRow * cols + cCol] - test[cRow * cols + cCol]) > 0.01f) {
      atomicExch(mismatchFlag, 1);
    }
  }
}
