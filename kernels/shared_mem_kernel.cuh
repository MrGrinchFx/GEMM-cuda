#pragma once
template <int TILE_SIZE>
__global__ void sharedMem(const float *a, const float *b, float *c, int M,
                          int N, int K) {
  // each thread block computes one tile of the output matrix C.
  // each thread in the block computes one element of the tile.

  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  __shared__ float sB[TILE_SIZE][TILE_SIZE];

  // thread's position within the 2D block.
  int innerRow = threadIdx.y;
  int innerCol = threadIdx.x;

  // identify the tile this thread block is responsible for.
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  int globalRow = blockRow * TILE_SIZE + innerRow;
  int globalCol = blockCol * TILE_SIZE + innerCol;

  float temp = 0.0f;

  // iterate over the tiles of A and B required to compute the C tile.
  for (int i = 0; i < (K + TILE_SIZE - 1) / TILE_SIZE; i++) {

    int aRow = globalRow;
    int aCol = i * TILE_SIZE + innerCol;

    // check bounds to prevent reading outside of matrix A.
    if (aRow < M && aCol < K) {
      sA[innerRow][innerCol] = a[aRow * K + aCol];
    }

    int bRow = i * TILE_SIZE + innerRow;
    int bCol = globalCol;

    if (bRow < K && bCol < N) {
      sB[innerRow][innerCol] = b[bRow * N + bCol];
    }

    // wait for all threads in the block to finish loading their data
    __syncthreads();

    for (int j = 0; j < TILE_SIZE; ++j) {
      temp += sA[innerRow][j] * sB[j][innerCol];
    }

    // wait for all threads to finish their computation before moving on
    __syncthreads();
  }

  // check bounds to prevent writing outside of the output matrix C.
  if (globalRow < M && globalCol < N) {
    c[globalRow * N + globalCol] = temp;
  }
}
