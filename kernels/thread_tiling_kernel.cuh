#pragma once

template <int blockM, int blockN, int blockK, int threadM, int threadN>
__global__ void threadTiling(const float *a, const float *b, float *c, int M,
                             int N, int K) {

  // block's coordinates in the output matrix C
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  // thread's unique 2D coordinate within the thread block
  const int threadRow = threadIdx.y;
  const int threadCol = threadIdx.x;

  // shared memory for the tiles
  __shared__ float sharedA[blockM * blockK];
  __shared__ float sharedB[blockK * blockN];

  // registers for thread-local results and computation
  float threadResults[threadM][threadN] = {{0.0f}};
  float regA[threadM];
  float regB[threadN];
  // main loop that iterates through K
  for (int blockIdx = 0; blockIdx < K; blockIdx += blockK) {

    const int numThreads = blockDim.x * blockDim.y;

    // load the A tile
    for (int i = 0; i < (blockM * blockK) / numThreads; ++i) {
      int flatIdx = i * numThreads + (threadRow * blockDim.x + threadCol);
      int loadRow = flatIdx / blockK;
      int loadCol = flatIdx % blockK;
      int globalRow = cRow * blockM + loadRow;
      int globalCol = blockIdx + loadCol;

      if (globalRow < M && globalCol < K) {
        sharedA[loadRow * blockK + loadCol] = a[globalRow * K + globalCol];
      } else {
        sharedA[loadRow * blockK + loadCol] = 0.0f;
      }
    }

    // load the B tile
    for (int i = 0; i < (blockK * blockN) / numThreads; ++i) {
      int flatIdx = i * numThreads + (threadRow * blockDim.x + threadCol);
      int loadRow = flatIdx / blockN;
      int loadCol = flatIdx % blockN;
      int globalRow = blockIdx + loadRow;
      int globalCol = cCol * blockN + loadCol;

      if (globalRow < K && globalCol < N) {
        sharedB[loadRow * blockN + loadCol] = b[globalRow * N + globalCol];
      } else {
        sharedB[loadRow * blockN + loadCol] = 0.0f;
      }
    }

    __syncthreads();
    for (int i = 0; i < blockK; i++) {
      // load a sliver of the A tile from shared memory into registers
      for (int idxM = 0; idxM < threadM; idxM++) {
        regA[idxM] = sharedA[(threadRow * threadM + idxM) * blockK + i];
      }

      // load a sliver of the B tile from shared memory into registers
      for (int idxN = 0; idxN < threadN; idxN++) {
        regB[idxN] = sharedB[i * blockN + (threadCol * threadN + idxN)];
      }

      // perform the outer product on the registers and accumulate results
      for (int resM = 0; resM < threadM; resM++) {
        for (int resN = 0; resN < threadN; resN++) {
          threadResults[resM][resN] += regA[resM] * regB[resN];
        }
      }
    }
    __syncthreads();
  }

  for (int resM = 0; resM < threadM; resM++) {
    for (int resN = 0; resN < threadN; resN++) {
      int globalRow = cRow * blockM + threadRow * threadM + resM;
      int globalCol = cCol * blockN + threadCol * threadN + resN;

      if (globalRow < M && globalCol < N) {
        // Assuming alpha=1 and beta=0 for simplicity, like in your code
        c[globalRow * N + globalCol] = threadResults[resM][resN];
      }
    }
  }
}
