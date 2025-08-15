#pragma once
#include <cuda_runtime.h>

void __global__ eqCheck(const float *truth, const float *test, int rows,
                        int cols, int *mismatchFlag);
void __global__ naiveKernel(const float *a, const float *b, float *c, int M,
                            int N, int K);
// void __global__ memCoalesce(const float *a, const float *b, float *c, int M,
//                          int N, int K);

void __global__ sharedMem(const float *a, const float *b, float *c, int M,
                          int N, int K);

void __global__ tiling2D(const float *a, const float *b, float *c, int M, int N,
                         int K);

void __global__ tiling2D_V2(const float *a, const float *b, float *c, int M,
                            int N, int K);
