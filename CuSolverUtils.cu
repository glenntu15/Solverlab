
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include "Timers.h"

#include <stdio.h>

__global__ void TransposeKernel3(const double* input, double* output, int n) {
    // Here we alter the 64x64 array to allow an offset for better memory bank access.
    // In addition, the loops are unrolled.  
    // (Small loops, a lot of overhead for the work they do.)

    __shared__ double data[65][66];    // 64*64 expanded 

    int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    //int end_j = j + 4;

    int it = threadIdx.x;
    int jt = 4 * threadIdx.y;

    data[it][jt] = input[i + n * j];
    data[it][jt + 1] = input[i + n * (j + 1)];
    data[it][jt + 2] = input[i + n * (j + 2)];
    data[it][jt + 3] = input[i + n * (j + 3)];

    //
    __syncthreads();        // make sure "data" is filled

    i = threadIdx.x + 64 * blockIdx.y;
    j = 4 * threadIdx.y + 64 * blockIdx.x;

    jt = 4 * threadIdx.y;
    output[i + n * j] = data[jt][it];
    output[i + n * (j + 1)] = data[jt + 1][it];
    output[i + n * (j + 2)] = data[jt + 2][it];
    output[i + n * (j + 3)] = data[jt + 3][it];

}
void TranposeMatrix(const double* d_input, double* d_output, int n)
{
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    //naiveTransposeKernel << <gridSize, blockSize >> > (dev_a, dev_o, n);
    TransposeKernel3 << <gridSize, blockSize >> > (d_input, d_output, n);
    checkCudaErrors(cudaDeviceSynchronize());
}