#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include "Timers.h"
//unistd.h for Linux
#include <Windows.h>
#include <stdio.h>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
//----------------------------------------------------------------------------
__global__ void MultKernel1(double* a, double* x, double* b, int N)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;  // one thread per row;
    double temp = 0.0;
    int j;
    if (i < N) {
        int ioff = i * N;
        for (j = 0; j < N; j++)
            temp = temp + a[ioff + j] * x[j];
        b[i] = temp;
    }
    
}
//----------------------------------------------------------------------------
__global__ void MultKernel2(double* a, double* x, double* b, int N)
{
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // one thread per row;
    int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // column

    if (i < N) {
        
        int ioff = i * N;
        b[i] += a[ioff + j] * x[i];
    }

}
/////////////////////////////////////////////////////////////////////////////////////////
//
// MatVect : this kernel will perform actual MatrixVector Multiplication 
//
// **** launched like this...
// int max = BLOCKSIZE * BLOCKSIZE;
// int BlocksPerGrid = matRowSize / max + 1;
// dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
// if (matRowSize % max == 0)BlocksPerGrid--;
// dim3 dimGrid(1, BlocksPerGrid);
// check_block_grid_dim(deviceProp, dimBlock, dimGrid);

// MatVectMultiplication << <dimGrid, dimBlock >> > (device_Mat, device_Vect, matRowSize, vlength, device_ResVect);
//   ...
/////////////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE 16
#define SIZE 1024
#define EPS 1.0e-15
__global__ void MatVectMultiplication(double* device_Mat, double* device_Vect, int matRowSize, int vlength, double* device_ResVect)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int tindex = tidx + gridDim.x * BLOCKSIZE * tidy;


    if (tindex < matRowSize)
    {
        int i; int m = tindex * vlength;
        device_ResVect[tindex] = 0.00;
        for (i = 0; i < vlength; i++)
            device_ResVect[tindex] += device_Mat[m + i] * device_Vect[i];
    }

    __syncthreads();

}//end of MatVect device function
//------------------------------------------------------------------------------
// Taanspose stuff
__global__ void naiveTransposeKernel(const double* input, double* output, int n) 
{
    // TODO: do not modify code, just comment on suboptimal accesses

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];
}
//
__global__ void optimalTransposeKernel(const double* input, double* output, int n) {
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
//
//Cu - cuda, d - double, MatV -- matrix vector multiply
int CudMatMulV(double* a, double* x, double* b, int n)
{
    cudaError_t cudaStatus;

    double* dev_a;
    double* dev_x;
    double* dev_b;
    double start, stop;
    Timers* tp = new Timers();
  
    //start = tp->second();
    //cudaStatus = cudaSetDevice(0);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    //    return 2;
    //}
    //stop = tp->second();
    //printf(" time to set device %.4f\n", (stop - start));

    //***
    // Try call to cudafree to create context (whatever context is)
    //cudaFree(0);

   
    start = tp->second();
    
    checkCudaErrors(cudaMalloc((void**)&dev_a, n * n * sizeof(double)));

   
    checkCudaErrors(cudaMalloc((void**)&dev_x, n * sizeof(double)));

   
    checkCudaErrors(cudaMalloc((void**)&dev_b, n * sizeof(double)));
    stop = tp->second();
    printf(" time to cuda malloc %.4f\n", (stop - start));


    // Copy input vectors from host memory to GPU buffers.
    
    //start = tp->second();
    checkCudaErrors(cudaMemcpy(dev_a, a, n * n * sizeof(double), cudaMemcpyHostToDevice));

    // Copy input vectors from host memory to GPU buffers.
    
    checkCudaErrors(cudaMemcpy(dev_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    //stop = tp->second();
    //printf(" time copy data to device %.4f\n", (stop - start));

    //start = tp->second();
    // Launch a kernel on the GPU with one thread for each element.
    int nblks = 1;
    int nthreads = n;
    if (n > 1024) {
        nblks = n / 1024;
        nthreads = 1024;
    }
    MultKernel1 << <nblks, nthreads >> > (dev_a, dev_x, dev_b, n);

    //dim3 dimGrid(nblks, nblks);
    //MultKernel2 <<< dimGrid, nthreads >> > (dev_a, dev_x, dev_b, n);
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    ///stop = tp->second();
    //printf(" time to run kernel and syncornize %.4f\n", (stop - start));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
   
    checkCudaErrors(cudaMemcpy(b, dev_b, n * sizeof(double), cudaMemcpyDeviceToHost));
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    //start = tp->second();
    //checkCudaErrors(cudaDeviceReset());
//

Error:
    cudaFree(dev_x);
    cudaFree(dev_a);
    cudaFree(dev_b);
    //stop = tp->second();
    //printf(" reset and free %.4f\n\n", (stop - start));
    return 0;
}

int CudaInit(int *dimensions,int *threadsperblock)
{
    cudaError_t cudaStatus;
    cudaDeviceProp deviceProp;
    int dev = 0;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 2;
    }
    cudaGetDeviceProperties(&deviceProp, dev);
    dimensions[0] = deviceProp.maxGridSize[0];
    dimensions[1] = deviceProp.maxGridSize[1];
    dimensions[2] = deviceProp.maxGridSize[1];

    cudaFree(0);
}
//-------------------------------------------------------------------------------------------------
// Transpose a square matrix;
int CudaTranspose(double* a, double* ao, int n)
{
    double* dev_a;
    double* dev_o;
    double* dev_extra;

    register int nsize = n * n * sizeof(double);

    //checkCudaErrors(cudaMalloc((void**)&d_input, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&dev_o, nsize));

    checkCudaErrors(cudaMalloc((void**)&dev_a, nsize));

    checkCudaErrors(cudaMemcpy(dev_a, a, nsize, cudaMemcpyHostToDevice));

    //...For study of extra memoroy allocated
    //int nextra = 2048;
    //int nsizeextra = nextra * nextra *sizeof(double);
    //printf(" size extra = %d\n", nsizeextra);
    //checkCudaErrors(cudaMalloc((void**)&dev_extra, nsizeextra));
    //checkCudaErrors(cudaMemcpy(dev_extra, a, nsizeextra, cudaMemcpyHostToDevice));
    //...end extra

   
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    //naiveTransposeKernel << <gridSize, blockSize >> > (dev_a, dev_o, n);
    optimalTransposeKernel << <gridSize, blockSize >> > (dev_a, dev_o, n);
    checkCudaErrors(cudaDeviceSynchronize());

    //checkCudaErrors(cudaMemcpy(ao, dev_o, nsize, cudaMemcpyDeviceToHost));

    cudaFree(dev_o);
    cudaFree(dev_a);

    //cudaFree(dev_extra);

    checkCudaErrors(cudaDeviceReset());
    return 0;
}
void CudaMemoryTest(double* a, int  n)
{
    int ns = n * n * sizeof(double);
    double* dev_a;
    checkCudaErrors(cudaMalloc((void**)&dev_a, ns));
    Sleep(600);
    cudaFree(dev_a);
}
double CudaBandwidth(double* a, int  n)
{
    double bandwidth = 0.0;  // GB per second
    double* dev_a;

    cudaEvent_t start;
    cudaEvent_t stop;
    int nsize = n * n * sizeof(double);
    float cpu_ms = -1;

    checkCudaErrors(cudaMalloc((void**)&dev_a, nsize));

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    //checkCudaErrors(cudaMemcpy(dev_a, a, nsize, cudaMemcpyHostToDevice));
    cudaMemcpy(dev_a, a, nsize, cudaMemcpyHostToDevice);

    //checkCudaErrors(cudaMemcpy(a, dev_a, nsize, cudaMemcpyDeviceToHost));
    cudaMemcpy(a, dev_a, nsize, cudaMemcpyDeviceToHost);

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop)); 
    checkCudaErrors(cudaEventElapsedTime(&cpu_ms, start, stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    
    cudaFree(dev_a);

    bandwidth = (nsize * 2.) / (cpu_ms *1000000.) ; // Gbytes per second
    //bandwidth = bandwidth / 1000.  // bytes per second

    return bandwidth;
}