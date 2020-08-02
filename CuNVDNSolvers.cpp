#include "CuNVDNSolvers.h"
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "Cusolverutils.cuh"

#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"

#include "Timers.h"

// try these for "second"
#include "helper_cusolver.h"


/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 //-----------------------------------------------------------------------------.-------------------
// Mods:
// Made a class to wrap the solver functions
// Made a driver function to do device setup before calling solver routines
// Removed cusolverDnHandle_t from calling arguments
// added use of timer class
// added matrix transpose before QR factorization

CuNVDNSolvers::CuNVDNSolvers() : timer(Timers::getInstance())
{
    
};

int CuNVDNSolvers::SolverDriver(double* A, double* B, double* X, int N, int methodFlag)
{
    int err = 0;
    cusolverDnHandle_t handle = NULL;
    cublasHandle_t cublasHandle = NULL;
    cudaStream_t stream = NULL;
    // device pointers
    double* d_A = NULL;  // a copy of h_A
    double* d_x = NULL;  // x = A \ b
    double* d_b = NULL;  // a copy of h_b

    checkCudaErrors(cusolverDnCreate(&handle));
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cusolverDnSetStream(handle, stream));

    checkCudaErrors(cudaMalloc((void**)&d_A, sizeof(double) * N * N));
    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(double) * N));
    checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(double) * N));

    checkCudaErrors(cudaMemcpy(d_A, A, sizeof(double) * N * N, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_b, B, sizeof(double) * N, cudaMemcpyHostToDevice));

  
    if (methodFlag == 0) {
        //int linearSolverLU(int n, const double* Acopy,
    //    int lda, const double* b, double* x);
        err = linearSolverLU(handle, N, d_A, N, d_b, d_x);
    } else if(methodFlag == 1) {
        linearSolverQR(handle, N, d_A, N, d_b, d_x);
    }
    //int linearSolverLU(int n, const double* Acopy,
    //    int lda, const double* b, double* x);
    //err = linearSolverLU(handle, N, d_A, N, d_b, d_x);

    checkCudaErrors(cudaMemcpy(X, d_x, sizeof(double) * N, cudaMemcpyDeviceToHost));

    if (handle) {
        checkCudaErrors(cusolverDnDestroy(handle));
    }
    if (stream) {
        checkCudaErrors(cudaStreamDestroy(stream));
    }
    
    if (d_A) {
        checkCudaErrors(cudaFree(d_A));
    }
    if (d_x) {
        checkCudaErrors(cudaFree(d_x));
    }
    if (d_b) {
        checkCudaErrors(cudaFree(d_b));
    }

    checkCudaErrors(cudaDeviceReset());
    return err;
}
//-----------------------------------------------------------------------------.-------------------
/*
 *  solve A*x = b by LU with partial pivoting
 *
 *  lda is rows A
 */
int CuNVDNSolvers::linearSolverLU(cusolverDnHandle_t handle, int n, double* d_A,
         int lda, const double* d_b, double* d_x) 
{
    int err = 0;
    int bufferSize = 0;
    int* info = NULL;
    double* buffer = NULL;
    double* Acopy = NULL;
    int* ipiv = NULL;  // pivoting sequence
    int h_info = 0;
    //double start, stop;
    //double time_solve;

    checkCudaErrors(cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy,
        lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double) * bufferSize));
    checkCudaErrors(cudaMalloc(&Acopy, sizeof(double) * n * n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int) * n));

    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(Acopy, d_A, sizeof(double) * n * n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    //start = timer.second();

    // create the LU decomp
    checkCudaErrors(cusolverDnDgetrf(handle, n, n, d_A, lda, buffer, ipiv, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 != h_info) {
        fprintf(stderr, "Error: LU factorization failed\n");
        err = h_info;
    }

    checkCudaErrors(cudaMemcpy(d_x, d_b, sizeof(double) * n, cudaMemcpyDeviceToDevice));
    // solve for x
    checkCudaErrors(
        cusolverDnDgetrs(handle, CUBLAS_OP_T, n, 1, d_A, lda, ipiv, d_x, n, info));
    checkCudaErrors(cudaDeviceSynchronize());
    //stop = timer.second();

    double *temp = new double[16];
    checkCudaErrors(cudaMemcpy(temp, d_x, sizeof(double)*16, cudaMemcpyDeviceToHost));

    //time_solve = stop - start;
    //fprintf(stdout, "timing: LU = %10.6f sec\n", time_solve);

    if (info) {
        checkCudaErrors(cudaFree(info));
    }
    if (buffer) {
        checkCudaErrors(cudaFree(buffer));
    }
    if (Acopy) {
        checkCudaErrors(cudaFree(Acopy));
    }
    if (ipiv) {
        checkCudaErrors(cudaFree(ipiv));
    }

    return err;
}
//-----------------------------------------------------------------------------.-------------------
int CuNVDNSolvers::linearSolverQR(cusolverDnHandle_t handle, int n, const double* Acopy,
    int lda, const double* b, double* x) 
{
    cublasHandle_t cublasHandle = NULL;  // used in residual evaluation
    int bufferSize = 0;
    int bufferSize_geqrf = 0;
    int bufferSize_ormqr = 0;
    int* info = NULL;
    double* buffer = NULL;
    double* A = NULL;
    double* tau = NULL;
    int h_info = 0;
    double start, stop;
    double time_solve;
    const double one = 1.0;

    //double* d_atranspose;
    //checkCudaErrors(cudaMalloc((void**)&d_atranspose, n * n * sizeof(double)));

    //...

    checkCudaErrors(cublasCreate(&cublasHandle));

    checkCudaErrors(cusolverDnDgeqrf_bufferSize(handle, n, n, (double*)Acopy,
        lda, &bufferSize_geqrf));
    checkCudaErrors(cusolverDnDormqr_bufferSize(handle, CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T, n, 1, n, A, lda,                                     //            <<-
        NULL, x, n, &bufferSize_ormqr));

    printf("buffer_geqrf = %d, buffer_ormqr = %d \n", bufferSize_geqrf,
        bufferSize_ormqr);

    bufferSize = (bufferSize_geqrf > bufferSize_ormqr) ? bufferSize_geqrf
        : bufferSize_ormqr;
    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)* bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)* lda* n));
    checkCudaErrors(cudaMalloc((void**)&tau, sizeof(double)* n));

// prepare a copy of A because getrf will overwrite A with L
    //checkCudaErrors(
    //    cudaMemcpy(A, d_atranspose, sizeof(double)* lda* n, cudaMemcpyDeviceToDevice));
    //checkCudaErrors(cudaMemset(info, 0, sizeof(int)));
    //TranposeMatrix(Acopy, d_atranspose, n);

    TranposeMatrix(Acopy, A, n);
    // debug -- check the transpose
    //double* dbgA = new double[n * n];
    //checkCudaErrors(cudaMemcpy(dbgA, d_atranspose, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
    //printf(" A after transpose %g, %g, %g\n", *dbgA, *(dbgA + 1), *(dbgA + 2));

    //debug
    //checkCudaErrors(cudaMemcpy(dbgA, A, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
    //printf(" A to use %g, %g, %g\n", *dbgA, *(dbgA + 1), *(dbgA + 2));

    start = second();

    // compute QR factorization
    checkCudaErrors(
        cusolverDnDgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    checkCudaErrors(
        cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 != h_info) {
        fprintf(stderr, "Error: QR factorization failed\n");
    }

    checkCudaErrors(
        cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    checkCudaErrors(cusolverDnDormqr(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1,
        n, A, lda, tau, x, n, buffer, bufferSize,
        info));

    // x = R \ Q^T*b
    checkCudaErrors(cublasDtrsm(cublasHandle, CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, n, 1, &one, A, lda, x, n));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf(stdout, "timing: QR = %10.6f sec\n", time_solve);

    if (cublasHandle) {
        checkCudaErrors(cublasDestroy(cublasHandle));
    }
    if (info) {
        checkCudaErrors(cudaFree(info));
    }
    if (buffer) {
        checkCudaErrors(cudaFree(buffer));
    }
    if (A) {
        checkCudaErrors(cudaFree(A));
    }
    if (tau) {
        checkCudaErrors(cudaFree(tau));
    }

    return 0;
}
//-----------------------------------------------------------------------------.-------------------
/*int CuDNSolvers::linearSolverCHOL(cusolverDnHandle_t handle, int n, const double* Acopy,
    int lda, const double* b, double* x) {

    int status = 0;

    return status;
}*/
