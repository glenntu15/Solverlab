#include "CuNVSPSolvers.h"
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//
//
#include "Cusolverutils.cuh"
//
#include <cuda_runtime.h>
//
#include "cublas_v2.h"
//#include "cusolverDn.h"
//#include "helper_cuda.h"
//
#include "Timers.h"
//
//// try these for "second"
//#include "helper_cusolver.h"

CuNVSPSolvers::CuNVSPSolvers() : timer(Timers::getInstance())
{

}
//
// Assume the the row index array is sorted
void ConvertCooToCRS(int* inRowindx, int* outRowptr)
{

}