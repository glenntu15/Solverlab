// SolverLab.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>


#include <stdlib.h>
#include <fstream>
#include <stdio.h>
#include <string>
#include <string.h>
#include <ctime>

#include "Matrix.h"
#include "MatUtils.h"
#include "Solvers.h" 
#include "CuNVDNSolvers.h"
#include "Timers.h"
#include "mmio.h"

//#define N 12288
static int N = 13312;
//static int N = 14336;
//static int N = 4096;

// Function prototypes for CuMatUtils.cu 

int  CudMatMulV(double *a, double *x, double *b, int n);
int  CudaInit(int* dimensions, int* threadsperblock);  // dimensions[3]
double CudaBandwidth(double* a, int  n);
int CudaTranspose(double* input, double* output, int n);
void CudaMemoryTest(double* a, int  n);

// function prototypes for this source file
void cpuTranspose(const double* input, double* output, int n);
void SolveWithDense(int n);
void SolveWithSparse(int n);
template <typename T_ELEM>
int loadMMSparseMatrix(char* filename, char elem_type, bool csrFormat, int* m,
    int* n, int* nnz, T_ELEM** aVal, int** aRowInd,
    int** aColInd, int extendSymMatrix);

//
//------------------------------------------------------------------------------
//     *****  M A I N    ******
//------------------------------------------------------------------------------
int main()
{
    SolveWithDense(N);
   // SolveWithSparse(N);
}
//------------------------------------------------------------------------------
/* Naive CPU transpose, takes an n x n matrix in input and writes to output. */
void cpuTranspose(const double* input, double* output, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            output[j + n * i] = input[i + n * j];
        }
    }
}
//
//------------------------------------------------------------------------------
//
void SolveWithDense(int N)
{
    double xval = 2.0;
    double error = 0;

    int dimensions[3];
    int threads = 0;

    Timers* tm = new Timers();
    double start, stop;

    // create and clear A and U 

    printf("\n ** \n Dense Matrix (and vector) size is %d (x %d)\n ** \n", N, N);

    // For Dense matrix use this
    Matrix* ma = new Matrix(N, N);
    ma->FillRandom(MatrixFormat::Dense);
    double* A = ma->getDataPtr();

    // Create a B vector from Ax, then clear x and using B solv for X, then find the error
    MatUtils* mu = new MatUtils();

    int ierr = CudaInit(dimensions, &threads);

    printf(" calling memory test\n");
    CudaMemoryTest(A, N);
    printf(" test done\n");
    Matrix* mb = new Matrix(N, 1);
    double* B = mb->getDataPtr();

    Matrix* mx = new Matrix(N, 1);
    double* X = mx->getDataPtr();

    mx->FillConstant(xval);

   

    start = tm->second();

    mu->dMatMulV(A, X, B, N);

    stop = tm->second();
    printf(" CPU  Matrix vector multiplication time %.5f\n", (stop - start));

    //
    // ***    Now Cuda stuff    ***
    //
    //mu->Printit(B, N, 1);
   
    int ierr2 = CudaInit(dimensions, &threads);

    //double bandwidth = CudaBandwidth(A, N);
    //printf(" Measured Bandwidth:  %g GB / second\n", bandwidth);


    Matrix* mnub = new Matrix(N, 1);
    double* nub = mnub->getDataPtr();
    mnub->Clear();
    // Make the same vector multiplication using Cuda
    //start = tm->second();
    //int err = CudMatMulV(A, X, nub, N); 
    //stop = tm->second();
    //printf(" Cuda Matrix vector multiplication time %g\n", (stop - start));

    //mu->Printit(nub, N, 1);
    //
    //printf(" ** analysis of error from cpu vs gpu matrix - vector multiply\n");
    //mu->Errorprnt(B, nub, error, 0, N, 3);


    // clear x and solve for it using B
    mx->Clear();

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // Dense Matrix Stuff here 
    //Solvers* s = new Solvers(); // CPU solvers

    //
    //    **** Gausian Solver ****
    //
    //start = tm->second();
    //s->GauseSolve(A, B, X, N);
    //stop = tm->second();
    //printf(" ---> Gaussian solution time %g\n\n", (stop - start));


    Matrix* mbase = new Matrix(N, 1);
    //mbase->FillConstant(xval);          // mbase is the "right answer"
    double* bx = mbase->getDataPtr();
    // check the solved X with what X should be: xval[][]...

    //printf(" ** analysis of error GauseSolver \n");
    //mu->Errorprnt(X, bx, error, 0, N, 3);

// Now solve on the GPU
    mx->Clear();
    CuNVDNSolvers* pCusolver = new CuNVDNSolvers();
    //int linearSolverLU(int n, const double* Acopy,
    //    int lda, const double* b, double* x);

    // create A transpose
    Matrix* Atranspose = new Matrix(N, N);
    double* at = Atranspose->getDataPtr();
   // Matrix* ACutranspose = new Matrix(N, N);
    //double* acut = ACutranspose->getDataPtr();

    //start = tm->second();
    //cpuTranspose(A, at, N);
    //stop = tm->second();
    //printf(" ---> CPU matrix Transpose time %g\n\n", (stop - start));

    //start = tm->second();
    //CudaTranspose(A, acut, N);
    //stop = tm->second();
    //printf(" ---> Cuda matrix Transpose time %g\n\n", (stop - start));
    //printf(" acut %g, %g, %g\n", *acut, *(acut + 1), *(acut + 2));

    //start = tm->second();
    //pCusolver->SolverDriver(A, B, X, N, 0);
    //stop = tm->second();
    //printf(" ---> GPU LU decomp solution time %g\n\n", (stop - start));
    //printf(" ** analysis of error LU Solver \n");
    //mu->Errorprnt(X, bx, error, 0, N, 3);

    //mx->Clear();

    //start = tm->second();
    //pCusolver->SolverDriver(A, B, X, N, 1);
    //stop = tm->second();
    //printf("\n ---> GPU QR decomp solution time %g\n\n", (stop - start));
    //printf(" ** analysis of error QR Solver \n");
    //mu->Errorprnt(X, bx, error, 0, N, 3);

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // Sparse Matrix Stuff here 

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // wrap it up :)
    printf(" ******** Done **********\n");
    delete ma;
    delete mx;
    delete mbase;
}
//
//------------------------------------------------------------------------------
//
void SolveWithSparse(int n)
{
    double xval = 2.0;
    double error = 0;

    Timers* tm = new Timers();
    double start, stop;

    // create and clear A and U 

    printf("\n ** \n Sparse Matrix (and vector) size is %d (x %d)\n ** \n", N, N);

    // For Sparse matrix use this
   // Matrix* ma = new Matrix(N, N);
    //ma->FillRandom(MatrixFormat::SparseAsDense);
    
    //int nonzeros = ma->ConvertDensetoCOO();
    //printf(" ** Matrix has %d nonzeros\n", nonzeros);
//
//  *** Read in Matrix Market 
//
    std::string Filename = "lap2D_5pt_n100.mtx";
    Matrix* ma = new Matrix(Filename);
    double* A = ma->getDataPtr();
    //const char* filename = "lap2D_5pt_n100.mtx";
    const char* filename = Filename.c_str();
    int N;
    
    int nnzA = 0;  /* number of nonzeros of A */
    int baseA = 0; /* base index in CSR format */
   
    //int* coo_RowA = NULL;
    int* coo_ColA = NULL;
    double* cooValA = NULL;

    N = ma->getNrows();
   
    Matrix* mb = new Matrix(N, 1);
    double* B = mb->getDataPtr();

    Matrix* mx = new Matrix(N, 1);
    double* X = mx->getDataPtr();

    mx->FillConstant(xval);

    // Create a B vector from Ax, then clear x and using B solv for X, then find the error
    MatUtils* mu = new MatUtils();

    start = tm->second();

    mu->dMatMulV(A, X, B, N);

    stop = tm->second();
    printf(" CPU  Matrix vector multiplication time %.5f\n", (stop - start));

    Matrix* mbase = new Matrix(N, 1);
    mbase->FillConstant(xval);          // mbase is the "right answer"
    double* bx = mbase->getDataPtr();

    int rowsA = 0; /* number of rows of A */
    int colsA = 0; /* number of columns of A */
    //int nnzA = 0;  /* number of nonzeros of A */
    //int baseA = 0; /* base index in CSR format */
    /* CSR(A) from I/O */
    int* h_csrRowPtrA = NULL;
    int* h_csrColIndA = NULL;
    double* h_csrValA = NULL;


    

    // Now solve on the GPU
    mx->Clear();
    CuNVDNSolvers* pCusolver = new CuNVDNSolvers();

    // reload in crs format

    if (loadMMSparseMatrix<double>((char*)filename, 'd', true, &rowsA,
        &colsA, &nnzA, &h_csrValA, &h_csrRowPtrA,
        &h_csrColIndA, true)) {
        exit(EXIT_FAILURE);
    }

    delete mx;
    delete mb;
    delete ma;
    //delete mx;
}
