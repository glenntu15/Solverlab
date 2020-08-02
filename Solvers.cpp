#include "Solvers.h"
#include "MatUtils.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h> // for memset
#include <ctime>
#ifdef _WINDOWS
#include <Windows.h> // for sleep
#include <tchar.h> // 
#else
#include <unistd.h>
#include <sys/resource.h>
#endif
//#include "lapacke.h"

#ifdef _WINDOWS
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG
#endif  // _WINDOWS

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
// Default constructor
Solvers::Solvers()
{
    pMatutils = new MatUtils();
	outputError = 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
// Default destructor
Solvers::~Solvers()
{
	delete pMatutils;
	
}
//
// N is the side of the large matrix -- it is the same as nn used outside of this class
int Solvers::SolveEigenPower(int N, double* M, int maxiter, int sleeptime, int reportintv)
{
    int err = 0;
	bool quit = false;
    MaxIteration = maxiter;
    double Tolerance = 1.0e-6;
    
    double *R = new double[N];
    memset(R, 0, N*sizeof(*R));
    double *X = new double[N];
    double *Y = new double[N];
    memset(Y, 0, N*sizeof(*Y));
    for (int i = 0; i < N; i++)
    {
        X[i] = 1.0;
    }

    //   Iterate
    //   M is beta * M 

    double error;
    double r, normal;
    int k;
    // select which element to compare 

    int m = N / 2;
    
    for (k = 1; k <= MaxIteration; k++)
    {
        pMatutils->dMatMulV(M, X, Y, N);

        r = (Y[m]) / (X[m]);  // new eigenvalue estimate

        normal = 1.0 / pMatutils->dnorm(Y, N);

        //debug
        double min = 0, max = 0.;
       
        if (k > 1)
        {
            // compute residual
            for (int i = 0; i <N; i++)
            {
                R[i] = X[i] - (Y[i] * normal);
            }
            for (int i = 0; i < N; i++)
            {
                X[i] = Y[i] * normal;
            }
            pMatutils->dminmax(X, N, &min, &max);
            error = pMatutils->dnorm(R, N);
			if (quit)
				break;
#ifdef _WINDOWS
            Sleep(sleeptime);
#else
            usleep(sleeptime*1000);
#endif
			ReportProgres(k, reportintv, X, N, error);
            if (error < Tolerance)
            {
                return k;
            }
        }
    }
	delete [] X;
	delete [] Y;
    err = -k;
    return err;
}
//==============================================================================
//             L i n e a r   A l g e b r a   S o l v e r s
//==============================================================================
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
//     use Gaussian Elimination for direct solve 
//
void Solvers::GauseSolve(double *A, double *B, double *X, int nn)
{
	
	

// * * * * * * *  
	//clock_t startcputime = clock();
//
// *** lapack
//
	//int N = nn;
	//lapack_int n = nn;
	////lapack_int ipiv[12000];
 //       lapack_int *ipiv = (lapack_int *)malloc(n*sizeof(lapack_int)) ;
	//lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, A, n, ipiv,
	//			B, 1);
 //       printf(" ** info = %d, N = %d\n",info,N);
 //       for (int i = 0;i<N;i++)
 //       {
 //          base[i] = B[i];
 //       }

//
// *** custom 
//
	Decomp(A, B, nn);

    Backsub(A, B, X, nn);

	//clock_t endcputime = clock();
	//float t = (float)(endcputime - startcputime) / CLOCKS_PER_SEC;
	//printf(" Gausian Elimination calculataion time %f\n", t);
	//printf(" enter character to continue\n");
	//int retval = scanf("%c", &c);
// * * * * * * *  

	
#ifdef DEBUG_PRINT
	int nx = (int)sqrt(nn);
	pMatutils->Fprintit('b', base, nx);
	pMatutils->Fprintit('b', base, 0);// closes file
#endif
	

	//memset(B, 0, nn*sizeof(B));

}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
//     Jocobi solver functions 
int Solvers::JacobiSolv(double *A, double *U, double *B, int N, int maxiter, int sleeptime, 
						int reportintv)
{
    int MaxIterations = maxiter;
    double Tolerance = 1.0e-8;
    double currentError = 0.0;

    int i, j, k;
    //int nn = N*N;
    int err = 0;
    // create and clear a vector X
    double* X = new double[N];
    memset(X, 0, N*sizeof(*X));
    double* R = new double[N];
    memset(R, 0, N*sizeof(*R));
    double* Btemp = new double[N];
    memset(Btemp, 0, N*sizeof(*Btemp));
//    double ratio = 1.0;


    double d, sum, sumx = 0.0, sumu = 0.0;
    for (k = 1; k <= MaxIterations; k++)
    {
        for (i = 0; i < N; i++)
        {
            d = 1.0 / A[i*N + i]; // a(i,i)
            sum = 0.0;
            for (j = 0; j < N; j++)
            {
                if (i != j)
                    sum = sum + A[i*N + j] * X[j];
            }
            U[i] = (B[i] - sum) * d;
        }

        
		
        
#ifdef _WINDOWS
        Sleep(sleeptime);
#else
        usleep(sleeptime*1000);
#endif
        if (k > 3)
        {
            sumx = X[0] * X[0];
            sumu = U[0] * U[0];
            double maxu = 0.0;
            for (int it = 1; it < N; it++)
            {
                sumx = sumx + X[it] * X[it];
                sumu = sumu + U[it] * U[it];
                if (U[it] > maxu)
                    maxu = U[it];
            }

            sumx = sqrt(sumx);
            sumu = sqrt(sumu);
            currentError = fabs(sumx - sumu);
            
			ReportProgres(k, reportintv, U, N, currentError);
            if (currentError <= Tolerance)
            {
                err = k;
                
                delete[] X;
                delete[] R;
                delete[] Btemp;
                return err;
            }
        }
       
        for (i = 0; i < N; i++)
        {
            X[i] = U[i];
        }
        err = -k;
    }
    //(" residual %g\n", (fabs(sumx - sumu)));
    delete[] X;
	delete [] Btemp;
	delete[] R;
    return err;
}

int Solvers::ConjugateGradSolv(double *A, double *U, double *B, int N, int MaxIterations, int sleeptime, 
							   int reportintv)
{
    //int nn = N * N;
    double Tolerance = 1.0e-6;
    double *R = new double[N];
    memset(R, 0, N*sizeof(*R));
    double *X = new double[N];
    memset(X, 0, N*sizeof(*X));
    double *W = new double[N];
    memset(W, 0, N*sizeof(*W));
    double*P = new double[N];
    memset(P, 0, N*sizeof(*P));
    double*Resid = new double[N];
    memset(Resid, 0, N*sizeof(*Resid));

    double R0norm = 1.0;
    double Rnorm;
    double ratio, alpha, delta, deltaOld = 1.0;
    double beta;

    // 
    for (int i = 0; i < N; i++)
    {
        R[i] = B[i] - X[i]; // r <- B - AU
    }
	
    R0norm = pMatutils->dnorm(R, N);
    // start iterative process
    double  wp;

    delta = 0.0;
    for (int ii = 0; ii < N; ii++)
    {
        delta = delta + R[ii] * R[ii];
    }
    // delta is now the inner product of R with itself

	

//    double min, max;
    int k;
	bool converged = false;
    for (k = 1; k < MaxIterations; k++)
    {
        if (k == 1)
        {
            for (int ii = 0; ii < N; ii++)
            {
                P[ii] = R[ii];
            }

        }
        else
        {
            beta = delta / deltaOld;
            for (int ii = 0; ii < N; ii++)
            {
                P[ii] = R[ii] + beta * P[ii];
            }

        }

        pMatutils->dMatMulV(A, P, W, N);

        wp = 0.0;
        for (int ii = 0; ii < N; ii++)
        {
            wp = wp + W[ii] * P[ii];
        }
        alpha = delta / wp;                     // alpha <-delta / <p,w>
        //  u <- u + t * p
        for (int ii = 0; ii < N; ii++)
        {
            U[ii] = U[ii] + alpha * P[ii];
        }

        // r <- r - alpha * W
        for (int ii = 0; ii < N; ii++)
        {
            R[ii] = R[ii] - alpha * W[ii];
        }

        deltaOld = delta;
        delta = 0.0;
        for (int ii = 0; ii < N; ii++)
        {
            delta = delta + R[ii] * R[ii];
        }
        pMatutils->dMatMulV(A, U, X, N);
        // 
        for (int ii = 0; ii <N; ii++)
        {
            Resid[ii] = B[ii] - X[ii]; // r <- B - AU
        }

        Rnorm = pMatutils->dnorm(Resid, N);
        ratio = Rnorm / R0norm;
		
			

        //pMatutils->dminmaxi(U, N, &min, &max, &imin, &imax);
		//printf(" U:  %g  %g  %g, ratio %g min %g max %g, at: %d  %d\n", U[0], U[1], U[2], ratio, min, max, imin, imax);
		ReportProgres(k, reportintv, U, N, ratio);
#ifdef _WINDOWS
        Sleep(sleeptime);
#else
        usleep(sleeptime * 1000);
#endif

        if (ratio <= Tolerance)
		{
			converged = true;
			break;
		}
        double bnorm = pMatutils->dnorm(B, N);
        double ratiob = Rnorm / bnorm;
		if (ratiob <= Tolerance)
		{
			converged = true;
			break;
		}

    }
	// we're done set output error to -1 to flag for file to be closed
	ReportProgres(k, reportintv, U, N, ratio);
	if (!converged)
	{
		k = k * -1;
	}

	delete[] R;
    delete[] X;
    delete[] W;
    delete[] P;
	delete[] Resid;
    return k;
}

//
// ============================================================================
// Unility routines
//

void Solvers::ReportProgres(int k, int rfreq, double *U, int n, double error)
{
	int imax, imin;
	double min, max;
	if (rfreq > 0)
	{
		if (k % rfreq == 0)
		{
			pMatutils->dminmaxi(U, n, &min, &max, &imin, &imax);
			printf(" K: %d, U:  %g  %g  %g, error %g min %g max %g, at: %d  %d\n", k, U[0], U[1], U[2], error, min, max, imin, imax);
		}
	}
	
	
}
//=============================================================================
//    D i r e c t   S o l v e r   ( G A U S S)
//
void Solvers::Decomp(double *A, double *B, int N)
{
	//double small = 1.0e-6;
	double c, ra;
	int n = N;
	int i, j, k, it;
	double eps = 1.0e-9;
	int countpivot = 0;
	// debug
	// 
	for (i = 0; i < n; i++)
	{
		if (abs(A[i*n + i]) < eps)
			printf(" error on diagonal at i = %d\n",i);
	}
	for (k = 0; k<n; k++) /* loop for the generation of upper triangular matrix*/
	{
		//printit(A
		// count pivots;
		it = Pivot(A, B, k, n);
		countpivot = countpivot + it;

		if (abs(A[k*n + k]) > eps)
		{
			//if (abs(A[k*n + k]) < .00001)
			//	printf(" error at K = %d, i = %d\n", k, i);
			ra = 1.0 / A[k*n + k];
			for (i = k + 1; i<n; i++)
			{
				c = A[i*n + k] * ra;
				for (j = k; j<n; j++)
				{
					A[i*n + j] = A[i*n + j] - c*A[k*n + j];
				}
				B[i] = B[i] - c*B[k];
			}
		}
	}
	printf(" Pivot count is %d\n", countpivot);
}
//
int Solvers::Pivot(double *A, double *B, int k, int N)
{
	int i, ii, ibigest;
	double max;
	max = fabs(A[k *N + k]);
	ibigest = k;

	for (i = k + 1; i<N; i++)
	{
		if (fabs(A[i*N + k]) > max)
		{
			max = fabs(A[i*N + k]);
			ibigest = i;
		}
	}
	if (ibigest == k)
		return 0;
	//
	//  Swap rows using max as a temp variable
	//
	for (ii = k; ii<N; ii++)
	{
		max = A[k*N + ii];
		A[k*N + ii] = A[ibigest * N + ii];
		A[ibigest * N + ii] = max;
	}
	max = B[ibigest];
	B[ibigest] = B[k];
	B[k] = max;

	return 1;
}
void Solvers::Backsub(double *A, double *B, double* X, int N)
{
	double sum;
	int i, j;
	if (abs(A[(N - 1) * N + (N - 1)]) < .0001)
		X[N - 1] = 0.0;
	else
		X[N - 1] = B[N - 1] / A[(N - 1) * N + (N - 1)];
	for (i = N - 2; i >= 0; i--)
	{
		sum = 0.0;
		for (j = i + 1; j <= N - 1; j++)
		{
			sum = sum + A[i * N +j] * X[j];
		}
		j = N - 1;
		if (abs(A[i * N + i]) < .0001)
			printf(" error\n");
		

		X[i] = (B[i] - sum) / A[i * N + i];
	}
}
//==============================================================================
//             L a   P l a c e   S o l v e r 
//==============================================================================
int Solvers::LaPlace(int n, double* boundry, int maxiterations,int sleeptime)
{
	// allocate and initialize the four boundry conditions
	
	int i, j;
	//
	 	
	// 
	int nphi = n - 2;
	double** A = new double*[n];
	for (i = 0; i < n; i++)
	{
		A[i] = new double[n];
		memset(A[i], 0, n*sizeof(double));
	}

	// temp
	double *X = new double[nphi*nphi];
	memset(X, 0, (nphi*nphi)*sizeof(double));
	
	// put boundry in North, then south, east and west
	int ns = n - 2;
	int bi = 0;
	for (j = 1; j < n-1; j++)
	{
		A[0][j] = boundry[bi];
		A[n - 1][j] = boundry[bi + ns];
		bi++;
	}
	
	bi = 2 * ns;
	for (i = 1; i < n-1; i++)
	{
		A[i][0] = boundry[bi];
		A[i][n - 1] = boundry[bi+ns];
		bi++;
	}
	int iter = 0;
	double err = 0.0;
	double tolerance = 1.0e-6;
	
	for (int k = 0; k <= maxiterations; k++)
	{
		// loop over the interior points -- away from rows / columns ajcent to boundry points
		err = 0.0;
		for (i = 1; i <(n - 1); i++)
		{
			for (j = 1; j < (n - 1); j++)
			{
				A[i][j] = .25 * (A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + A[i][j - 1]);
				err += abs(X[(i - 1)*nphi + (j - 1)] - A[i][j]);
				X[(i-1)*nphi + (j-1)] = A[i][j];
				if (err < tolerance)
					iter = k;
				else
					iter = -k;
			}
			//printf("\n");
		}
		
		
		err = 0.0;
#ifdef _WINDOWS
		Sleep(sleeptime);
#else
		usleep(sleeptime * 1000);
#endif
	}
	
	delete[] X;
	
	for (i = 0; i < n; i++)
	{
		delete[] A[i];
	}
	delete[] A;
	A = NULL;
	return iter;
}

