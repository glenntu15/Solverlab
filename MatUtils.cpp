#pragma once
#include "MatUtils.h"
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <stdio.h>
#include <string>
#include <string.h>

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
MatUtils::MatUtils()
{
	pFiled = 0;
	pFilee = 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
// Default destructor
MatUtils::~MatUtils()
{
	if (pFiled != 0)
		fclose(pFiled);
	if (pFilee != 0)
		fclose(pFilee);
}

//  This function copies the entire double precision matrix 'from' to the matrix
//      'to' beginning at startrow, startcol
//  Matrix indicies are starting from 0
//
int MatUtils::dmatcopy(double *from, int fromrows, int fromcols, double *to,
    int torows, int tocols, int startrow, int startcol)
{
    int err = 0;
    int i, j, ii, jj;
    //int idbg1, idbg2;
    //int limit = torows*tocols;
    if ((startrow + fromrows) > torows)
        return -1;
    if ((startcol + fromcols) > tocols)
        return -1;

    for (i = 0; i < fromrows; i++)
    {
        for (j = 0; j < fromcols; j++)
        {
            ii = startrow + i;
			jj = startcol + j;
			to[(ii*tocols) + jj] = from[(i*fromcols) + j];
		}
	}

	return err;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
//  This funciton computes b = Ax
//
void MatUtils::dMatMulV(double *a, double *x, double *b, int N)
{
	int i, j;
	for (i = 0; i < N; i++)
	{
		b[i] = 0;
		for (j = 0; j < N; j++)
		{
			b[i] += a[i * N + j] * x[j];
		}
	}
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
//  This funciton computes ||r||2
//
double MatUtils::dnorm(double *r, int n)
{
	double sum = r[0] * r[0];
	for (int i = 1; i < n; i++)
	{
		sum = sum + r[i] * r[i];
	}
	sum = sqrt(sum);

	return sum;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
//  This funciton computes mean and rms error between two arrays
//   errflag = 1, write k, error to file
//           = 2, write k, error, mean error, rms error to file
//   errflag = 3, write mean error, rms error, max magnitue error to screen
	
void MatUtils::Errorprnt(double *r, double * b, double error, int k, int n, int errflag)
{
	if (pFilee == 0)
	{
		pFilee = fopen("errorprint.txt", "w");
	}
	else if ((pFilee != 0) && (k < 0)) // k = -1 closes file
	{
		fclose(pFilee);
		pFilee = 0;
		return;
	}
	if (errflag == 1)
	{
		fprintf(pFilee, " %d, %g\n", k, error);
	}
	if (errflag > 1)
	{
		double meanerror = abs(r[0] - b[0]);
		maxmagerror = meanerror;

		double rmserror = meanerror * meanerror;
		double temp;
		for (int i = 1; i < n; i++)
		{
			temp = abs(r[i] - b[i]);
			if (temp > maxmagerror)
				maxmagerror = temp;
			meanerror = meanerror + temp;
			rmserror = rmserror + temp * temp;
		}
		meanerror = meanerror / (double)n;
		rmserror = sqrt(rmserror / (double)n);
		if (errflag == 2)
			fprintf(pFilee, " %d, %g, %g, %g\n", k, error, meanerror, rmserror);
		else
			printf(" Mean error %g, RMS error = %g, Max magnitude error = %g\n", meanerror, rmserror, maxmagerror);
	}
}
//
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
// for debug printing to screen
void MatUtils::Printit(double *A, int M, int N)
{
	int i, j;
	int lr = M;
	int lc = N;
	
	if (M > 40)
		lr = 40;
	if (N > 40)
		lc = 40;
	for (i = 0; i < lr; i++)
	{
		for (j = 0; j < lc; j++)
		{
			printf(" %+5.3f   ", A[i * N + j]);
		}
		printf("\n");
	}
	printf("\n");
}
// for debug printing to file
void MatUtils::Fprintit(char c, double *A, int N)
{
	if (pFiled == 0)
	{
		pFiled = fopen("matutildebugprint.txt", "w");
	}
	else
	{
		if (N < 1)
		{
			fclose(pFiled);
			pFiled = 0;
			return;
		}
	}

	int i, j;
	int l = N;
	fprintf(pFiled, "Matutil Fprintit: %c\n", c);
	if (N > 40)
		l = 40;
	for (i = 0; i < l; i++)
	{
		for (j = 0; j < l; j++)
		{
			fprintf(pFiled, " %+5.3f   ", A[i * N + j]);
		}
		fprintf(pFiled, "\n");
	}
	fprintf(pFiled, "\n");

}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
void MatUtils::dminmax(double *X, int N, double *min, double *max)
{
	*min = X[0];
	*max = X[0];
	for (int i = 1; i < N; i++)
	{
		if (X[i] < *min)
			*min = X[i];
		if (X[i] > *max)
			*max = X[i];
	}
}
void MatUtils::dminmaxi(double *X, int N, double *min, double *max, int* mini, int* maxi)
{
	*min = X[0];
	*max = X[0];
	*mini = 0;
	*maxi = 0;
	for (int i = 1; i < N; i++)
	{
		if (X[i] < *min)
		{
			*min = X[i];
			*mini = i;
		}

		if (X[i] > *max)
		{
			*maxi = i;
			*max = X[i];
		}
	}
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
