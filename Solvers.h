#ifndef _DEF_SOLVERS_H_DEFINED
#define _DEF_SOLVERS_H_DEFINED
#include <string>
#pragma once
class Plotter;
class MatUtils;

class Solvers
{
public:
    Solvers();
    ~Solvers();

    int SolveEigenPower(int N, double* M, int maxiter, int sleeptime, int reportintv);
    int JacobiSolv(double *A, double *U, double *B, int N, int maxiter, int sleeptime, int reportintv);
    int ConjugateGradSolv(double *A, double *U, double *B, int nn, int maxiter, int sleeptime, int reportintv);

	void GauseSolve(double *A, double *B, double *X, int N);

	int LaPlace(int n, double* boundry, int maxiterations, int sleeptime);

	void setErrorFlag(int val) {outputError = val; }
	
private:
    
	void ReportProgres(int k, int rfreq, double* u, int n, double error);

    int MaxIteration;
	void Decomp(double *A, double *B, int N);
	int Pivot(double *A, double *B, int k, int N);
	void Backsub(double *A, double *B, double* X, int nn);
	
    MatUtils* pMatutils;
	int outputError;	// 0 - none, 1, solver convergence criteria, 2 also difference from baseline
};
#endif

