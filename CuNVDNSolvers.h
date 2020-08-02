#pragma once
#include "Solvers.h"
#include "cusolverDn.h"
class Timers;
//class cusolverDnHandle_t;
// * This class is a wrapper for the Nvida solvers
class CuNVDNSolvers :
	public Solvers
{
public:
	CuNVDNSolvers();

	int linearSolverLU(cusolverDnHandle_t handle, int n, double* Acopy,
		int lda, const double* b, double* x);

	int linearSolverQR(cusolverDnHandle_t handle, int n, const double* Acopy,
		int lda, const double* b, double* x);

	int SolverDriver(double* A, double* B, double* X, int N, int MethodFlag);

	//int linearSolverCHOL(cusolverDnHandle_t handle, int n, const double* Acopy,
	//	int lda, const double* b, double* x);
private:
	Timers& timer;
};

