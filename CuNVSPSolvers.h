#pragma once
#include "Solvers.h"
//#include <cusparse.h>
//#include "cusolverDn.h"
class Timers;
//class cusolverDnHandle_t;
// * This class is a wrapper for the Nvida solvers
class CuNVSPSolvers :
	public Solvers
{
public:
	CuNVSPSolvers();


private:
	Timers& timer;
};
//cusparseXcoo2csr(cusparseHandle_t   handle,
//    const int* cooRowInd,
//    int                nnz,
//    int                m,
//    int* csrRowPtr,
//    cusparseIndexBase_t idxBase)