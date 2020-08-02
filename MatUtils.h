#ifndef _DEF_MATUTILS_H_DEFINED
#define _DEF_MATUTILS_H_DEFINED
#include <string>
class MatUtils
{
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    //***************************************************************************** 
    //                          U t i l i t y   R o u t i n e s
private:
	FILE* pFiled;
	FILE* pFilee;
    double maxmagerror;

public:
    MatUtils();
    ~MatUtils();

    void dMatMulV(double *a, double *x, double *b, int N);
    int dmatcopy(double *from, int fromrows, int fromcols, double *to, int torows, int tocols, int startrow, int startcol);
    double dnorm(double *r, int n);
	void Errorprnt(double *r, double * b, double error, int k, int n, int errflag);
    void dminmax(double *X, int N, double *min, double *max);
	void dminmaxi(double *X, int N, double *min, double *max, int *mini, int *maxi);
    void Printit(double *A, int M, int N);
	void Fprintit(char c, double *A, int N);
	
};
#endif