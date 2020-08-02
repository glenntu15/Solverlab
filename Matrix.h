#pragma once
#include <stdio.h>
#include <string>
#define WORDSIZE 8

enum MatrixFormat {
	Unknown,
	Dense,
	SparseAsDense,
	COO,
	CSR
};

struct Temp_cooFormat {
	int i;
	int j;
	int p; // permutation
};

class Matrix
{
public:
	Matrix(int rows, int columns);
	Matrix(std::string Filename);
	~Matrix();
	//Matrix(int rows, int columns, MatrixFormat format);
	double* getDataPtr() { return data; }
	int getNrows() { return nrows; }

	void FillRandom(MatrixFormat format);
	void FillConstant(double c);
	void Clear();
	int FreeDenseSpace();
	int ConvertDensetoCOO();
	void ConvertCooToCrs();
	int StoreCOOAsDense(int* row, int* col, double* vals, int nr, int nc, int nz, int base);
		
	double* data;
	int* loci;
	int* locj;
	double* vals;
private:
	MatrixFormat MFormat;
	int m;		// rows -- row major
	int nrows;
	int ncols;
	int n;		// columns
	int numnz;	// number of non zeros

	int ReadMatrix(std::string filename, int* n, double*& mp,
		bool allocateNew);

};

