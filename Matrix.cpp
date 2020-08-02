#include "Matrix.h"
#include <string>
#include <string.h>
#include <random>
#include <time.h>
#include <chrono>

#include <vector>

#include <iostream>


Matrix::Matrix(int rows, int columns)
{
	MFormat = Dense;
	data = new double[rows * columns];
	m = rows;
	n = columns;
	nrows = m;
	ncols = columns;
	size_t isize = rows * columns * sizeof(double);
	memset(data, 0, isize);
	loci = NULL;
	locj = NULL;
	vals = NULL;
}
Matrix::Matrix(std::string Filename)
{
	if (data)
		delete[] data;
	int n;
	int ierr = ReadMatrix(Filename, &n, data, true);
	if (ierr == 0) {
		nrows = n;
		ncols = n;
	}
}
Matrix::~Matrix()
{
	if (data)
		delete[] data;
	if (loci)
		delete[] loci;
	if (locj)
		delete[] locj;
	if (vals)
		delete[] vals;
	
}
//-----------------------------------------------------------------------------
void Matrix::FillRandom(MatrixFormat format)
{
	unsigned int seed;
	typedef std::uniform_real_distribution<double> distribution;
	//typedef std::default_random_engine engine;
	// better
	std::mt19937* gen;
	std::uniform_real_distribution<double>* dis;

	using namespace std::chrono;
	microseconds ms = duration_cast<microseconds>(
		system_clock::now().time_since_epoch());

	unsigned int iu = (unsigned int)pow(2, 31);
	seed = 2020;  // (unsigned int)(ms.count()) - (unsigned int)pow(2, 31);
	gen = new std::mt19937(seed);
	dis = new std::uniform_real_distribution<double>(-1.0, 1.0);
	
	double myrand = (*dis)(*gen);
	//std::cout  << " Number is: " << myrand << std::endl;

	if (format == MatrixFormat::Dense) {
		for (int i = 0; i < nrows; i++) {
			for (int j = 0; j < ncols; j++) {
				data[i * ncols + j] = (*dis)(*gen);
			}
		}
	}
	else {	
		MFormat = MatrixFormat::SparseAsDense;
		// square matrix
		int i, io, j;
		
		data[0] = (*dis)(*gen);  // top left corner
		data[1] = (*dis)(*gen);  // top left + 1
		data[2] = (*dis)(*gen);  // top left +2
		i = 1;
		data[i * ncols] = (*dis)(*gen);			// column 0
		data[i * ncols + 1] = (*dis)(*gen);		// column 1
		for (i = 2; i < nrows-2; i++) {
			data[i * ncols + i] = (*dis)(*gen); // diagonal
			j = i;
			data[i * ncols + (j - 1)] = (*dis)(*gen); // off diagonal left
			data[i * ncols + (j - 2)] = (*dis)(*gen); // off diagonal left
			data[i * ncols + (j + 1)] = (*dis)(*gen); // off diagonal right
			data[i * ncols + (j + 2)] = (*dis)(*gen); // off diagonal right
		}
		i = nrows - 2;
		data[i * ncols + i] = (*dis)(*gen);
		data[i * ncols + (i - 1)] = (*dis)(*gen);
		data[i * ncols + (i - 2)] = (*dis)(*gen);
		data[i * ncols + (i + 1)] = (*dis)(*gen);
		i++;
		data[i * ncols + i] = (*dis)(*gen);
		data[i * ncols + (i - 1)] = (*dis)(*gen);
		data[i * ncols + (i - 2)] = (*dis)(*gen);
		j = 0;
		for (i = nrows / 4; i < nrows; i++) {
			data[i * ncols + j] = (*dis)(*gen);
			j++;
		}
		i = 0;
		for (j = ncols / 4; j < ncols; j++) {
			data[i * ncols + j] = (*dis)(*gen);
			i++;
		}
		int is = ncols - 10;
		int js = nrows - 20;
		for (i = is; i < is + 6; i++) {
			for (j = js; j < js + 6; j++) {
				data[i * ncols + j] = (*dis)(*gen);
			}
		}
		is = nrows / 2;;
		js = 10;
		for (i = is; i < is + 6; i++) {
			for (j = js; j < js + 6; j++) {
				data[i * ncols + j] = (*dis)(*gen);
			}
		}
	}
	
}
//-----------------------------------------------------------------------------
void Matrix::FillConstant(double c)
{
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			data[j * n + i] = c;
		}
	}
}
//-----------------------------------------------------------------------------
void Matrix::Clear()
{
	memset(data, 0, WORDSIZE * m * n);
	return;
}
//-----------------------------------------------------------------------------
int Matrix::FreeDenseSpace()
{
	int error = 0;

	return error;
}
//-----------------------------------------------------------------------------
int Matrix::ConvertDensetoCOO()
{
	double eps = 1.0e-14;

	if ((MFormat != MatrixFormat::Dense) && (MFormat != MatrixFormat::SparseAsDense))
		return -1;
	if ((nrows == 0) || (ncols == 0))
		return -1;

	std::vector<int> rowindx;
	std::vector<int> colindx;
	std::vector<double> tempvals;
	
	numnz = 0;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			if (abs(data[i * ncols + j]) > eps) {
				rowindx.push_back(i);
				colindx.push_back(j);
				tempvals.push_back(data[i * ncols + j]);
				numnz++;
			}
		}
	}
	//int* loci;
	//int* locj;
	//double* vals;
	loci = new int[numnz];
	locj = new int[numnz];
	vals = new double[numnz];
	
	for (int i = 0; i < numnz; i++) {
		loci[i] = rowindx[i];
		locj[i] = colindx[i];
		vals[i] = tempvals[i];
	}
	return numnz;
}
//-----------------------------------------------------------------------------
void Matrix::ConvertCooToCrs()
{

}
//-----------------------------------------------------------------------------
///  stores sparse values as dense in already allocated space
/// row[] row locations, col [] column locations,  vals[] values to be loaded
/// nr, nc, number of rows and columns, used to check against allocated space
/// nz number of non zero entries, base ->0,1 start index, converted to 0 if 1
/// returns int = 0 -> success
int Matrix::StoreCOOAsDense(int* row, int* col, double* vals, int nr, int nc, int nz, int base) 
{
	int error = 0;
	if ((nr > nrows) || (nc > ncols))
		return 1;
	if ((base != 0) && (base != 1))
		return 2;
	int i, j;
	for (int n = 0; n < nz; n++) {
		i = row[n] - base;
		j = col[n] - base;
		data[i * ncols + j] = vals[n];
		if (n < 10)
			printf(" i %d, j %d , val %g \n", i, j, vals[n]);
	}
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
// N is one dimension of the square matrix A
int Matrix::ReadMatrix(std::string filename, int* n, double*& mp,
	bool allocateNew)
{
	// We make the risky assumption that when allocateNew is false
	// the input matrix is the same one read before, or,
	// at least the same size
	//
#define MM_MAX_LINE_LENGTH 122
#define MM_MAX_TOKEN_LENGTH  20

#define MM_DENSE_STR	    "array"
#define MM_COORDINATE_STR   "coordinate" 
#define MM_COMPLEX_STR	    "complex"
#define MM_REAL_STR		    "real"
#define MM_INT_STR		    "integer"

	char line[MM_MAX_LINE_LENGTH];

	char banner[MM_MAX_TOKEN_LENGTH];
	char mtx[MM_MAX_TOKEN_LENGTH];
	char crd[MM_MAX_TOKEN_LENGTH];
	char data_type[MM_MAX_TOKEN_LENGTH];
	char storage_scheme[MM_MAX_TOKEN_LENGTH];
	int M, N, nz;
	bool isCoordinate = true;
	FILE* f;
	int len;
	// on windows it's in .../SolverDisplay/Solverdisplay
	if ((f = fopen(filename.c_str(), "r")) == NULL)
	{
		printf(" unable to open %s\n", filename.c_str());
		perror("error: ");
		return -1;
	}

	if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
	{
		printf(" file %s is apparently empty\n", filename.c_str());
		return -2;
	}

	sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,
		storage_scheme);

	int numheaderitems = 3;
	if (strcmp(crd, MM_DENSE_STR) == 0)
	{
		isCoordinate = false;
		numheaderitems = 2;
	}

	// only real is supported now
	if (strcmp(data_type, MM_REAL_STR) != 0)
	{
		return -2;
	}

	// loop past comments to find line with blank first char
	do {
		if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
			return -2;
	} while (line[0] == '%');

	bool headerRead = false;
	/* line[] is either blank or has M,N, nz */
	if (sscanf(line, "%d %d %d", &M, &N, &nz) == numheaderitems)
		headerRead = true;
	else
	{
		int num_items_read;
		do
		{
			if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
				return -2;
			num_items_read = sscanf(line, "%d %d %d", &M, &N, &nz);
			if (num_items_read == EOF) return -2;
			if (num_items_read == 3)
				headerRead = true;
		} while (!headerRead);
	}
	if (isCoordinate)
	{
		if (M != N)
			printf(" @@@@ NOT A SQUARE MATRIX @@@@\n");
		*n = N;
		// Create A matrix
		if (allocateNew)
			mp = new double[M * N];
		memset(mp, 0, M * N * sizeof(*mp));
		//   Read and set values
		int i, j;
		double val = 0.0;

		for (int k = 0; k < nz; k++)
		{
			len = fscanf(f, "%d %d %lg\n", &i, &j, &val);
			i--; j--;   // input is 1 based, we use 0 based
			mp[M * i + j] = val;
		}
	}
	else
	{
		if (allocateNew)
			mp = new double[M * N];
		memset(mp, 0, M * N * sizeof(*mp));
		//   Read and set values
		double val;
		for (int k = 0; k < M; k++)
		{
			len = fscanf(f, "%lg\n", &val);
			mp[k] = val;
		}
	}
	return 0;
}

//int cmp_cooFormat_csr(struct cooFormat* s, struct cooFormat* t)
//{
//	if (s->i < t->i) {
//		return -1;
//	}
//	else if (s->i > t->i) {
//		return 1;
//	}
//	else {
//		return s->j - t->j;
//	}
//}
