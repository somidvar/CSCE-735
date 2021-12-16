#ifndef majorProject_h
#define majorProject_h

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double **K, **I, **F, **KStar;              // 2D arrays of I, K, F and KStar
double *IFlat, *KFlat, *FFlat, *KStarFlat;  // 1D vectors of I, K F and K*
double *x, *y;                              // grid x and y
double l1, l2, t, d;                        // paramteres for the f function where l1 and l2 are normalization and t and d are noise
double xR, yR;                              // coordinate of a random point
double FStar, RealF;                        // predicted and real values of a random point
int m, n;                                   // matrix sizes

void preparationHost();                                                       // instantiating the values of parameters and F function
void initialization();                                                        // memory allocation
void LUDecomposition(double **input, double **L, double **U, int matrixDim);  // Decomposing a matrix into L and U on CPU
void LUDecompositionPrep(double **I, double **L, double **U);
void matrixConcat(double **input, double *ouput, int matDim);        // concatanting a 2D array into a 1D vector on CPU
void matrixUnConcat(double **input, double *ouput, int matDim);      // re-forming a 1D vector into 2D array on CPU
void matrixTranspose(double **input, int matDim);                    // tarnsosing a 2D matrix on CPU
void matrixMul(double **a, double **b, double **c, int matDim);      // multiplying two matrices on CPU
void LSolver(double **L, double *B, int matDim);                     // Solving LW=F on CPU
void USolver(double **U, double *B, int matDim);                     // Solving W=UX
void matrixMulPrep(double **A, double **B, double **C, int matDim);  // preparing and calling matrix multipilication on GPU
double *LUSolverPrep(double **_L, double **_U, double **_Coeff);     // preparing and calling solver on GPU
void solver();                                                       // solving the problem on CPU

void LUDecompositionTester(double **_I, double **_L, double **_U);
void matrixPrinter(double **_mat, int _matDim);

__global__ void cudaMatrixMul(double *cA, double *cB, double *cC, int commonSize);                                 // multiplying two matrices on GPU
__global__ void cudaUFactorization(double *_input, double *_L, double *_U, , double *_LU, int _matDim, int _row);  // U factorization on GPU
__global__ void cudaLFactorization(double *_input, double *_L, double *_U, , double *_LU, int _matDim, int _row);  // L factorization on GPU
__global__ void cudaLSolver(double *_L, double *_coeff, int matDim, int _rowCounter);                              // solving LW=F on GPU
__global__ void cudaUSolver(double *_U, double *_coeff, int matDim, int _rowCounter);                              // solving W=UX on GPU
__global__ void cudaLUNormalizer(double *_A, double *_coeff, int _matDim);                                         // dividing a matrix by its diagonal elements on GPU
#endif