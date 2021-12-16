#ifndef majorProject_h
#define majorProject_h

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float *KFlat, *FFlat, *KStarFlat;  // 1D vectors of I, K F and K*
float *x, *y;                      // grid x and y
float l1, l2, t, d;                // paramteres for the f function where l1 and l2 are normalization and t and d are noise
float xR, yR;                      // coordinate of a random point
float FStar, RealF;                // predicted and real values of a random point
int m, n;                           // matrix sizes

void preparationGPU();                                              // instantiating the values of parameters and F function
void initialization();                                               // memory allocation
void LUDecompositionPrep(float **L, float **U);        // preparing and calling LUDecomposition on GPU
void matrixConcat(float **input, float *ouput, int matDim);        // concatanting a 2D array into a 1D vector on CPU
void matrixUnConcat(float **input, float *ouput, int matDim);      // re-forming a 1D vector into 2D array on CPU
void matrixTranspose(float **input, int matDim);                    // tarnsosing a 2D matrix on CPU
void matrixMul(float **a, float **b, float **c, int matDim);      // multiplying two matrices on CPU
void matrixMulPrep(float **A, float **B, float **C, int matDim);  // preparing and calling matrix multipilication on GPU
void LUSolverPrep(float **_L, float **_U);     // preparing and calling solver on GPU
void solver();                                                       // solving the problem on CPU
void LUDecompositionTester(float **_I, float **_L, float **_U);   // making sure that L*U==A
void matrixPrinter(float **_mat, int _matDim);                      // printing a matrix

__global__ void cudaMatrixMul(float *cA, float *cB, float *cC, int commonSize);                                                      // multiplying two matrices on GPU
__global__ void cudaUFactorization(float *_input, float *_L, float *_U, float *_LU, int _matDim, int _row);                         // U factorization on GPU
__global__ void cudaLFactorization(float *_input, float *_L, float *_U, float *_LU, int _matDim, int _row);                         // L factorization on GPU
__global__ void cudaLSolver(float *_L, float *_coeff, int matDim, int _rowCounter);                                                   // solving LW=F on GPU
__global__ void cudaUSolver(float *_U, float *_coeff, int matDim, int _rowCounter);                                                   // solving W=UX on GPU
__global__ void cudaLUNormalizer(float *_A, float *_coeff, int _matDim);                                                              // dividing a matrix by its diagonal elements on GPU
__global__ void cudaInitK(float *_cK);                                                                                                 // Initializing the K value on GPU
__global__ void cudaInitKStar(float *_cKStar);                                                                                         // Initializing the KStar value on GPU
__global__ void cudaInitF(float *_cF, float *_cX, float *_cY, int _matDim);                                                          // Initializing the F value on GPU
__global__ void cudaInitXY(float *_cX, float *_cY, int _matDim);                                                                      // Initializing X and Y on GPU
__global__ void cudaInitK(float *_cK, float *_cX, float *_cY, int _matDim, float _L1, float _L2);                                  // Initializing the K value on GPU
__global__ void cudaInitKStar(float *_cKStar, float *_cX, float *_cY, int _matDim, float _L1, float _L2, float _xR, float _yR);  // Initializing the K* value on GPU
#endif