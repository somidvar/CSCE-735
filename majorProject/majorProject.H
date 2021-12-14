#ifndef majorProject_h
#define majorProject_h

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void preparationHost();
void initialization();
void LUdecomposition(double **input, double **L, double **U, int matrixDim);
void matrixConcat(double **input, double *ouput, int row, int column);
void matrixTranspose(double **input, int matDim);
void matrixMul(double** a, double**b, double**c,int aRow, int aCol, int bCol);
void LSolver(double **L, double *B, int matDim);
void USolver(double **U, double *B, int matDim);
void solver();
void test();
#endif