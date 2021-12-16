#include "majorProject.H"

int main() {
    xR = 0.2;
    yR = 0.8;
    m = 16;
    l1 = 1.0;
    l2 = 1.0;
    t = 0.01;

    initialization();
    preparationHost();
    solver();
    return 0;
}
void initialization() {
    RealF = 1 - (pow((xR - 0.5), 2) + pow((yR - 0.5), 2));

    n = m * m;
    x = new double[m];
    y = new double[m];

    FFlat = new double[n];
    KStarFlat = new double[n];
    IFlat = new double[n * n];
    KFlat = new double[n * n];

    for (int i = 0; i < n; i++) {
        I = new double *[n];
        K = new double *[n];
        for (int j = 0; j < n; j++) {
            I[j] = new double[n];
            K[j] = new double[n];
        }
    }
    for (int i = 0; i < m; i++) {
        F = new double *[m];
        KStar = new double *[m];
        for (int j = 0; j < m; j++) {
            F[j] = new double[m];
            KStar[j] = new double[m];
        }
    }
}
void preparationHost() {
    double temp;
    for (int i = 0; i < m; i++) {
        x[i] = i * 1.00 / (m + 1);
        y[i] = i * 1.00 / (m + 1);
    }

    for (int i = 0; i < m; i++) {  // initializing matrix F
        for (int j = 0; j < m; j++) {
            double d = rand() * 1.00 / (RAND_MAX);
            d -= 0.5;
            d /= 10;
            F[i][j] = 1 - (pow(x[i] - 0.5, 2) + pow(y[j] - 0.5, 2)) + d;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                I[i][j] = 1;
            } else
                I[i][j] = 0;
        }
    }
    temp = 0;
    for (int i = 0; i < m; i++) {  // first point
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < m; l++) {  // second point
                for (int k = 0; k < m; k++) {
                    temp = (pow(x[i] - x[l], 2)) / (2 * l1 * l1) + (pow(y[j] - y[k], 2)) / (2 * l2 * l2);  // row of first point and column of second point
                    temp = exp(-temp) / (sqrt(2 * M_PI));
                    K[k + l * m][j + i * m] = temp;
                }
            }
        }
    }

    temp = 0;
    for (int i = 0; i < m; i++) {  // first point
        for (int j = 0; j < m; j++) {
            temp = (pow(x[i] - xR, 2)) / (2 * l1 * l1) + (pow(y[j] - yR, 2)) / (2 * l2 * l2);  // row of first point and column of second point
            temp = exp(-temp) / (sqrt(2 * M_PI));
            KStar[i][j] = temp;
        }
    }
}

void solver() {
    double **L, **U;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            I[i][j] += t;
            I[i][j] += K[i][j];
        }
    }

    matrixConcat(F, FFlat, m);
    for (int i = 0; i < n; i++) {
        L = new double *[n];
        U = new double *[n];
        for (int j = 0; j < n; j++) {
            L[j] = new double[n];
            U[j] = new double[n];
        }
    }

    // LUDecomposition(I, L, U, n);
    LUDecompositionPrep(I, L, U);
    LUDecompositionTester(I, L, U);
    FFlat = LUSolverPrep(L, U, F);
    matrixConcat(KStar, KStarFlat, m);
    FStar = 0;
    for (int i = 0; i < n; i++) {
        FStar += FFlat[i] * KStarFlat[i];
    }
    printf("The predicted value=%2.4f and the real is %2.4f\n", FStar, RealF);
}
void LUDecompositionTester(double **_I, double **_L, double **_U) {
    double **temp;
    for (int i = 0; i < n; i++) {
        temp = new double *[n];
        for (int j = 0; j < n; j++) {
            temp[j] = new double[n];
        }
    }
    matrixMul(_L, _U, temp, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            assert(fabs(_I[i][j] - temp[i][j]) < 0.001);
        }
    }
}
void LUDecompositionPrep(double **_I, double **_L, double **_U) {
    matrixConcat(_I, IFlat, n);

    double *LFlat = new double[n * n];
    double *UFlat = new double[n * n];

    double *cLFlat, *cUFlat, *cLUFlat, *cIFlat;

    cudaMalloc(&cLFlat, n * n * sizeof(double));
    cudaMalloc(&cUFlat, n * n * sizeof(double));
    cudaMalloc(&cLUFlat, n * n * sizeof(double));
    cudaMalloc(&cIFlat, n * n * sizeof(double));

    cudaMemcpy(cIFlat, IFlat, n * n * sizeof(double), cudaMemcpyHostToDevice);

    int TRHEAD_NUM = 4;
    dim3 dimBlockLU(TRHEAD_NUM, 1);
    dim3 dimGridLU(n / TRHEAD_NUM, 1);

    dim3 dimBlockMul(TRHEAD_NUM, TRHEAD_NUM);
    dim3 dimGridMul(n / TRHEAD_NUM, n / TRHEAD_NUM);

    if (n % TRHEAD_NUM != 0)
        assert(false);

    for (int i = 0; i < n; i++) {
        cudaMatrixMul<<<dimGridMul, dimBlockMul>>>(cLFlat, cUFlat, cLUFlat, n);
        cudaUFactorization<<<dimGridLU, dimBlockLU>>>(cIFlat, cLFlat, cUFlat, n, i);
        cudaMatrixMul<<<dimGridMul, dimBlockMul>>>(cLFlat, cUFlat, cLUFlat, n);
        cudaLFactorization<<<dimGridLU, dimBlockLU>>>(cIFlat, cLFlat, cUFlat, n, i);
    }

    cudaMemcpy(LFlat, cLFlat, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(UFlat, cUFlat, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    matrixUnConcat(_L, LFlat, n);
    matrixUnConcat(_U, UFlat, n);
}
__global__ void cudaUFactorization(double *_input, double *_L, double *_U, double *_LU, int _matDim, int _row) {
    int rowPerThread = _matDim / blockDim.x / gridDim.x;
    int firstRow = rowPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
    int lastRow = firstRow + rowPerThread;

    if (firstRow < _row)
        firstRow = _row;

    for (int k = firstRow; k < lastRow; k++) {
        _U[k + _row * _matDim] = _input[k + _row * _matDim] - _LU[k + _row * _matDim];
    }
}
__global__ void cudaLFactorization(double *_input, double *_L, double *_U, double *_LU, int _matDim, int _row) {
    int rowPerThread = _matDim / blockDim.x / gridDim.x;
    int firstRow = rowPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
    int lastRow = firstRow + rowPerThread;

    if (firstRow < _row)
        firstRow = _row;

    for (int k = _row; k < _matDim; k++) {
        _L[_row + k * _matDim] = (_input[_row + k * _matDim] - _LU[_row + k * _matDim]) / _U[_row + _row * _matDim];
    }
}

void LUDecomposition(double **_input, double **_L, double **_U, int matrixDim) {
    for (int i = 0; i < matrixDim; i++) {
        for (int k = i; k < matrixDim; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (_L[i][j] * _U[j][k]);
            _U[i][k] = _input[i][k] - sum;
        }

        for (int k = i; k < n; k++) {
            if (i == k)
                _L[i][i] = 1;
            else {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (_L[k][j] * _U[j][i]);
                _L[k][i] = (_input[k][i] - sum) / _U[i][i];
            }
        }
    }
}

void matrixConcat(double **input, double *output, int matDim) {
    for (int x = 0; x < matDim; x++) {      // y direction
        for (int y = 0; y < matDim; y++) {  // x direction
            output[y + x * matDim] = input[x][y];
        }
    }
}

void matrixUnConcat(double **input, double *output, int matDim) {
    for (int x = 0; x < matDim; x++) {      // y direction
        for (int y = 0; y < matDim; y++) {  // x direction
            input[x][y] = output[y + x * matDim];
        }
    }
}

__global__ void cudaLUNormalizer(double *_A, double *_coeff, int _matDim) {  // diving each row to change the diagonal elements to 1 for A*W=Coeff
    if (_matDim % blockDim.x != 0)
        assert(false);
    int rowPerThread = _matDim / blockDim.x / gridDim.x;
    int firstRow = rowPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
    int lastRow = firstRow + rowPerThread;
    for (int i = firstRow; i < lastRow; i++) {  // each thread
        double temp = _A[i + i * _matDim];
        _coeff[i] /= temp;
        for (int j = 0; j < _matDim; j++) {
            _A[j + i * _matDim] /= temp;
        }
    }
    __syncthreads();
}

__global__ void cudaLSolver(double *_L, double *_coeff, int _matDim, int _row) {  // assuming L*U=A and A^-1 * coeff=X-> coeff=A*X where A and coeff are known and X is unknown. L*U*X=Coeff->L*W=Coeff then we find W and then W=UX and we find X
    if (_matDim % blockDim.x != 0)
        assert(false);
    int rowPerThread = _matDim / blockDim.x / gridDim.x;
    int firstRow = rowPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
    int lastRow = firstRow + rowPerThread;

    if (firstRow < _row + 1)
        firstRow = _row + 1;

    for (int j = firstRow; j < lastRow; j++) {  // each thread
        _coeff[j] -= _coeff[_row] * _L[_row + _matDim * j];
        _L[_row + _matDim * j] = 0;
    }
    __syncthreads();
}
__global__ void cudaUSolver(double *_U, double *_coeff, int _matDim, int _row) {  // now solving for X in UX=W
    if (_matDim % blockDim.x != 0)
        assert(false);
    int rowPerThread = _matDim / blockDim.x / gridDim.x;
    int firstRow = rowPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
    int lastRow = firstRow + rowPerThread;

    if (lastRow > _row)
        lastRow = _row;

    for (int j = firstRow; j < lastRow; j++) {  // each thread
        _coeff[j] -= _coeff[_row] * _U[_row + _matDim * j];
        _U[_row + _matDim * j] = 0;
    }
    __syncthreads();
}

double *LUSolverPrep(double **_L, double **_U, double **_Coeff) {  // Coeff will be our result of (tI+k)^-1*F
    double *LFlat, *UFlat, *CoeffFlat;
    LFlat = new double[n * n];
    UFlat = new double[n * n];
    CoeffFlat = new double[m * m];

    matrixConcat(_L, LFlat, n);
    matrixConcat(_U, UFlat, n);
    matrixConcat(_Coeff, CoeffFlat, m);

    double *cLFlat, *cUFlat, *cCoeffFlat;
    cudaMalloc(&cLFlat, n * n * sizeof(double));
    cudaMalloc(&cUFlat, n * n * sizeof(double));
    cudaMalloc(&cCoeffFlat, m * m * sizeof(double));

    cudaMemcpy(cLFlat, LFlat, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cUFlat, UFlat, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cCoeffFlat, CoeffFlat, m * m * sizeof(double), cudaMemcpyHostToDevice);

    int TRHEAD_NUM = 4;
    dim3 dimBlock(TRHEAD_NUM, 1);
    dim3 dimGrid(n / TRHEAD_NUM, 1);
    if (n % TRHEAD_NUM != 0)
        assert(false);

    cudaLUNormalizer<<<dimGrid, dimBlock>>>(cLFlat, cCoeffFlat, n);
    for (int i = 0; i < n; i++) {  // solving LW=F
        cudaLSolver<<<dimGrid, dimBlock>>>(cLFlat, cCoeffFlat, n, i);
    }
    cudaLUNormalizer<<<dimGrid, dimBlock>>>(cUFlat, cCoeffFlat, n);
    for (int i = n - 1; i >= 0; i--) {  // solving UX=W
        cudaUSolver<<<dimGrid, dimBlock>>>(cUFlat, cCoeffFlat, n, i);
    }

    cudaMemcpy(CoeffFlat, cCoeffFlat, m * m * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(LFlat, cLFlat, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(UFlat, cUFlat, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    matrixUnConcat(_L, LFlat, n);
    matrixUnConcat(_U, UFlat, n);
    matrixUnConcat(_Coeff, CoeffFlat, m);
    return CoeffFlat;
}
void matrixPrinter(double **_mat, int _matDim) {
    for (int i = 0; i < _matDim; i++) {
        for (int j = 0; j < _matDim; j++) {
            printf("%f\t", _mat[i][j]);
        }
        printf("\n");
    }
}

// void matrixMulPrep(double **A, double **B, double **C, int matDim) {
//     double *AFlat, *BFlat, *CFlat;
//     AFlat = new double[matDim * matDim];
//     BFlat = new double[matDim * matDim];
//     CFlat = new double[matDim * matDim];

//     matrixConcat(A, AFlat, matDim);
//     matrixConcat(B, BFlat, matDim);

//     double *cAFlat, *cBFlat, *cCFlat;

//     cudaMalloc(&cAFlat, matDim * matDim * sizeof(double));
//     cudaMalloc(&cBFlat, matDim * matDim * sizeof(double));
//     cudaMalloc(&cCFlat, matDim * matDim * sizeof(double));

//     cudaMemcpy(cAFlat, AFlat, matDim * matDim * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(cBFlat, BFlat, matDim * matDim * sizeof(double), cudaMemcpyHostToDevice);

//     dim3 dimBlock(4, 4);
//     dim3 dimGrid(matDim / dimBlock.x, matDim / dimBlock.y);

//     cudaMatrixMul<<<dimGrid, dimBlock>>>(cAFlat, cBFlat, cCFlat, matDim);

//     cudaMemcpy(CFlat, cCFlat, matDim * matDim * sizeof(double), cudaMemcpyDeviceToHost);
//     matrixUnConcat(C, CFlat, matDim);
// }

__global__ void cudaMatrixMul(double *cA, double *cB, double *cC, int commonSize) {
    double Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < commonSize; ++e)
        Cvalue += cA[row * commonSize + e] * cB[e * commonSize + col];
    cC[row * commonSize + col] = Cvalue;
}
void matrixMul(double **a, double **b, double **c, int matDim) {
    double temp;
    for (int i = 0; i < matDim; i++) {
        for (int j = 0; j < matDim; j++) {
            temp = 0;
            for (int k = 0; k < matDim; k++) {  // common dimension
                temp += a[i][k] * b[k][j];
            }
            c[i][j] = temp;
        }
    }
}

// void LSolver(double **_L, double *_B, int _matDim) {  // Solving LW=B knowing L is lower-triangular assuming L is matDim*matDim and W and B are matDim*1
//     for (int i = 0; i < _matDim; i++) {               // i is row
//         _B[i] /= _L[i][i];
//         _L[i][i] = 1;
//         for (int j = i + 1; j < _matDim; j++) {  // j is column
//             _B[j] -= _B[i] * _L[j][i];
//             _L[j][i] = 0;
//         }
//     }
// }
// void USolver(double **_U, double *_B, int _matDim) {  // We assumed that LUA=B and then UA=W. After solving for LW=B now we have to find UA=W knowing that U is upper-triangular and U is matDim*matDim and A and W are matDim*1
//     for (int i = _matDim - 1; i >= 0; i--) {          // i is row
//         _B[i] /= _U[i][i];
//         _U[i][i] = 1;
//         for (int j = i - 1; j >= 0; j--) {  // j is column
//             _B[j] -= _B[i] * _U[j][i];
//             _U[j][i] = 0;
//         }
//     }
// }
// void matrixTranspose(double **input, int matDim) {
//     double temp;
//     for (int i = 0; i < matDim; i++) {
//         for (int j = i; j < matDim; j++) {
//             temp = input[i][j];
//             input[i][j] = input[j][i];
//             input[j][i] = temp;
//         }
//     }
// }