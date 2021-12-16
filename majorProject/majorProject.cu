#include "majorProject.H"

int main() {
    xR = 0.25;
    yR = 0.35;
    m = 32;
    l1 = 1.0;
    l2 = 1.0;
    t = 0.01;

    initialization();
    printf("Initialization is done\n");
    preparationGPU();
    printf("Preparation is done\n");
    solver();
    return 0;
}
void initialization() {
    RealF = 1 - (pow((xR - 0.5), 2) + pow((yR - 0.5), 2));

    n = m * m;
    x = new float[m];
    y = new float[m];

    FFlat = new float[n];
    KStarFlat = new float[n];
    KFlat = new float[n * n];

    K = new float *[n];

    for (int j = 0; j < n; j++) {
        K[j] = new float[n];
    }
    F = new float *[m];
    KStar = new float *[m];
    for (int j = 0; j < m; j++) {
        F[j] = new float[m];
        KStar[j] = new float[m];
    }
}
void preparationGPU() {
    float *cFFlat, *cX, *cY, *cKFlat, *cKStarFlat;

    cudaMalloc(&cX, m * sizeof(float));
    cudaMalloc(&cY, m * sizeof(float));
    cudaMalloc(&cFFlat, m * m * sizeof(float));
    cudaMalloc(&cKFlat, n * n * sizeof(float));
    cudaMalloc(&cKStarFlat, m * m * sizeof(float));

    int TRHEAD_NUM = 16;

    dim3 dimBlockXY(TRHEAD_NUM, 1);
    dim3 dimGridXY(m / TRHEAD_NUM, 1);

    dim3 dimBlockF(TRHEAD_NUM, TRHEAD_NUM);
    dim3 dimGridF(m / TRHEAD_NUM, m / TRHEAD_NUM);

    dim3 dimBlockK(TRHEAD_NUM, TRHEAD_NUM);
    dim3 dimGridK(m / TRHEAD_NUM, m / TRHEAD_NUM);

    dim3 dimBlockKStar(TRHEAD_NUM, TRHEAD_NUM);
    dim3 dimGridKStar(m / TRHEAD_NUM, m / TRHEAD_NUM);

    if (m % TRHEAD_NUM != 0)
        assert(false);

    cudaInitXY<<<dimGridF, dimBlockF>>>(cX, cY, m);

    cudaMemcpy(x, cX, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, cY, m * sizeof(float), cudaMemcpyDeviceToHost);

    cudaInitF<<<dimGridF, dimBlockF>>>(cFFlat, cX, cY, m);
    cudaMemcpy(FFlat, cFFlat, m * m * sizeof(float), cudaMemcpyDeviceToHost);
    matrixUnConcat(F, FFlat, m);

    cudaInitK<<<dimGridK, dimBlockK>>>(cKFlat, cX, cY, m, l1, l2);
    cudaMemcpy(KFlat, cKFlat, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    matrixUnConcat(K, KFlat, n);

    cudaInitKStar<<<dimGridKStar, dimBlockKStar>>>(cKStarFlat, cX, cY, m, l1, l2, xR, yR);
    cudaMemcpy(KStarFlat, cKStarFlat, m * m * sizeof(float), cudaMemcpyDeviceToHost);
    matrixUnConcat(KStar, KStarFlat, m);

    cudaFree(cFFlat);
    cudaFree(cX);
    cudaFree(cY);
    cudaFree(cKFlat);
    cudaFree(cKStarFlat);
}

__global__ void cudaInitK(float *_cK, float *_cX, float *_cY, int _matDim, float _L1, float _L2) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float temp = 0;
    for (int l = 0; l < _matDim; l++) {  // second point
        for (int k = 0; k < _matDim; k++) {
            temp = (pow(_cX[column] - _cX[k], 2)) / (2 * _L1 * _L1) +
                   (pow(_cY[row] - _cY[l], 2)) / (2 * _L2 * _L2);  // row of first point and column of second point
            temp = exp(-temp) / (sqrt(2 * M_PI));
            _cK[k + l * _matDim + (column + row * _matDim) * _matDim * _matDim] = temp;
        }
    }
}
__global__ void cudaInitKStar(float *_cKStar, float *_cX, float *_cY, int _matDim, float _L1, float _L2, float _xR, float _yR) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float temp = (pow(_cX[column] - _xR, 2)) / (2 * _L1 * _L1) +
                  (pow(_cY[row] - _yR, 2)) / (2 * _L2 * _L2);
    temp = exp(-temp) / (sqrt(2 * M_PI));
    _cKStar[column + _matDim * row] = temp;
}

__global__ void cudaInitF(float *_cF, float *_cX, float *_cY, int _matDim) {
    int perThread = _matDim / blockDim.x / gridDim.x;
    int row = perThread * (blockIdx.x * blockDim.x + threadIdx.x);
    int column = perThread * (blockIdx.y * blockDim.y + threadIdx.y);

    curandState_t state;
    int seed = row * _matDim + column;
    curand_init(seed, seed, 0, &state);
    float d = curand(&state) % 1000;
    d /= 1000;
    d -= 0.5;
    d /= 10;

    _cF[column + row * _matDim] = 1 - (pow(_cX[column] - 0.5, 2) + pow(_cY[row] - 0.5, 2)) + d;
}
__global__ void cudaInitXY(float *_cX, float *_cY, int _matDim) {
    int perThread = _matDim / blockDim.x / gridDim.x;
    int row = perThread * (blockIdx.x * blockDim.x + threadIdx.x);

    _cX[row] = row * 1.00 / (_matDim + 1);
    _cY[row] = row * 1.00 / (_matDim + 1);
}
void solver() {
    float **L, **U;
    for(int i=0;i<n*n;i++)
        KFlat[i+n*i]+=t + 1;

    matrixConcat(F, FFlat, m);

    L = new float *[n];
    U = new float *[n];
    for (int j = 0; j < n; j++) {
        L[j] = new float[n];
        U[j] = new float[n];
    }

    // LUDecomposition(K, L, U, n);
    LUDecompositionPrep(K, L, U);
    printf("Decomposition is done\n");
    // LUDecompositionTester(K, L, U);
    FFlat = LUSolverPrep(L, U, F);
    matrixConcat(KStar, KStarFlat, m);
    FStar = 0;
    for (int i = 0; i < n; i++) {
        FStar += FFlat[i] * KStarFlat[i];
    }
    printf("The predicted value=%2.4f and the real is %2.4f\n", FStar, RealF);
}
void LUDecompositionTester(float **_K, float **_L, float **_U) {
    float **temp;
    temp = new float *[n];
    for (int j = 0; j < n; j++) {
        temp[j] = new float[n];
    }

    matrixMul(_L, _U, temp, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            assert(fabs(_K[i][j] - temp[i][j]) < 0.001);
        }
    }
}
void LUDecompositionPrep(float **_K, float **_L, float **_U) {
    matrixConcat(_K, KFlat, n);

    float *LFlat = new float[n * n];
    float *UFlat = new float[n * n];

    float *cLFlat, *cUFlat, *cLUFlat, *cKFlat;

    cudaMalloc(&cLFlat, n * n * sizeof(float));
    cudaMalloc(&cUFlat, n * n * sizeof(float));
    cudaMalloc(&cLUFlat, n * n * sizeof(float));
    cudaMalloc(&cKFlat, n * n * sizeof(float));

    cudaMemcpy(cKFlat, KFlat, n * n * sizeof(float), cudaMemcpyHostToDevice);

    int TRHEAD_NUM = 16;
    dim3 dimBlockLU(TRHEAD_NUM, 1);
    dim3 dimGridLU(n / TRHEAD_NUM, 1);

    dim3 dimBlockMul(TRHEAD_NUM, TRHEAD_NUM);
    dim3 dimGridMul(n / TRHEAD_NUM, n / TRHEAD_NUM);

    if (n % TRHEAD_NUM != 0)
        assert(false);

    for (int i = 0; i < n; i++) {
        cudaMatrixMul<<<dimGridMul, dimBlockMul>>>(cLFlat, cUFlat, cLUFlat, n);
        cudaUFactorization<<<dimGridLU, dimBlockLU>>>(cKFlat, cLFlat, cUFlat, cLUFlat, n, i);
        cudaMatrixMul<<<dimGridMul, dimBlockMul>>>(cLFlat, cUFlat, cLUFlat, n);
        cudaLFactorization<<<dimGridLU, dimBlockLU>>>(cKFlat, cLFlat, cUFlat, cLUFlat, n, i);
    }

    cudaMemcpy(LFlat, cLFlat, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(UFlat, cUFlat, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    matrixUnConcat(_L, LFlat, n);
    matrixUnConcat(_U, UFlat, n);
}
__global__ void cudaUFactorization(float *_input, float *_L, float *_U, float *_LU, int _matDim, int _row) {
    int rowPerThread = _matDim / blockDim.x / gridDim.x;
    int firstRow = rowPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
    int lastRow = firstRow + rowPerThread;

    if (firstRow < _row)
        firstRow = _row;

    for (int k = firstRow; k < lastRow; k++) {
        _U[k + _row * _matDim] = _input[k + _row * _matDim] - _LU[k + _row * _matDim];
    }
}
__global__ void cudaLFactorization(float *_input, float *_L, float *_U, float *_LU, int _matDim, int _row) {
    int rowPerThread = _matDim / blockDim.x / gridDim.x;
    int firstRow = rowPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
    int lastRow = firstRow + rowPerThread;

    if (firstRow < _row)
        firstRow = _row;

    for (int k = _row; k < lastRow; k++) {
        _L[_row + k * _matDim] = (_input[_row + k * _matDim] - _LU[_row + k * _matDim]) / _U[_row + _row * _matDim];
    }
}
void matrixConcat(float **input, float *output, int matDim) {
    for (int x = 0; x < matDim; x++) {      // y direction
        for (int y = 0; y < matDim; y++) {  // x direction
            output[y + x * matDim] = input[x][y];
        }
    }
}

void matrixUnConcat(float **input, float *output, int matDim) {
    for (int x = 0; x < matDim; x++) {      // y direction
        for (int y = 0; y < matDim; y++) {  // x direction
            input[x][y] = output[y + x * matDim];
        }
    }
}

__global__ void cudaLUNormalizer(float *_A, float *_coeff, int _matDim) {  // diving each row to change the diagonal elements to 1 for A*W=Coeff
    if (_matDim % blockDim.x != 0)
        assert(false);
    int rowPerThread = _matDim / blockDim.x / gridDim.x;
    int firstRow = rowPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
    int lastRow = firstRow + rowPerThread;
    for (int i = firstRow; i < lastRow; i++) {  // each thread
        float temp = _A[i + i * _matDim];
        _coeff[i] /= temp;
        for (int j = 0; j < _matDim; j++) {
            _A[j + i * _matDim] /= temp;
        }
    }
}

__global__ void cudaLSolver(float *_L, float *_coeff, int _matDim, int _row) {  // assuming L*U=A and A^-1 * coeff=X-> coeff=A*X where A and coeff are known and X is unknown. L*U*X=Coeff->L*W=Coeff then we find W and then W=UX and we find X
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
}
__global__ void cudaUSolver(float *_U, float *_coeff, int _matDim, int _row) {  // now solving for X in UX=W
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
}

float *LUSolverPrep(float **_L, float **_U, float **_Coeff) {  // Coeff will be our result of (tI+k)^-1*F
    float *LFlat, *UFlat, *CoeffFlat;
    LFlat = new float[n * n];
    UFlat = new float[n * n];
    CoeffFlat = new float[m * m];

    matrixConcat(_L, LFlat, n);
    matrixConcat(_U, UFlat, n);
    matrixConcat(_Coeff, CoeffFlat, m);

    float *cLFlat, *cUFlat, *cCoeffFlat;
    cudaMalloc(&cLFlat, n * n * sizeof(float));
    cudaMalloc(&cUFlat, n * n * sizeof(float));
    cudaMalloc(&cCoeffFlat, m * m * sizeof(float));

    cudaMemcpy(cLFlat, LFlat, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cUFlat, UFlat, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cCoeffFlat, CoeffFlat, m * m * sizeof(float), cudaMemcpyHostToDevice);

    int TRHEAD_NUM = 16;
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

    cudaMemcpy(CoeffFlat, cCoeffFlat, m * m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(LFlat, cLFlat, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(UFlat, cUFlat, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    matrixUnConcat(_L, LFlat, n);
    matrixUnConcat(_U, UFlat, n);
    matrixUnConcat(_Coeff, CoeffFlat, m);
    return CoeffFlat;
}
void matrixPrinter(float **_mat, int _matDim) {
    for (int i = 0; i < _matDim; i++) {
        for (int j = 0; j < _matDim; j++) {
            printf("%2.2f ", _mat[i][j]);
        }
        printf("\n");
    }
}
__global__ void cudaMatrixMul(float *cA, float *cB, float *cC, int commonSize) {
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < commonSize; ++e)
        Cvalue += cA[row * commonSize + e] * cB[e * commonSize + col];
    cC[row * commonSize + col] = Cvalue;
}
void matrixMul(float **a, float **b, float **c, int matDim) {
    float temp;
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