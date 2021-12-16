#include "majorProjectCPU.H"

double **K, **I, **F, **KStar;
double *IFlat, *KFlat, *FFlat, *KStarFlat;
double *x, *y;
double xR, yR, l1, l2, t, FStar, RealF;
int m, n;

double **L, **U;

int main() {
    initialization();
    preparationHost();
    solver();

    return 0;
}
void solver() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            I[i][j] += t;
            I[i][j] += K[i][j];
        }
    }
    LUdecomposition(I, L, U, n);
    LSolver(L, FFlat, n);
    USolver(U, FFlat, n);
    double FStar;
    for (int i = 0; i < n; i++)
        FStar += KStarFlat[i] * FFlat[i];
    printf("Real= %f \t Pred=%f \n", RealF, FStar);
}
void preparationHost() {
    {
        for (int i = 0; i < m; i++) {
            x[i] = i * 1.00 / (m + 1);
            y[i] = i * 1.00 / (m + 1);
        }
    }

    {
        for (int i = 0; i < m; i++) {  // initializing matrix F
            for (int j = 0; j < m; j++) {
                double d = rand() * 1.00 / (RAND_MAX);
                d -= 0.5;
                d /= 10;
                F[i][j] = 1 - (pow(x[i] - 0.5, 2) + pow(y[j] - 0.5, 2)) + d;
            }
        }
        matrixConcat(F, FFlat, m, m);
    }

    {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    I[i][j] = 1;
                } else
                    I[i][j] = 0;
            }
        }
        matrixConcat(I, IFlat, n, n);
    }

    {
        double temp;
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
        matrixConcat(K, KFlat, n, n);
    }

    {
        double temp;
        for (int i = 0; i < m; i++) {  // first point
            for (int j = 0; j < m; j++) {
                temp = (pow(x[i] - xR, 2)) / (2 * l1 * l1) + (pow(y[j] - yR, 2)) / (2 * l2 * l2);  // row of first point and column of second point
                temp = exp(-temp) / (sqrt(2 * M_PI));
                KStar[i][j] = temp;
            }
        }
        matrixConcat(KStar, KStarFlat, m, m);
    }
}

void initialization() {
    xR = 0.2;
    yR = 0.3;
    m = 32;
    l1 = 1.0;
    l2 = 1.0;
    t = 0.01;
    RealF = 1 - (pow((xR - 0.5), 2) + pow((yR - 0.5), 2));

    n = m * m;
    x = new double[m];
    y = new double[m];

    FFlat = new double[n];
    KStarFlat = new double[n];
    IFlat = new double[n * n];
    KFlat = new double[n * n];

    for (int i = 0; i < n; i++) {
        L = new double *[n];
        U = new double *[n];
        for (int j = 0; j < n; j++) {
            L[j] = new double[n];
            U[j] = new double[n];
        }
    }

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

void LUdecomposition(double **input, double **L, double **U, int matrixDim) {
    for (int i = 0; i < matrixDim; i++) {
        // Upper Triangular
        for (int k = i; k < matrixDim; k++) {
            // Summation of L(i, j) * U(j, k)
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[i][j] * U[j][k]);

            // Evaluating U(i, k)
            U[i][k] = input[i][k] - sum;
        }

        // Lower Triangular
        for (int k = i; k < n; k++) {
            if (i == k)
                L[i][i] = 1;  // Diagonal as 1
            else {
                // Summation of L(k, j) * U(j, i)
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (L[k][j] * U[j][i]);

                // Evaluating L(k, i)
                L[k][i] = (input[k][i] - sum) / U[i][i];
            }
        }
    }
}
void matrixConcat(double **input, double *output, int xSize, int ySize) {
    for (int y = 0; y < ySize; y++) {      // y direction
        for (int x = 0; x < xSize; x++) {  // x direction
            output[x + y * xSize] = input[x][y];
        }
    }
}
void matrixTranspose(double **input, int matDim) {
    double temp;
    for (int i = 0; i < matDim; i++) {
        for (int j = i; j < matDim; j++) {
            temp = input[i][j];
            input[i][j] = input[j][i];
            input[j][i] = temp;
        }
    }
}

void matrixMul(double **a, double **b, double **c, int aRow, int aCol, int bCol) {
    double temp;
    for (int i = 0; i < aRow; i++) {
        for (int j = 0; j < bCol; j++) {
            temp = 0;
            for (int k = 0; k < aCol; k++) {  // common dimension
                temp += a[i][k] * b[k][j];
            }
            c[i][j] = temp;
        }
    }
}
void LSolver(double **_L, double *_B, int _matDim) {  // Solving LW=B knowing L is lower-triangular assuming L is matDim*matDim and W and B are matDim*1
    for (int i = 0; i < _matDim; i++) {               // i is row
        _B[i] /= _L[i][i];
        _L[i][i] = 1;
        for (int j = i + 1; j < _matDim; j++) {  // j is column
            _B[j] -= _B[i] * _L[j][i];
            _L[j][i] = 0;
        }
    }
}
void USolver(double **_U, double *_B, int _matDim) {  // We assumed that LUA=B and then UA=W. After solving for LW=B now we have to find UA=W knowing that U is upper-triangular and U is matDim*matDim and A and W are matDim*1
    for (int i = _matDim - 1; i >= 0; i--) {          // i is row
        _B[i] /= _U[i][i];
        _U[i][i] = 1;
        for (int j = i - 1; j >= 0; j--) {  // j is column
            _B[j] -= _B[i] * _U[j][i];
            _U[j][i] = 0;
        }
    }
}