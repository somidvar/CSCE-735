// #include "majorProject.H"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

double **K, **I;
double *f, *kStar;
double *x, *y;
int m;
int n;

double** A;

__global__ void devicePreparation(double* cX,double*Cy,double* cF) {
    // printf("block x=%d,block y=%d, thread Idx=%d, thread Idy=%d\n",);
    // calculate A=tI+K
    // now using f*=tran(k*)A^-1f we want to know f*
    // assume A^-1f=z-->f=Az-->so, instead of calculating A^-1 we can do below:
    // decompose A into LU and f=LUz
    // assume Uz=w and Lw=f where we know f and L so we solve for w easiliy as L is lower triangular (gauss method)
    // after finding w, now we do the similar thing for w=Uz where we know w and U and we find z
    // finally multiply tran(k*) into z to find f*

}




// void preparationHost() {
//     for (int i = 0; i < m; i++) {
//         x[i] = i * 1.00 / (m + 1);
//         y[i] = i * 1.00 / (m + 1);
//     }
//     for (int counter = 0; counter < n; counter++) {
//         int i = counter % m;
//         int j = counter / m;
//         double d = rand() * 1.00 / (RAND_MAX);
//         d -= 0.5;
//         d /= 10;
//         f[counter] = 1 - (pow(x[i] - 0.5, 2) + pow(y[j] - 0.5, 2)) + d;
//     }
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             if (i == j) {
//                 I[i][j] = 1;
//             } else
//                 I[i][j] = 0;
//         }
//     }
// }

// void initialization() {
//     m = 10;
//     n = m * m;
//     x = new double[m];
//     y = new double[m];

//     f = new double[n];
//     kStar = new double[n];

//     for (int i = 0; i < n; i++) {
//         I = new double*[n];
//         K = new double*[n];
//         for (int j = 0; j < n; j++) {
//             I[j] = new double[n];
//             K[j] = new double[n];
//         }
//     }
// }

// void LUdecomposition(float a[10][10], float l[10][10], float u[10][10], int n) {
//     int i = 0, j = 0, k = 0;
//     for (i = 0; i < n; i++) {
//         for (j = 0; j < n; j++) {
//             if (j < i)
//                 l[j][i] = 0;
//             else {
//                 l[j][i] = a[j][i];
//                 for (k = 0; k < i; k++) {
//                     l[j][i] = l[j][i] - l[j][k] * u[k][i];
//                 }
//             }
//         }
//         for (j = 0; j < n; j++) {
//             if (j < i)
//                 u[i][j] = 0;
//             else if (j == i)
//                 u[i][j] = 1;
//             else {
//                 u[i][j] = a[i][j] / l[i][i];
//                 for (k = 0; k < i; k++) {
//                     u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
//                 }
//             }
//         }
//     }
// }
int main() {
    // initialization();
    // preparationHost();

    double *cX,*cY,*cF;

    cudaMalloc(&cX, m*sizeof(double)); 
    cudaMalloc(&cY, m*sizeof(double)); 
    cudaMalloc(&cF, n*sizeof(double)); 

    cudaMemcpy(cX,x,n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(cY,y,n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(cF,f,n*sizeof(double),cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(n,n);
    devicePreparation<<<1, 1>>> (cX,cY,cF);

    return 0;
}