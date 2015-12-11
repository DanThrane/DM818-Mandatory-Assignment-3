#include <stdlib.h>
#include <mpi.h>
#include "matrix.h"

void fillMatrices(int matrixDimensions, double *matrixA, double *matrixB) {
    for (int i = 0; i < matrixDimensions * matrixDimensions; i++) {
        matrixA[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
        matrixB[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
        //matrixA[i] = i;
        //matrixB[i] = i;
    }
}

void sumMatrices(void *in, void *inout, int *length, MPI_Datatype *type) {
    double *a = (double *) in;
    double *b = (double *) inout;
    for (int i = 0; i < *length; i++) {
        b[i] += a[i];
    }
}