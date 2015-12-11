
#ifndef MATRIX_H
#define MATRIX_H

/* Function prototypes */
void fillMatrices(int matrixDimensions, double *matrixA, double *matrixB);
void sumMatrices(void *in, void *inout, int *length, MPI_Datatype *type);

#endif /* MATRIX_H */
