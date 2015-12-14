#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "matrix_mul.h"
#include "matrix.h"

void initMPI(int argc, char **argv);
void dns();
void blockAndDistribute(int processorCount, int matrixDimension, double *pDouble, double *matrixB);
void distribute();
void broadcast();
void multiplyAndReduce();
void reduction(double *matrixC);
void checkResult(int n, double *A, double *B);
void debugPrintMatrix();
void waitForDebugger();

int rank;
int maxRank;

int coordinates[3];
MPI_Comm iComm, jComm, kComm, ijComm;

double *receivedMatrixA;
double *receivedMatrixB;
double *resultMatrix;

int blockLength;
MPI_Op matrixSum;

/*
 * Print a header for results output.
 * SOURCE: Skeleton code.
 */
void resultHeader() {
    printf("Dims  No. Proc.  Avg. RT / Dev. (Eff.)\n");
}

/* Print the stats for 1 run */
void writeResult(int full_dim, int procs, double rt, double dev, double eff) {
    printf("%-5i %-10i %-5.5f / %-5.5f (%-5.5f)\n", full_dim, procs, rt, dev, eff);
}

/**
 * Average and standard deviation.
 * SOURCE: Skeleton code.
 */
double average(int count, double *list, double *dev) {
    int i;
    double sum = 0.0, avg;

    for (i = 0; i < count; i++) {
        sum += list[i];
    }

    avg = sum / (double) count;

    if (dev != 0) {
        sum = 0.0;
        for (i = 0; i < count; i++) {
            sum += (list[i] - avg) * (list[i] - avg);
        }

        *dev = sqrt(sum / (double) count);
    }

    return avg;
}

/**
 * Main method
 * Iterates the algorithm 10 times and prints the average running time and efficiency.
 */
int main(int argc, char **argv) {
    int processorCount = atoi(argv[1]);
    int matrixDimensions = atoi(argv[2]);
    initMPI(argc, argv);

    if (processorCount != maxRank) {
        if (rank == 0) printf("Error! Number of processes specified are not identical! Exiting..\n");
        MPI_Finalize();
        exit(-1);
    }

    /* Padding calculations */
    int matrixDimPadded = matrixDimensions;
    if (matrixDimensions % (int)cbrt(processorCount))
        matrixDimPadded += matrixDimensions / (int)cbrt(processorCount) - matrixDimensions % (int)cbrt(processorCount);

    /* Statistics */
    double startTime = 0.0, endTime = 0.0, avg, dev; /* Timing */
    double times[10]; /* Times for all runs */

    /* Write header */
    if (rank == 0) {
        resultHeader();
    }

    /* Make and allocate matrices */
    double *matrixA = NULL;
    double *matrixB = NULL;
    /* Do work */
    for (int k = 0; k < 10; k++) {
        if (rank == 0) {
            matrixA = (double *) malloc(sizeof(double) * matrixDimPadded * matrixDimPadded);
            matrixB = (double *) malloc(sizeof(double) * matrixDimPadded * matrixDimPadded);
            memset(matrixA, 0, sizeof(double) * matrixDimPadded * matrixDimPadded);
            memset(matrixB, 0, sizeof(double) * matrixDimPadded * matrixDimPadded);
            fillMatrices(matrixDimensions, matrixA, matrixB);
        }
        // Todo: Ask Daniel!
        // Is it OK to assume initial distribution is already done?
        // or should this be done inside MPI timer?
        blockAndDistribute(processorCount, matrixDimPadded, matrixA, matrixB);
        if (coordinates[2] == 0) {
            resultMatrix = (double *) malloc(sizeof(double) * blockLength * blockLength);
        }

        /* Start timer */
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            startTime = MPI_Wtime();
        }

        /* Do work */
        dns();

        /* End timer */
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            endTime = MPI_Wtime();
            times[k] = endTime - startTime;
        }
        /* Verify result */
        if (coordinates[2] == 0) {
            //checkResult(matrixDimPadded, matrixA, matrixB);
        }
        /* Reset matrices */
        if (rank == 0) {
            free(matrixA);
            free(matrixB);
            free(resultMatrix);
        }
    }

    /* Print stats */
    if (rank == 0) {
        dev = 0;
        avg = average(10, times, &dev);

        // To calculate Ts i assume we need to figure out what Ts would be with 8.4Gflop/s.
        // Since multiplying two n-by-n matrices we would have O(n^3) multiplications this could be:
        // O(n^3) / 8.400.000.000
        // source: https://en.wikipedia.org/wiki/FLOPS
        double Ts = pow(matrixDimensions, 3)*2 / 8400000000;
        double Tp = avg;
        double S = Ts/Tp;
        double efficiency = S / processorCount;
        writeResult(matrixDimensions, processorCount, avg, dev, efficiency);

        printf("\n-------------------------------\n");
        printf("pcount: %i\n", processorCount);
        printf("Time for serial (Ts): %f \n", Ts);
        printf("Time for parallel (Tp): %f \n", Tp);
        printf("Speedup (S): %f \n", S);
        printf("Total for all p: %f\n", avg*processorCount);
        printf("-------------------------------\n");
    }

    MPI_Finalize();
    return 0;
}

/**
 * Initialize MPI and create the cartesian structure.
 */
void initMPI(int argc, char **argv) {
    MPI_Comm gridCommunicator;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &maxRank);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int pEachDimension = (int) cbrt(maxRank);
    int dimensions[3] = {pEachDimension, pEachDimension, pEachDimension};
    int periods[3] = {false, false, false};

    if ((pEachDimension * pEachDimension * pEachDimension) != maxRank) {
        if (rank == 0) printf("Error! Number of processes is not a 3dim cube! Exiting..\n");
        MPI_Finalize();
        exit(-1);
    }

    MPI_Cart_create(MPI_COMM_WORLD, 3, dimensions, periods, 0, &gridCommunicator);
    MPI_Comm_rank(gridCommunicator, &rank);
    MPI_Cart_coords(gridCommunicator, rank, 3, coordinates);

    int iDimensions[3] = {1, 0, 0};
    int jDimensions[3] = {0, 1, 0};
    int kDimensions[3] = {0, 0, 1};
    int ijDimensions[3] = {1, 1, 0};

    MPI_Cart_sub(gridCommunicator, iDimensions, &iComm);
    MPI_Cart_sub(gridCommunicator, jDimensions, &jComm);
    MPI_Cart_sub(gridCommunicator, kDimensions, &kComm);
    MPI_Cart_sub(gridCommunicator, ijDimensions, &ijComm);

    MPI_Op_create(sumMatrices, true, &matrixSum);
}

/**
 * Step A
 * Performs blocking of the matrices and performs initial distribution among processes
 * at k=0.
 */
void blockAndDistribute(int processorCount, int matrixDimension, double *matrixA, double *matrixB) {
    double *preparedMatrixA = NULL;
    double *preparedMatrixB = NULL;

    int sendCount[processorCount];
    int displacements[processorCount];

    int length = (int) cbrt(processorCount);
    blockLength = matrixDimension / length;

    receivedMatrixA = (double *) malloc(sizeof(double) * blockLength * blockLength);
    receivedMatrixB = (double *) malloc(sizeof(double) * blockLength * blockLength);

    if (coordinates[2] == 0) { // Only perform initial distribution to k = 0
        if (rank == 0) {
            preparedMatrixA = (double *) malloc(sizeof(double) * matrixDimension * matrixDimension);
            preparedMatrixB = (double *) malloc(sizeof(double) * matrixDimension * matrixDimension);

            for (int i = 0; i < length; i++) {
                for (int j = 0; j < length; j++) {
                    for (int k = 0; k < blockLength; k++) {
                        // Offset into the prepared matrix
                        int offsetPrepared = i * length * (blockLength * blockLength) +
                                             j * (blockLength * blockLength) +
                                             k * blockLength;

                        // The start of the matrix
                        int offsetMatrix = j * matrixDimension * blockLength +
                                           i * blockLength +
                                           k * matrixDimension;

                        // Copy them into the prepared matrices
                        memcpy(&preparedMatrixA[offsetPrepared], &matrixA[offsetMatrix], sizeof(double) * blockLength);
                        memcpy(&preparedMatrixB[offsetPrepared], &matrixB[offsetMatrix], sizeof(double) * blockLength);
                    }
                }
            }
        }

        for (int i = 0; i < processorCount; i++) {
            sendCount[i] = blockLength * blockLength;
            displacements[i] = i * blockLength * blockLength;
        }

        // Perform the communication:
        MPI_Scatterv(preparedMatrixA, sendCount, displacements, MPI_DOUBLE, receivedMatrixA,
                     blockLength * blockLength, MPI_DOUBLE, 0, ijComm);
        MPI_Scatterv(preparedMatrixB, sendCount, displacements, MPI_DOUBLE, receivedMatrixB,
                     blockLength * blockLength, MPI_DOUBLE, 0, ijComm);

        if (rank == 0) {
            free(preparedMatrixA);
            free(preparedMatrixB);
        }
    }
}

/**
 * The DNS algorithm,
 * assumes that initial distribution is already performed.
 */
void dns() {
    distribute();
    broadcast();
    multiplyAndReduce();
#if DEBUG
    debugPrintMatrix();
#endif
}

/**
 * Step B
 * Distribute along k dimension to k=j for A and k=i for B.
 */
void distribute() {
    // Distribute matrix A
    if (coordinates[2] == 0 && coordinates[1] != 0) {
        MPI_Send(receivedMatrixA, blockLength * blockLength, MPI_DOUBLE, coordinates[1], 0, kComm);
    } else if (coordinates[1] == coordinates[2] && coordinates[1] != 0) {
        MPI_Recv(receivedMatrixA, blockLength * blockLength, MPI_DOUBLE, 0, 0, kComm, MPI_STATUS_IGNORE);
    } /* else do nothing */

    if (coordinates[2] == 0 && coordinates[0] != 0) {
        MPI_Send(receivedMatrixB, blockLength * blockLength, MPI_DOUBLE, coordinates[0], 0, kComm);
    } else if (coordinates[0] == coordinates[2] && coordinates[2] != 0) {
        MPI_Recv(receivedMatrixB, blockLength * blockLength, MPI_DOUBLE, 0, 0, kComm, MPI_STATUS_IGNORE);
    } /* else do nothing */
}

/**
 * Step C
 * Broadcast elements along j dimension for A and i dimension for B.
 */
void broadcast() {
    MPI_Bcast(receivedMatrixA, blockLength * blockLength, MPI_DOUBLE, coordinates[2], jComm);
    MPI_Bcast(receivedMatrixB, blockLength * blockLength, MPI_DOUBLE, coordinates[2], iComm);
}

/*
 * Step D
 * Multiply local process' A and B together.
 * Then reduce all k>0 to k=0
 */
void multiplyAndReduce() {
    double *matrixC = (double *) malloc(sizeof(double) * blockLength * blockLength);
    memset(matrixC, 0, sizeof(double) * blockLength * blockLength);
    squareDgemm(blockLength, receivedMatrixA, receivedMatrixB, matrixC);
    reduction(matrixC);
    free(matrixC);
}

void reduction(double *matrixC) {
    MPI_Reduce(matrixC, resultMatrix, blockLength * blockLength, MPI_DOUBLE, matrixSum, 0, kComm);
}

/*
 * Checks a random element of a local process, to see if its correct.
 */
void checkResult(int n, double *A, double *B) {
    // Todo: This should be fixed so it checks for all locals. And use dot-product instead.
    if (rank == 0) {
        printf("Checking result...\n");
        /* Calculate expected result matrix */
        double *C = (double *) malloc(sizeof(double) * n * n);
        memset(C, 0, sizeof(double) * n * n);
        squareDgemm(n, A, B, C);

        /* random int between 0 and 19 */
        int r = rand() % blockLength;
        /* make sure numbers are equal within error diff of 0.001 */
        assert(resultMatrix[r] <= C[r]+0.0001 && resultMatrix[r] >= C[r]-0.0001);
        printf("Done!\n");
    }

}

/*
 * Prints a local process matrix, used for debugging only
 */
void debugPrintMatrix() {
    if (coordinates[2] == 0) {
        int commRank;
        MPI_Comm_rank(ijComm, &commRank);

        for (int iwant = 0; iwant < 16; iwant++) {
            MPI_Barrier(ijComm);
            if (commRank == iwant) {
                printf("FOR RANK: (%d, %d)", coordinates[0], coordinates[1]);
                for (int i = 0; i < blockLength * blockLength; i++) {
                    if (i % blockLength == 0) {
                        printf("\n");
                    }
                    printf("%.4f\t", resultMatrix[i]);
                }
                printf("\n");
            }
        }
    }
}

/* Used to pause execution, to attach a debugger */
void waitForDebugger() {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i) { // Set i to some value != 0 using the debugger to continue
        sleep(5);
    }
}