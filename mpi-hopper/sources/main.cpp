#include <mpi.h>
#include <math.h>
#include <unistd.h>

void initMPI(int &argc, char **&argv);

void fillMatrices(int matrixDimensions, double *matrixA, double *matrixB);

void dns(double *matrixA, double *matrixB);

void blockAndDistribute(int processorCount, int matrixDimension, double *pDouble, double *matrixB);

void waitForDebugger();

/**
 * The rank of this processor
 */
int rank;

/**
 * The maximum rank (this is the total number of processors in the system)
 */
int maxRank;
int coordinates[3];
MPI_Comm iComm, jComm, kComm, ijComm;

/* Print a header for results output */
void resultHeader() {
    printf("Dims  No. Proc.  Avg. RT / Dev. (Eff.)\n");
}

/* Print the stats for 1 run */
void writeResult(int full_dim, int procs, double rt, double dev, double eff) {
    printf("%-5i %-10i %-5.5f / %-5.5f (%-5.5f)\n", full_dim, procs, rt, dev, eff);
}

/* Average and standard deviation */
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

int main(int argc, char **argv) {
    /* Launch MPI, if we we're launching from the command line */
    if (getenv("OMPI_COMM_WORLD_RANK") == NULL) {
        // TODO I don't think this will work on hopper. We should look into how this is really done.
        if (argc == 3) {
            char **args = (char **) calloc(6, sizeof(char *));
            args[0] = (char *) "mpirun";
            args[1] = (char *) "-np";
            args[2] = argv[1]; // Number of processors
            args[3] = (char *) "dnsmat";
            args[4] = argv[1]; // Number of processors
            args[5] = argv[2]; // Matrix dimensions
            execvp("mpirun", args);
            exit(0);
        }
        else {
            printf("Wrong number of parameters\n");
            exit(-1);
        }
    }

    int processorCount = atoi(argv[1]);
    int matrixDimensions = atoi(argv[2]);
    initMPI(argc, argv);

    /* Statistics */
    double startTime = 0.0, endTime = 0.0, avg, dev; /* Timing */
    double times[10]; /* Times for all runs */

    /* Write header */
    if (rank == 0) {
        resultHeader(); // TODO ? Todo what? Print/append result i guess?
    }

    /* Make and allocate matrices */
    double *matrixA = NULL;
    double *matrixB = NULL;
    if (rank == 0) {
        matrixA = (double *) malloc(sizeof(double) * matrixDimensions * matrixDimensions);
        matrixB = (double *) malloc(sizeof(double) * matrixDimensions * matrixDimensions);
        fillMatrices(matrixDimensions, matrixA, matrixB);
    }
    blockAndDistribute(processorCount, matrixDimensions, matrixA, matrixB);
#if false
    /* Run each config 10 times */
    for (int k = 0; k < 10; k++) {
        /* Start timer */
        MPI_Barrier(MPI_COMM_WORLD); // TODO Do we want to sync on all processors?
        if (rank == 0) {
            startTime = MPI_Wtime();
        }

        /* Do work */
        dns(matrixA, matrixB);

        /* End timer */
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            endTime = MPI_Wtime();
            times[k] = endTime - startTime;
        }
        /* Reset matrices */
        fillMatrices(matrixDimensions, matrixA, matrixB);
    }
    /* Destroy matrices */
    free(matrixA);
    free(matrixB);
#endif

    /* Print stats */
    if (rank == 0) {
        avg = average(10, times, &dev);
        /*
         *
         * TODO Calculate efficiency:
         * For determining the speedup and the efficiency you shall not compare your measured parallel runtimes to
         * actual runtimes using p=1, but you shall assume a 8.4 Gflop/s peak performance per processor and infer the
         * sequential runtime for p=1 based on that assumption. (Therefore the column "p=1" will have efficiency
         * values <1.0 that are identical to the fraction of peak performance based on Cray's LibSci runtime
         * measurements from the first mandatory assignment.)
         */
        writeResult(matrixDimensions, processorCount, avg, dev, 0/* EFFICIENCY */);
    }

    /* Exit program */
    MPI_Finalize();
    return 0;
}

void blockAndDistribute(int processorCount, int matrixDimension, double *matrixA, double *matrixB) {
    if (coordinates[2] != 0) return;

    double *preparedMatrixA = NULL;
    double *preparedMatrixB = NULL;

    int sendCount[processorCount];
    int displacements[processorCount];

    int length = (int) ceil(cbrt(processorCount));
    int blockLength = matrixDimension / length;

    double *receivedMatrixA = (double *) malloc(sizeof(double) * blockLength * blockLength);
    double *receivedMatrixB = (double *) malloc(sizeof(double) * blockLength * blockLength);

    if (rank == 0) {
        preparedMatrixA = (double *) malloc(sizeof(double) * matrixDimension * matrixDimension);
        preparedMatrixB = (double *) malloc(sizeof(double) * matrixDimension * matrixDimension);

        for (int i = 0; i < length; i++) {
            for (int j = 0; j < length; j++) {
                for (int k = 0; k < blockLength; k++) {
                    // Offset into the prepared matrix
                    int offsetPreparedA = j * length * (blockLength * blockLength) +
                                          i * (blockLength * blockLength) +
                                          k * blockLength;

                    int offsetPreparedB = i * length * (blockLength * blockLength) +
                                          j * (blockLength * blockLength) +
                                          k * blockLength;

                    // The start of the matrix
                    int offsetMatrix = j * matrixDimension * blockLength +
                                       i * blockLength +
                                       k * matrixDimension;

                    // Copy them into the prepared matrices
                    memcpy(&preparedMatrixA[offsetPreparedA], &matrixA[offsetMatrix], sizeof(double) * blockLength);
                    memcpy(&preparedMatrixB[offsetPreparedB], &matrixB[offsetMatrix], sizeof(double) * blockLength);
                }
            }
        }
    }

    for (int i = 0; i < processorCount; i++) {
        sendCount[i] = blockLength * blockLength;
        displacements[i] = i * blockLength * blockLength;
    }

    MPI_Scatterv(preparedMatrixA, sendCount, displacements, MPI_DOUBLE, receivedMatrixA, blockLength * blockLength,
                 MPI_DOUBLE, 0, ijComm);
    MPI_Scatterv(preparedMatrixB, sendCount, displacements, MPI_DOUBLE, receivedMatrixB, blockLength * blockLength,
                 MPI_DOUBLE, 0, ijComm);

#ifdef DEBUG
    MPI_Barrier(ijComm);
    if (rank == 0) {
        for (int i = 0; i < matrixDimension * matrixDimension; i++) {
            if (i % matrixDimension == 0) {
                printf("\n");
            }
            printf("%.2f\t", matrixA[i]);
        }
        printf("\n");
        printf("----------------------------------------------\n");
    }

    int commRank;
    MPI_Comm_rank(ijComm, &commRank);

    for (int iwant = 0; iwant < 16; iwant++) {
        MPI_Barrier(ijComm);
        if (commRank == iwant) {
            printf("FOR RANK: %d", iwant);
            for (int i = 0; i < blockLength * blockLength; i++) {
                if (i % blockLength == 0) {
                    printf("\n");
                }
                printf("%.2f\t", receivedMatrixA[i]);
            }
            printf("\n");
        }
    }
#endif

    if (rank == 0) {
        free(preparedMatrixA);
        free(preparedMatrixB);
    }
}

void fillMatrices(int matrixDimensions, double *matrixA, double *matrixB) {
    for (int i = 0; i < matrixDimensions * matrixDimensions; i++) {
        matrixA[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
        matrixB[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
    }
}

void initMPI(int &argc, char **&argv) {
    MPI_Comm gridCommunicator;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &maxRank);

    int processesForEachDimension = (int) ceil(cbrt(maxRank));
    int dimensions[3] = {processesForEachDimension, processesForEachDimension, processesForEachDimension};
    int periods[3] = {false, false, false};

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

    printf("rank= %d coordinates= %d %d %d\n", rank, coordinates[0], coordinates[1], coordinates[2]);
}

void dns(double *matrixA, double *matrixB) {

}

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