#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "comm.h"
#include "matrix.h"

void initMPI(int &argc, char **&argv);
void fillMatrices(int matrixDimensions, double **matrixA, double **matrixB);
void createMpiTopology(int processorCount , int matrixDimensions);

void createMpiTopology(int argc, char **argv);
/**
 * The rank of this processor
 */
int rank;

/**
 * The maximum rank (this is the total number of processors in the system)
 */
int maxRank;

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

    /* Make cartesian grid */
    createMpiTopology(processorCount ,matrixDimensions);
    /* Make and allocate matrices */
    double *matrixA = (double *) malloc(sizeof(double) * matrixDimensions * matrixDimensions);
    double *matrixB = (double *) malloc(sizeof(double) * matrixDimensions * matrixDimensions);
    if (rank == 0) {
        fillMatrices(matrixDimensions, &matrixA, &matrixB);
    }

    /* Run each config 10 times */
    for (int k = 0; k < 10; k++) {
        /* Start timer */
        MPI_Barrier(MPI_COMM_WORLD); // TODO Do we want to sync on all processors?
        if (rank == 0) {
            startTime = MPI_Wtime();
        }

        /* Do work */
        // do dns algorithm

        /* End timer */
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            endTime = MPI_Wtime();
            times[k] = endTime - startTime;
        }
        /* Reset matrices */
        fillMatrices(matrixDimensions, &matrixA, &matrixB);
    }
    /* Destroy matrices */
    free(matrixA);
    free(matrixB);

    /* Print stats */
    if (rank == 0) {
        avg = average(10, times, &dev);
        /*
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

void fillMatrices(int matrixDimensions, double **matrixA, double **matrixB) {
    for (int i = 0; i < matrixDimensions * matrixDimensions; i++) {
        *matrixA[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
        *matrixB[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
    }
}

void createMpiTopology(int processorCount , int matrixDimensions) {
    int maxRank;
    MPI_Comm_size(MPI_COMM_WORLD, &maxRank);

    #ifdef DEBUG
    if (maxRank != processorCount) {
        printf("Mismatch in proc count between mpi and argument!\n");
        exit(-1);
    }
    #endif

    int pForEachDim = maxRank/3; // todo: Figure out how many processes we got.
    int processPerDim [3]= {pForEachDim, pForEachDim, pForEachDim};
    int period [3]= {1,1,1};

    //http://www.mpich.org/static/docs/v3.1/www3/MPI_Cart_create.html
    MPI_Comm newComm = 0;
    MPI_Cart_create(
            MPI_COMM_WORLD,
            /*# of dimensions*/ 3,
            /*# of proc*/ processPerDim,
            /*grid is periodic*/ period,
            /*no reordering*/ 0,
            /*new comm*/ &newComm);

    #ifdef DEBUG
    if (3 * pForEachDim != processorCount) {
        printf("Mismatch in proc count between sum of each proc and total!\n");
        exit(-1);
    }
    #endif

    //Determines the rank of the calling process in the communicator:
    int rankInDim;
    MPI_Comm_rank(newComm, &rankInDim);

    // This wont work dynamically, has to be another way. MPI_Cart_sub?
    int rank_source, rank_desta, rank_destb, rank_destc, rank_destd;
    MPI_Cart_shift(newComm, 0,1,&rank_source, &rank_desta);
    MPI_Cart_shift(newComm, 1,1,&rank_source, &rank_destb);
    MPI_Cart_shift(newComm, 2,1,&rank_source, &rank_destc);
    MPI_Cart_shift(newComm, 3,1,&rank_source, &rank_destd);

}

void initMPI(int &argc, char **&argv) {
    MPI_Init(&argc, &argv);
    // Not needed:
    //MPI_Comm_size(MPI_COMM_WORLD, &maxRank);
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}
