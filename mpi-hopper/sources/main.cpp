#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "comm.h"
#include "matrix.h"

void initMPI(int &argc, char **&argv);

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

    /* Statistics */
    double startTime = 0.0, endTime = 0.0, avg, dev; /* Timing */
    double times[10]; /* Times for all runs */

    initMPI(argc, argv);

    /* Get MPI process stats */
    /*
    ...
    */

    /* Get parameters */
    if (argc == 3) {
        /* Get number of processes */

        /* Get maximum matrix dimension */
    }
    else {
        printf("Wrong number of parameters\n");
        exit(-1);
    }

    /* Write header */
    if (rank == 0) {
        resultHeader(); // TODO ?
    }

    /* Make cartesian grid */

    /* Make and allocate matrices */

    /* Run each config 10 times */
    for (int k = 0; k < 10; k++) {
        /* Start timer */
        MPI_Barrier(MPI_COMM_WORLD); // TODO Do we want to sync on all processors?
        if (rank == 0) {
            startTime = MPI_Wtime();
        }

        /* Do work */

        /* End timer */
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            endTime = MPI_Wtime();
            times[k] = endTime - startTime;
        }
        /* Reset matrices */
    }
    /* Destroy matrices */

    /* Print stats */
    if (rank == 0) {
        avg = average(10, times, &dev);
        writeResult(0/* MATRIX SIZE */, 0/* GRID SIZE */, avg, dev, 0/* EFFICIENCY */);
    }

    /* Exit program */
    MPI_Finalize();
    return 0;
}

void initMPI(int &argc, char **&argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &maxRank);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}
