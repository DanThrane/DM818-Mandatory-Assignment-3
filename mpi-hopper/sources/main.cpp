#include <mpi.h>
#include <math.h>
#include <unistd.h>

void initMPI(int &argc, char **&argv);

void fillMatrices(int matrixDimensions, double **matrixA, double **matrixB);

void dns(double *matrixA, double *matrixB);

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
        dns(matrixA, matrixB);

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

#define SIZE 64
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

void initMPI(int &argc, char **&argv) {
    int maxRank;
    int rank;
    int neighbors[4];
    int reorder = 0;
    int dimensions[3] = {4, 4, 4};
    int periods[3] = {false, false, false};
    int coordinates[3];

    MPI_Comm gridCommunicator;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &maxRank);

    if (maxRank == SIZE) {
        MPI_Cart_create(MPI_COMM_WORLD, 3, dimensions, periods, reorder, &gridCommunicator);
        MPI_Comm_rank(gridCommunicator, &rank);
        MPI_Cart_coords(gridCommunicator, rank, 3, coordinates);
        MPI_Cart_shift(gridCommunicator, 0, 1, &neighbors[UP], &neighbors[DOWN]);
        MPI_Cart_shift(gridCommunicator, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);

        printf("rank= %d coordinates= %d %d %d  neighbors(u,d,l,r)= %d %d %d %d\n",
               rank, coordinates[0], coordinates[1], coordinates[2], neighbors[UP], neighbors[DOWN], neighbors[LEFT],
               neighbors[RIGHT]);
    } else {
        printf("Must specify %d processors. Terminating.\n", SIZE);
    }

    MPI_Finalize();
    exit(0);
}

void dns(double *matrixA, double *matrixB) {

}