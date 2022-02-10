#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <omp.h>

#define DEBUG 0
#define BUF_SIZE 1024

/* ----------- Project 2 - Problem 1 - Matrix Mult -----------

    This file will multiply two matricies in a parallel manner

*/ // ------------------------------------------------------ //

// Prototype Functions
void readCSVtoMatrix(FILE* fp, long int** in_matrix);
void writeMatrixtoCSV(FILE* fp, long int** out_matrix, int n_col, int n_row);

/**
 * Main function
 * Performs matrix multiplication in parallel using OpenMP
 */
int main(int argc, char* argv[])
{
    // Catch console errors
    if (argc != 10)
    {
        printf("USE LIKE THIS: parallel_mult_mat_mat   mat_1.csv n_row_1 n_col_1   mat_2.csv n_row_2 n_col_2   num_threads   results_matrix.csv   time.csv\n");
        return EXIT_FAILURE;
    }

    // Get the input files
    FILE* inputMatrix1 = fopen(argv[1], "r");
    FILE* inputMatrix2 = fopen(argv[4], "r");

    char* p1;
    char* p2;

    // Get matrix 1's dims
    int n_row1 = strtol(argv[2], &p1, 10);
    int n_col1 = strtol(argv[3], &p2, 10);

    // Get matrix 2's dims
    int n_row2 = strtol(argv[5], &p1, 10);
    int n_col2 = strtol(argv[6], &p2, 10);

    // Get num threads
    int thread_count = strtol(argv[7], NULL, 10);

    // Get output files
    FILE* outputFile = fopen(argv[8], "w");
    FILE* outputTime = fopen(argv[9], "w");


    // Create and malloc the two input matrices and the output matrix
    long int** matrix1 = (long int**)malloc(n_col1 * sizeof(long int*));
    for(int i = 0; i < n_row1; i++)
        matrix1[i] = (long int*)malloc(n_row1 * sizeof(long int));

    long int** matrix2 = (long int**)malloc(n_col2 * sizeof(long int*));
    for(int i = 0; i < n_row2; i++)
        matrix2[i] = (long int*)malloc(n_row2 * sizeof(long int));

    
    // Determine the dims of the output matrix and initialize all cells to 0
    int out_col = n_col2;
    int out_row = n_row1;

    long int** out_matrix = (long int**)malloc(out_col * sizeof(long int*));
    for(int i = 0; i < out_row; i++) 
        out_matrix[i] = (long int*)malloc(out_row * sizeof(long int));

    for(int i = 0; i < out_col; i++) {
        for(int j = 0; j < out_row; j++) {
            out_matrix[i][j] = 0;
        }
    }


    // Parse the input csv files and fill in the input matrices
    readCSVtoMatrix(inputMatrix1, matrix1);
    readCSVtoMatrix(inputMatrix2, matrix2);


    // We are interesting in timing the matrix-matrix multiplication only
    // Record the start time
    double start = omp_get_wtime();
    
    // Parallelize the matrix-matrix multiplication
#   pragma omp parallel for num_threads(thread_count)
    for(int i = 0; i < out_col; i++) {
        for(int j = 0; j < out_row; j++) {
            out_matrix[i][j] += matrix1[i][j] * matrix2[j][i];
        }
    }

    // Record the finish time        
    double end = omp_get_wtime();
    
    // Time calculation (in seconds)
    double time_passed = end - start;

    // Save time to file
    fprintf(outputTime, "%f", time_passed);

    // TODO: save the output matrix to the output csv file

    // Cleanup
    fclose(inputMatrix1);
    fclose(inputMatrix2);
    fclose(outputFile);
    fclose(outputTime);
    // Remember to free your buffers!

    return 0;
}

