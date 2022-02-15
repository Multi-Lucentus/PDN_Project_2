#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <omp.h>

#define DEBUG 0
#define BUF_SIZE 1024

/* ----------- Project 2 - Problem 1 - Matrix Mult -----------

    This file will multiply two matrices in a parallel manner

*/ // ------------------------------------------------------ //

// Prototype Functions
void readCSVtoMatrix(FILE* fp, long int* in_matrix, int width, int height);
void writeMatrixtoCSV(FILE* fp, long int* out_matrix, int n_col, int n_row);

long int dotProduct(long int* matrix1, long int* matrix2, int width, int col, int row);


/**
 * Main function
 * Performs matrix multiplication in parallel using OpenMP
 */
int main(int argc, char* argv[]) {
    // Catch console errors
    if (argc != 9) {
        printf("USE LIKE THIS: parallel_mult_mat_mat   mat_1.csv n_row_1 n_col_1   mat_2.csv n_row_2 n_col_2   num_threads   max_value.csv\n");
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


    // Create and malloc the two input matrices and the output matrix
    long int* matrix1 = (long int*)malloc((n_col1 * n_row1) * sizeof(long int));

    long int* matrix2 = (long int*)malloc((n_col2 * n_row2) * sizeof(long int));


    // Parse the input csv files and fill in the input matrices
    readCSVtoMatrix(inputMatrix1, matrix1, n_col1, n_row1);
    readCSVtoMatrix(inputMatrix2, matrix2, n_col2, n_row2);

    // Parallelize the max comparison
#   pragma omp parallel for num_threads(thread_count)
    for(int i = 0; i < n_col2; i++) {
        // Multiply each column vector of matrix B with matrix A and save result to C

        // Create a vector to store a column vector of matrix B in
        long int* vector = (long int*)malloc(n_row2 * sizeof(long int));
        
        for(int j = 0; j < n_row2; j++) {
            vector[j] = matrix2[j * n_col2 + i];
        }

        // Multiply the vector with Matrix A
        multMatrixVector(out_matrix, matrix1, vector, i, n_row1, n_col2, n_col1);

        // Free vectors
        free(vector);
    }

    long int max_value = 0;
#   pragma omp parallel for num_threads(thread_count) reduction (max : max_value)
    for(int row = 0; row < out_row; row++) {
        for(int col = 0; col < out_col; col++) {
            // long int value = out_matrix[row * out_col + col];
            long int value = dotProduct(matrix1, matrix2, n_col1, col, row);

#           pragma omp critical
            if(value > max_value)
                max_value = value;
        }
    }

    // Save time to file
    fprintf(outputTime, "%f", time_passed);

    // Save the output maximum to the file
    fprintf(outputFile, "%ld", max_value);

    // Free matrix memory
    free(matrix1);
    free(matrix2);

    // Cleanup
    fclose(inputMatrix1);
    fclose(inputMatrix2);
    fclose(outputFile);

    return 0;
}


/**
 * Reads in a CSV file given by the file pointer fp, and then 
 * reads in each cell of CSV into the corresponding matrix cell of
 * in_matrix
 *
 * Parameters:  fp is the file pointer to the input CSV file
 *              in_matrix is the matrix pointer to write the CSV data to
 *              width is the width (or number of columns) of the matrix
 *              height is the height (or number of rows) of the matrix
 */
void readCSVtoMatrix(FILE* fp, long int* in_matrix, int width, int height) {
    char* line_buffer = (char*)malloc(BUF_SIZE * sizeof(char));
    char* token;

    int row = 0, col = 0;

    // Read each token and input into the matrix
    while(fgets(line_buffer, BUF_SIZE, fp) != NULL) {
        token = strtok(line_buffer, ",");

        while(token != NULL) {
            in_matrix[row * width + col] = strtol(token, NULL, 10);

            col += 1;

            token = strtok(NULL, ",");
        }

        row += 1;
        col = 0;
    }

    // Free buffers
    free(line_buffer);
}


/**
 * Writes the given matrix, out_matrix, to the output file given by fp
 *
 * Parameters:  fp is the file pointer to the output CSV file to write the matrix to
 *              out_matrix is the matrix that will be written to the CSV file
 *              n_col is the number of columns the output matrix contains
 *              n_row is the number of rows the output matrix contains
 */
void writeMatrixtoCSV(FILE* fp, long int* out_matrix, int n_col, int n_row) {
    char* output_buffer = (char*)malloc(BUF_SIZE * sizeof(char));

    for(int i = 0; i < n_row; i++) {
        for(int j = 0; j < n_col; j++) {
            sprintf(output_buffer, "%ld", out_matrix[i * n_col + j]);
            fputs(output_buffer, fp);

            fputs(",", fp);
        }
        fputs("\n", fp);
    }

    // Free buffers
    free(output_buffer);
}

long int dotProduct(long int* matrix1, long int* matrix2, int width, int col, int row) {
    int result = 0;

    for(int i = 0; i < width; i++) {
        result += matrix1[row * width + i] * matrix2[i * width + col];
    }

    return result;
}