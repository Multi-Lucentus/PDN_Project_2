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

void multMatrixVector(long int* res_matrix, long int* matrix, long int* vector, int col_num, int height, int width, int mat_width);


/**
 * Main function
 * Performs matrix multiplication in parallel using OpenMP
 */
int main(int argc, char* argv[]) {
    // Catch console errors
    if (argc != 10) {
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
    long int* matrix1 = (long int*)malloc((n_col1 * n_row1) * sizeof(long int));

    long int* matrix2 = (long int*)malloc((n_col2 * n_row2) * sizeof(long int));
 

    // Determine the dims of the output matrix and initialize all cells to 0
    int out_col = n_col2;
    int out_row = n_row1;

    long int* out_matrix = (long int*)malloc((out_col * out_row) * sizeof(long int));

    for(int row = 0; row < out_row; row++) {
        for(int col = 0; col < out_col; col++) {
            out_matrix[row * out_col + col] = 0;
        }
    }

    // Parse the input csv files and fill in the input matrices
    readCSVtoMatrix(inputMatrix1, matrix1, n_col1, n_row1);

    readCSVtoMatrix(inputMatrix2, matrix2, n_col2, n_row2);


    // We are interesting in timing the matrix-matrix multiplication only
    // Record the start time
    double start = omp_get_wtime();
    

    // Parallelize the matrix-matrix multiplication
    // Use the matrix*vector multiplication function created for project 1
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

    // Record the finish time        
    double end = omp_get_wtime();
    
    // Time calculation (in seconds)
    double time_passed = end - start;

    // Save time to file
    fprintf(outputTime, "%f", time_passed);

    // Save the output matrix to the output csv file
    writeMatrixtoCSV(outputFile, out_matrix, out_col, out_row);

    // Free matrix memory
    free(matrix1);
    free(matrix2);
    free(out_matrix);

    // Cleanup
    fclose(inputMatrix1);
    fclose(inputMatrix2);
    fclose(outputFile);
    fclose(outputTime);

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


/**
 * Multiplies a given matrix with a vector and
 * returns the resulting vector
 *
 * Parameters:  res_matrix is the resulting matrix of matrix being multiplied with vector
 *              matrix is matrix A which will be multiplied with the column vector
 *              vector is the column vector of matrix B that will be multiplied with matrix
 *              col_num is the number of the column vector that is being multiplied
 *              height is the height of the resulting matrix (equivalent to the height of matrix A)
 *              width is the width of the resulting matrix (equivalent to the width of matrix B)
 *              mat_width is the width of matrix A
 */
void multMatrixVector(long int* res_matrix, long int* matrix, long int* vector, int col_num, int height, int width, int mat_width) {
    
    for(int row = 0; row < height; row++) {
        for(int col = 0; col < mat_width; col++) {
            res_matrix[row * width + col_num] += matrix[row * mat_width + col] * vector[col];
        }
    }
}