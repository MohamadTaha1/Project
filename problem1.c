#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define ROWS 1024
#define COLS 1024

int main() {

    int i, j;
    double start_time, end_time;

    // Allocate memory for the image and edge image
    int image = (int) malloc(sizeof(int) * ROWS * COLS);
    int edge = (int) malloc(sizeof(int) * ROWS * COLS);

    // Initialize the image with random values
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            image[i*COLS+j] = rand() % 256;
        }
    }

    // Perform Sobel edge detection in parallel using OpenMP
    start_time = omp_get_wtime();
    #pragma omp parallel for shared(image, edge) private(i, j)
    for (i = 1; i < ROWS-1; i++) {
        for (j = 1; j < COLS-1; j++) {
            int gx = -1*image[(i-1)*COLS+j-1] + image[(i-1)*COLS+j+1]
                     -2*image[i*COLS+j-1]     + 2*image[i*COLS+j+1]
                     -1*image[(i+1)*COLS+j-1] + image[(i+1)*COLS+j+1];

            int gy = -1*image[(i-1)*COLS+j-1] - 2*image[(i-1)*COLS+j] - 1*image[(i-1)*COLS+j+1]
                     + image[(i+1)*COLS+j-1]  + 2*image[(i+1)*COLS+j]  + 1*image[(i+1)*COLS+j+1];

            int val = (int)sqrt(gx*gx + gy*gy);
            if (val > 255) {
                val = 255;
            }
            edge[i*COLS+j] = val;
        }
    }
    end_time = omp_get_wtime();

    // Write the edge image to a file
    FILE *fp;
    fp = fopen("edge.pgm", "wb");
    fprintf(fp, "P5\n%d %d\n255\n", COLS, ROWS);
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            fputc((unsigned char)edge[i*COLS+j], fp);
        }
    }
    fclose(fp);

    // Print the execution time
    printf("Execution time: %f seconds\n", end_time-start_time);

    // Clean up
    free(image);
    free(edge);

    return 0;
}