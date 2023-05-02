#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

void dfs(int **graph, bool *visited, int n_vertices, int vertex, int level) {
    visited[vertex] = true;
    printf("Visited vertex: %d, Level: %d\n", vertex, level);

    for (int i = 0; i < n_vertices; i++) {
        if (graph[vertex][i] && !visited[i]) {
            dfs(graph, visited, n_vertices, i, level + 1);
        }
    }
}

int main(int argc, char **argv) {
    int n_vertices = 20;
    int **graph = malloc(n_vertices * sizeof(int *));
    for (int i = 0; i < n_vertices; i++) {
        graph[i] = calloc(n_vertices, sizeof(int));
    }

// Edges: (0-1), (0-2), (1-3), (1-4), (2-5), (2-6), (3-7), (4-7), (5-8), (5-9), (6-9), (7-10), (7-11), (8-12), (8-13), (9-14), (9-15), (10-16), (11-17), (12-18), (13-18), (14-19), (15-19), (16-19), (17-19)
graph[0][1] = graph[1][0] = 1;
graph[0][2] = graph[2][0] = 1;
graph[1][3] = graph[3][1] = 1;
graph[1][4] = graph[4][1] = 1;
graph[2][5] = graph[5][2] = 1;
graph[2][6] = graph[6][2] = 1;
graph[3][7] = graph[7][3] = 1;
graph[4][7] = graph[7][4] = 1;
graph[5][8] = graph[8][5] = 1;
graph[5][9] = graph[9][5] = 1;
graph[6][9] = graph[9][6] = 1;
graph[7][10] = graph[10][7] = 1;
graph[7][11] = graph[11][7] = 1;
graph[8][12] = graph[12][8] = 1;
graph[8][13] = graph[13][8] = 1;
graph[9][14] = graph[14][9] = 1;
graph[9][15] = graph[15][9] = 1;
graph[10][16] = graph[16][10] = 1;
graph[11][17] = graph[17][11] = 1;
graph[12][18] = graph[18][12] = 1;
graph[13][18] = graph[18][13] = 1;
graph[14][19] = graph[19][14] = 1;
graph[15][19] = graph[19][15] = 1;
graph[16][19] = graph[19][16] = 1;
graph[17][19] = graph[19][17] = 1;
    // ...

    bool *visited = calloc(n_vertices, sizeof(bool));
    int start_vertex = 0;
    printf("Depth-First Search from vertex %d:\n", start_vertex);

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size == 1) {
        dfs(graph, visited, n_vertices, start_vertex, 0);
    } else {
        if (rank == 0) {
            printf("Error: MPI implementation of DFS requires only 1 process, but %d processes were provided.\n", size);
        }
    }

    MPI_Finalize();

    for (int i = 0; i < n_vertices; i++) {
        free(graph[i]);
    }
    free(graph);
    free(visited);

    return 0;
}
