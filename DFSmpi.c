#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

void generate_large_structured_graph(int **graph, int n_vertices) {
    // Clear the graph
    for (int i = 0; i < n_vertices; i++) {
        for (int j = 0; j < n_vertices; j++) {
            graph[i][j] = 0;
        }
    }

    // Connect vertices in a structured way
    for (int i = 0; i < n_vertices - 1; i++) {
        for (int j = i + 1; j < i + 1 + (n_vertices / 200); j++) {
            if (j < n_vertices) {
                graph[i][j] = 1;
                graph[j][i] = 1;
            }
        }
    }
}

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
    int n_vertices = 10000;
    int **graph = malloc(n_vertices * sizeof(int *));
    for (int i = 0; i < n_vertices; i++) {
        graph[i] = calloc(n_vertices, sizeof(int));
    }

    generate_large_structured_graph(graph, n_vertices);

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