#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

void bfs_kernel(int *graph, int *visited, int *level, int n_vertices, int current_level) {

    for (int i = 0; i < n_vertices; i++) {
        if (visited[i] && level[i] == current_level) {
            for (int j = 0; j < n_vertices; j++) {
                int edge = graph[i * n_vertices + j];
                if (edge && !visited[j]) {
                    visited[j] = 1;
                    level[j] = current_level + 1;
                }
            }
        }
    }
}

void generate_complex_graph(int *graph, int n_vertices) {
    srand(time(NULL));

    // Clear the graph
    for (int i = 0; i < n_vertices * n_vertices; i++) {
        graph[i] = 0;
    }

    // Connect vertices in a more complex way
    for (int i = 0; i < n_vertices; i++) {
        int edges_to_add = rand() % (n_vertices / 5) + 1;
        for (int j = 0; j < edges_to_add; j++) {
            int target = rand() % n_vertices;
            if (i != target) {
                graph[i * n_vertices + target] = 1;
            }
        }
    }
}


void bfs(int *graph, int n_vertices, int start_vertex) {
    int *visited = (int *)calloc(n_vertices, sizeof(int));
    int *level = (int *)malloc(n_vertices * sizeof(int));
    memset(level, -1, n_vertices * sizeof(int));

    visited[start_vertex] = 1;
    level[start_vertex] = 0;

    int current_level = 0;
    int has_next_level = 1;

    while (has_next_level) {
        has_next_level = 0;

        bfs_kernel(graph, visited, level, n_vertices, current_level);

        for (int i = 0; i < n_vertices; i++) {
            if (level[i] == current_level + 1) {
                has_next_level = 1;
                break;
            }
        }

        current_level++;
    }

    for (int i = 0; i < n_vertices; i++) {
        printf("Vertex: %d, Level: %d\n", i, level[i]);
    }

    free(visited);
    free(level);
}

int main() {

    int n_vertices = 10000;
    float edge_probability = 0.1;

    int *graph = (int *)malloc(n_vertices * n_vertices * sizeof(int));

    generate_complex_graph(graph, n_vertices);

    clock_t begin = clock();

    bfs(graph, n_vertices, 0);

    clock_t end = clock();

    double elapsed_time = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
    printf("Elapsed time: %.2f milliseconds\n", elapsed_time);

    free(graph);

    return 0;
}
