#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
void bfs_kernel(int *graph, int *visited, int *level, int n_vertices, int *current_level_vertices, int current_level_size, int *next_level_vertices, int *next_level_size) {
    #pragma omp parallel
    {
        int local_next_level_size = 0;
        int *local_next_level_vertices = (int *)malloc(n_vertices * sizeof(int));

        #pragma omp for
        for (int i = 0; i < current_level_size; i++) {
            int u = current_level_vertices[i];
            for (int j = 0; j < n_vertices; j++) {
                int edge = graph[u * n_vertices + j];
                if (edge && !visited[j]) {
                    visited[j] = 1;
                    level[j] = level[u] + 1;
                    local_next_level_vertices[local_next_level_size++] = j;
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < local_next_level_size; i++) {
                next_level_vertices[*next_level_size] = local_next_level_vertices[i];
                (*next_level_size)++;
            }
        }

        free(local_next_level_vertices);
    }
}

void generate_complex_graph(int *graph, int n_vertices) {
    srand(time(NULL));

    // Clear the graph
    memset(graph, 0, n_vertices * n_vertices * sizeof(int));

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

    int current_level_size = 1;
    int *current_level_vertices = (int *)malloc(n_vertices * sizeof(int));
    current_level_vertices[0] = start_vertex;

    while (current_level_size > 0) {
        int *next_level_vertices = (int *)malloc(n_vertices * sizeof(int));
        int next_level_size = 0;

        bfs_kernel(graph, visited, level, n_vertices, current_level_vertices, current_level_size, next_level_vertices, &next_level_size);

        free(current_level_vertices);
        current_level_vertices = next_level_vertices;
        current_level_size = next_level_size;
    }

    for (int i = 0; i < n_vertices; i++) {
        printf("Vertex: %d, Level: %d\n", i, level[i]);
    }

    free(visited);
    free(level);
    free(current_level_vertices);
}

int main() {
    omp_set_num_threads(4);

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