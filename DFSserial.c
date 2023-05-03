#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctime>

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

int main() {
    int n_vertices = 10000;
    int **graph = malloc(n_vertices * sizeof(int *));
    for (int i = 0; i < n_vertices; i++) {
        graph[i] = calloc(n_vertices, sizeof(int));
    }

    generate_large_structured_graph(graph, n_vertices);

    bool *visited = calloc(n_vertices, sizeof(bool));
    int start_vertex = 0;
    printf("Depth-First Search from vertex %d:\n", start_vertex);
    clock_t begin = clock();
    dfs(graph, visited, n_vertices, start_vertex, 0);
    clock_t end = clock();
    double elapsed_time = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
    printf("Elapsed time: %.2f milliseconds\n", elapsed_time);
    for (int i = 0; i < n_vertices; i++) {
        free(graph[i]);
    }
    free(graph);
    free(visited);
    

    return 0;
}