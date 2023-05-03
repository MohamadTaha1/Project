#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h>

#define MAX_VERTICES 10000

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

Node* graph[MAX_VERTICES]; // adjacency list of the graph
bool visited[MAX_VERTICES]; // array to keep track of visited vertices
int n; // number of vertices in the graph

void dfs_kernel(int v, int depth) {
    visited[v] = true;

    for (Node* node = graph[v]; node != NULL; node = node->next) {
        int i = node->vertex;
        if (!visited[i]) {
            if (depth < 8) {
                #pragma omp task
                dfs_kernel(i, depth + 1);
            } else {
                dfs_kernel(i, depth + 1);
            }
        }
    }
}


void generate_large_structured_graph(int n_vertices) {
    // Clear the graph
    for (int i = 0; i < n_vertices; i++) {
        graph[i] = NULL;
    }

    // Connect vertices in a structured way
    for (int i = 0; i < n_vertices - 1; i++) {
        for (int j = i + 1; j < i + 1 + (n_vertices / 200); j++) {
            if (j < n_vertices) {
                Node* newNode = (Node*)malloc(sizeof(Node));
                newNode->vertex = j;
                newNode->next = graph[i];
                graph[i] = newNode;

                newNode = (Node*)malloc(sizeof(Node));
                newNode->vertex = i;
                newNode->next = graph[j];
                graph[j] = newNode;
            }
        }
    }
}

void dfs(int n_vertices, int start_vertex) {
    memset(visited, 0, n_vertices * sizeof(bool));
    #pragma omp parallel
    {
        #pragma omp single
        dfs_kernel(start_vertex, 0);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_vertices; i++) {
        printf("Vertex: %d, Visited: %d\n", i, visited[i]);
    }
}

int main() {
    n = MAX_VERTICES;

    generate_large_structured_graph(n);

    clock_t begin = clock();
    omp_set_num_threads(1);
    dfs(n, 0);

    clock_t end = clock();

    double elapsed_time = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
    printf("Elapsed time: %.2f milliseconds\n", elapsed_time);

    return 0;
}
