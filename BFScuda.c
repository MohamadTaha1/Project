
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <vector>
#include <chrono>
#include <iostream>


__global__ void bfs_kernel(int *graph, bool *visited, int *level, int n_vertices, int current_level) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vertices && visited[i] && level[i] == current_level) {
        for (int j = 0; j < n_vertices; j++) {
            int edge = graph[i * n_vertices + j];
            if (edge && !visited[j]) {
                visited[j] = true;
                level[j] = current_level + 1;
            }
        }
    }
}

void generate_random_graph(std::vector<int> &graph, int n_vertices, float edge_probability) {
    srand(time(NULL));

    for (int i = 0; i < n_vertices; i++) {
        for (int j = 0; j < n_vertices; j++) {
            if (i == j) {
                graph[i * n_vertices + j] = 0;
            } else {
                float random_value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                graph[i * n_vertices + j] = random_value < edge_probability ? 1 : 0;
            }
        }
    }
}

void bfs(int *graph, int n_vertices, int start_vertex) {
    int *d_graph;
    bool *d_visited;
    int *d_level;

    size_t graph_size = n_vertices * n_vertices * sizeof(int);
    size_t visited_size = n_vertices * sizeof(bool);
    size_t level_size = n_vertices * sizeof(int);

    cudaMalloc((void **)&d_graph, graph_size);
    cudaMalloc((void **)&d_visited, visited_size);
    cudaMalloc((void **)&d_level, level_size);

    cudaMemcpy(d_graph, graph, graph_size, cudaMemcpyHostToDevice);

    bool *visited = (bool *)calloc(n_vertices, sizeof(bool));
    int *level = (int *)malloc(n_vertices * sizeof(int));
    memset(level, -1, n_vertices * sizeof(int));

    visited[start_vertex] = true;
    level[start_vertex] = 0;

    cudaMemcpy(d_visited, visited, visited_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, level, level_size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (n_vertices + threads_per_block - 1) / threads_per_block;

    int current_level = 0;
    bool has_next_level = true;

    while (has_next_level) {
        has_next_level = false;

        bfs_kernel<<<blocks_per_grid, threads_per_block>>>(d_graph, d_visited, d_level, n_vertices, current_level);

        cudaMemcpy(visited, d_visited, visited_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(level, d_level, level_size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < n_vertices; i++) {
            if (level[i] == current_level + 1) {
                has_next_level = true;
                break;
            }
        }

        current_level++;
    }

    for (int i = 0; i < n_vertices; i++) {
        printf("Vertex: %d, Level: %d\n", i, level[i]);
    }

    cudaFree(d_graph);
    cudaFree(d_visited);
    cudaFree(d_level);

    free(visited);
    free(level);
}

int main() {
   
    int n_vertices = 10000;
    float edge_probability = 0.1;

    std::vector<int> graph(n_vertices * n_vertices);

    generate_random_graph(graph, n_vertices, edge_probability);

    bfs(graph.data(), n_vertices, 0);

    return 0;
}

