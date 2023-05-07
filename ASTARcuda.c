#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define BLOCK_SIZE 256

__device__ int d_min_distance_index(int* dist, bool* visited, int n) {
    int min_distance = INT_MAX;
    int min_index = -1;
    for (int i = 0; i < n; i++) {
        if (!visited[i] && dist[i] <= min_distance) {
            min_distance = dist[i];
            min_index = i;
        }
    }
    return min_index;
}

__device__ int d_heuristic(int* heuristic, int v, int goal) {
    return heuristic[v] + abs(v - goal);
}

__global__ void AstarKernel(int* graph, int* heuristic, int* dist, bool* visited, int* prev, int start, int goal) {
    int num_vertices = blockDim.x * gridDim.x;
    int v_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_id < num_vertices) {
        dist[v_id] = INT_MAX;
        visited[v_id] = false;
        prev[v_id] = -1;
    }
    __syncthreads();

    dist[start] = 0;
    for (int i = 0; i < num_vertices - 1; i++) {
        int u = d_min_distance_index(dist, visited, num_vertices);
        visited[u] = true;

        for (int v = 0; v < num_vertices; v++) {
            if (!visited[v] && graph[u * num_vertices + v] != 0 && dist[u] != INT_MAX) {
                int alt = dist[u] + graph[u * num_vertices + v] + d_heuristic(heuristic, v, goal);
                if (alt < dist[v]) {
                    dist[v] = alt;
                    prev[v] = u;
                }
            }
        }
    }
}

int main() {
    int num_vertices = 10000;
    int start = 0;
    int goal = 9999;

 
    int* graph = (int*)malloc(num_vertices * num_vertices * sizeof(int));
    int* heuristic = (int*)malloc(num_vertices * sizeof(int));
    int* dist = (int*)malloc(num_vertices * sizeof(int));
    int* prev = (int*)malloc(num_vertices * sizeof(int));

    

   
    int* d_graph, *d_heuristic, *d_dist, *d_prev;
    bool* d_visited;
    cudaMalloc(&d_graph, num_vertices * num_vertices * sizeof(int));
    cudaMalloc(&d_heuristic, num_vertices * sizeof(int));
    cudaMalloc(&d_dist, num_vertices * sizeof(int));
    cudaMalloc(&d_prev, num_vertices * sizeof(int));
    cudaMalloc(&d_visited, num_vertices * sizeof(bool));

   
    cudaMemcpy(d_graph, graph, num_vertices * num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heuristic, heuristic, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

   
    int num_blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    AstarKernel<<<num_blocks, BLOCK_SIZE>>>(d_graph, d_heuristic, d_dist, d_visited, d_prev, start, goal);


    cudaMemcpy(dist, d_dist, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

  
    cudaFree(d_graph);
    cudaFree(d_heuristic);
    cudaFree(d_dist);
    cudaFree(d_prev);
    cudaFree(d_visited);

   
    free(graph);
    free(heuristic);
    free(dist);
    free(prev);

    return 0;
}