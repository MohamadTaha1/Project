#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>

#define V 20000

int graph[V][V]; 
int dist[V]; 
bool visited[V]; 
int start, goal; 


void initializeGraph() {
    int weights[V] = {0}; 
    for(int i = 0; i < V; i++) {
        for(int j = 0; j < V; j++) {
            if(i == j) {
                graph[i][j] = 0;
            } else {
                int w = rand() % 100; 
                graph[i][j] = weights[i] + w; 
                weights[j] += w; 
            }
        }
    }
}


int minDistance() {
    int min = INT_MAX, min_index = 0;

#pragma omp parallel for num_threads(4) reduction(min:min)
    for(int i = 0; i < V; i++) {
        if(!visited[i] && dist[i] < min) {
            min = dist[i];
            min_index = i;
        }
    }

    return min_index;
}


void astar() {
    for(int i = 0; i < V; i++) {
        dist[i] = INT_MAX;
        visited[i] = false;
    }

    dist[start] = 0;

    for(int count = 0; count < V - 1; count++) {
        int u = minDistance();
        visited[u] = true;

        if(u == goal) {
            break;
        }

#pragma omp parallel for num_threads(4)
        for(int v = 0; v < V; v++) {
            if(!visited[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }

    printf("Shortest distance between nodes %d and %d: %d\n", start, goal, dist[goal]);
}

int main() {
    srand(time(NULL));
    initializeGraph();

    start = 0;
    goal = V - 1;

    astar();

    return 0;
}
