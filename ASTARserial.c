#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#define V 10000

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


int heuristic(int node) {
    int dx = abs(node / sqrt(V) - goal / sqrt(V));
    int dy = abs(node % (int)sqrt(V) - goal % (int)sqrt(V));
    return dx + dy;
}


int minDistance() {
    int min = INT_MAX, min_index;

    for(int i = 0; i < V; i++) {
        if(!visited[i] && dist[i] + heuristic(i) < min) {
            min = dist[i] + heuristic(i);
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

    clock_t start_time = clock();

    for(int count = 0; count < V - 1; count++) {
        int u = minDistance();
        visited[u] = true;

        if(u == goal) {
            break;
        }

        for(int v = 0; v < V; v++) {
            if(!visited[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }

    clock_t end_time = clock();

    printf("Execution time: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
}

int main() {
    srand(time(NULL));
    initializeGraph();

    start = 0;
    goal = V - 1;

    astar();

    printf("Shortest distance between nodes %d and %d: %d\n", start, goal, dist[goal]);

    return 0;
}
