#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define V 100000

int graph[V][V];
int visited[V];

void dfs(int u) {
    visited[u] = 1;

    for(int i=0; i<V; i++) {
        if(graph[u][i] && !visited[i])
            dfs(i);
    }
}

int main() {
    //initialize graph
    for(int i=0; i<V; i++) {
        for(int j=0; j<V; j++) {
            if(i == j)
                graph[i][j] = 0;
            else
                graph[i][j] = 1;
        }
    }

    clock_t start_time = clock();

    for(int i=0; i<V; i++) {
        if(!visited[i])
            dfs(i);
    }

    clock_t end_time = clock();

    printf("Execution time: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

    return 0;
}