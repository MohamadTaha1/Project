#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>


void generate_complex_graph(int *graph, int n_vertices);
void bfs(int *graph, int n_vertices, int start_vertex, int world_rank, int world_size);

int main(int argc, char *argv[]) {
    int world_rank, world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n_vertices = 10000;

    int *graph = NULL;
    if (world_rank == 0) {
        graph = (int *)malloc(n_vertices * n_vertices * sizeof(int));
        generate_complex_graph(graph, n_vertices);
    }

    bfs(graph, n_vertices, 0, world_rank, world_size);

    if (world_rank == 0) {
        free(graph);
    }

    MPI_Finalize();
    return 0;
}

void bfs(int *graph, int n_vertices, int start_vertex, int world_rank, int world_size) {
    int *visited = (int *)calloc(n_vertices, sizeof(int));
    int *level = (int *)malloc(n_vertices * sizeof(int));
    memset(level, -1, n_vertices * sizeof(int));

    if (world_rank == 0) {
        visited[start_vertex] = 1;
        level[start_vertex] = 0;
    }

    int *local_graph = (int *)malloc((n_vertices / world_size) * n_vertices * sizeof(int));
    MPI_Scatter(graph, (n_vertices / world_size) * n_vertices, MPI_INT, local_graph, (n_vertices / world_size) * n_vertices, MPI_INT, 0, MPI_COMM_WORLD);

    int current_level = 0;
    int has_next_level = 1;

    while (has_next_level) {
        int local_has_next_level = 0;

        for (int i = world_rank * (n_vertices / world_size); i < (world_rank + 1) * (n_vertices / world_size); i++) {
            if (visited[i] && level[i] == current_level) {
                for (int j = 0; j < n_vertices; j++) {
                    int edge = local_graph[(i - world_rank * (n_vertices / world_size)) * n_vertices + j];
                    if (edge && !visited[j]) {
                        visited[j] = 1;
                        level[j] = current_level + 1;
                        local_has_next_level = 1;
                    }
                }
            }
        }

        MPI_Allreduce(&local_has_next_level, &has_next_level, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, visited, n_vertices / world_size, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, level, n_vertices / world_size, MPI_INT, MPI_COMM_WORLD);

        current_level++;
    }

    if (world_rank == 0) {
        for (int i = 0; i < n_vertices; i++) {
            printf("Vertex: %d, Level: %d\n", i, level[i]);
        }
    }

    free(local_graph);
    free(visited);
    free(level);
}
