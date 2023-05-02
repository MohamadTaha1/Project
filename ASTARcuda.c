#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

typedef struct Node
{
    int x, y;
    double cost, priority;
} Node;

typedef struct PriorityQueue
{
    Node *nodes;
    int count;
    int capacity;
} PriorityQueue;

__device__ int heuristic(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

__device__ bool is_valid_move(int x, int y, int *grid, int grid_rows, int grid_cols)
{
    return x >= 0 && x < grid_rows && y >= 0 && y < grid_cols && grid[x * grid_cols + y] != -1;
}

__device__ void push(PriorityQueue *pq, Node node)
{
    if (pq->count == pq->capacity)
    {
        pq->capacity *= 2;
        Node *new_nodes = (Node *)malloc(pq->capacity * sizeof(Node));
        if (new_nodes)
        {
            memcpy(new_nodes, pq->nodes, pq->count * sizeof(Node));
            free(pq->nodes);
            pq->nodes = new_nodes;
        }
        else
        {
            printf("Memory allocation failed.\n");
            return;
        }
    }
    pq->nodes[pq->count++] = node;

    // Bubble up the new node
    int idx = pq->count - 1;
    while (idx > 0)
    {
        int parent_idx = (idx - 1) / 2;
        if (pq->nodes[parent_idx].priority > pq->nodes[idx].priority)
        {
            Node tmp = pq->nodes[parent_idx];
            pq->nodes[parent_idx] = pq->nodes[idx];
            pq->nodes[idx] = tmp;
            idx = parent_idx;
        }
        else
        {
            break;
        }
    }
}

__device__ Node pop(PriorityQueue *pq)
{
    Node node = pq->nodes[0];

    // Replace the root with the last node and bubble it down
    pq->nodes[0] = pq->nodes[--pq->count];
    int idx = 0;
    while (true)
    {
        int left_child_idx = 2 * idx + 1;
        int right_child_idx = 2 * idx + 2;
        int smallest_child_idx = -1;

        if (left_child_idx < pq->count)
        {
            smallest_child_idx = left_child_idx;
        }
        else
        {
            break;
        }

        if (right_child_idx < pq->count && pq->nodes[right_child_idx].priority < pq->nodes[left_child_idx].priority)
        {
            smallest_child_idx = right_child_idx;
        }

        if (pq->nodes[idx].priority > pq->nodes[smallest_child_idx].priority)
        {
            Node tmp = pq->nodes[smallest_child_idx];
            pq->nodes[smallest_child_idx] = pq->nodes[idx];
            pq->nodes[idx] = tmp;
            idx = smallest_child_idx;
        }
        else
        {
            break;
        }
    }

    return node;
}

__device__ bool is_empty(PriorityQueue *pq)
{
    return pq->count == 0;
}

__device__ void print_path(Node **came_from, Node start, Node goal, int grid_cols)
{
    Node current = goal;
    int path_length = 0;
    while (current.x != start.x || current.y != start.y)
    {
        printf("(%d, %d) <- ", current.x, current.y);
        current = came_from[current.x * grid_cols + current.y];
        path_length++;
    }
    printf("(%d, %d)\n", start.x, start.y);
    printf("Path length: %d\n", path_length);
}

__global__ void astar(int *grid, int grid_rows, int grid_cols, Node start, Node goal)
{
    PriorityQueue open_set;
    open_set.nodes = (Node *)malloc(grid_rows * grid_cols * sizeof(Node));
    open_set.count = 0;
    open_set.capacity = grid_rows * grid_cols;
    push(&open_set, start);

    Node *came_from = (Node *)malloc(grid_rows * grid_cols * sizeof(Node));
    double *g_scores = (double *)malloc(grid_rows * grid_cols * sizeof(double));
    for (int i = 0; i < grid_rows * grid_cols; i++)
    {
        g_scores[i] = INFINITY;
    }
    g_scores[start.x * grid_cols + start.y] = 0.0;

    int dx[] = {-1, 0, 1, 0};
    int dy[] = {0, 1, 0, -1};

    while (!is_empty(&open_set))
    {
        Node current = pop(&open_set);

        if (current.x == goal.x && current.y == goal.y)
        {
            print_path(came_from, start, goal, grid_cols);
            free(came_from);
            free(open_set.nodes);
            free(g_scores);
            return;
        }

        for (int i = 0; i < 4; i++)
        {
            int x = current.x + dx[i];
            int y = current.y + dy[i];

            if (is_valid_move(x, y, grid, grid_rows, grid_cols))
            {
                double tentative_g_score = g_scores[current.x * grid_cols + current.y] + grid[x * grid_cols + y];

                if (tentative_g_score < g_scores[x * grid_cols + y])
                {
                    came_from[x * grid_cols + y] = current;
                    g_scores[x * grid_cols + y] = tentative_g_score;
                    double f_score = tentative_g_score + heuristic(x, y, goal.x, goal.y);

                    Node neighbor;
                    neighbor.x = x;
                    neighbor.y = y;
                    neighbor.cost = tentative_g_score;
                    neighbor.priority = f_score;
                    push(&open_set, neighbor);
                }
            }
        }
    }

    printf("No path found.\n");
    free(came_from);
    free(open_set.nodes);
    free(g_scores);
}

int main()
{
    int grid_rows = 10;
    int grid_cols = 10;
    int grid[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -1, -1, 0, -1, -1, -1, 0, -1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -1, 0, -1, 0, -1, 0, -1, 0, 0,
        0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0,
        0, -1, -1, -1, -1, -1, -1, -1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -1, -1, -1, -1, -1, -1, -1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -1, 0, -1, 0, -1, 0, -1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    int *d_grid;
    cudaMalloc((void **)&d_grid, grid_rows * grid_cols * sizeof(int));
    cudaMemcpy(d_grid, grid, grid_rows * grid_cols * sizeof(int), cudaMemcpyHostToDevice);

    Node start;
    start.x = 0;
    start.y = 0;
    start.cost = 0;
    start.priority = heuristic(start.x, start.y, 9, 9);

    Node goal;
    goal.x = 9;
    goal.y = 9;
    goal.cost = 0;
    goal.priority = 0;

    astar<<<1, 1>>>(d_grid, grid_rows, grid_cols, start, goal);
    cudaDeviceSynchronize();

    cudaFree(d_grid);

    return 0;
}


