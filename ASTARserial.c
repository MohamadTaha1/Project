#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

typedef struct Node {
    int x, y;
    double cost, priority;
} Node;

typedef struct PriorityQueue {
    Node* nodes;
    int count;
    int capacity;
} PriorityQueue;

double heuristic(int x1, int y1, int x2, int y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

bool is_valid_move(int x, int y, int** grid, int grid_rows, int grid_cols) {
    return x >= 0 && x < grid_rows && y >= 0 && y < grid_cols && grid[x][y] != -1;
}

int compare(const void* a, const void* b) {
    Node* node_a = (Node*) a;
    Node* node_b = (Node*) b;
    return (node_a->priority > node_b->priority) - (node_a->priority < node_b->priority);
}

void push(PriorityQueue* pq, Node node) {
    if (pq->count == pq->capacity) {
        pq->capacity *= 2;
        Node* new_nodes = (Node*) realloc(pq->nodes, pq->capacity * sizeof(Node));
        if (new_nodes) {
            pq->nodes = new_nodes;
        } else {
            printf("Memory allocation failed.\n");
            exit(1);
        }
    }
    pq->nodes[pq->count++] = node;
    qsort(pq->nodes, pq->count, sizeof(Node), compare);
}

Node pop(PriorityQueue* pq) {
    Node node = pq->nodes[0];
    memmove(pq->nodes, pq->nodes + 1, (pq->count - 1) * sizeof(Node));
    pq->count--;
    return node;
}

bool is_empty(PriorityQueue* pq) {
    return pq->count == 0;
}

void print_path(Node** came_from, Node start, Node goal) {
    Node current = goal;
    printf("Path: ");
    while (current.x != start.x || current.y != start.y) {
        printf("(%d, %d) -> ", current.x, current.y);
        current = came_from[current.x][current.y];
    }
    printf("(%d, %d)\n", start.x, start.y);
}

void serial_a_star(int** grid, int grid_rows, int grid_cols, int start_x, int start_y, int goal_x, int goal_y) {
    PriorityQueue open_set = {.nodes = (Node*) malloc(sizeof(Node)), .count = 0, .capacity = 1};
    bool** closed_set = (bool**) malloc(grid_rows * sizeof(bool*));
    for (int i = 0; i < grid_rows; i++) {
        closed_set[i] = (bool*) calloc(grid_cols, sizeof(bool));
    }
    Node** came_from = (Node**) malloc(grid_rows * sizeof(Node*));
    for (int i = 0; i < grid_rows; i++) {
        came_from[i] = (Node*) malloc(grid_cols * sizeof(Node));
    }
    int moves[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};

    Node start_node = {start_x, start_y, 0, heuristic(start_x, start_y, goal_x, goal_y)};
    push(&open_set, start_node);

    while (!is_empty(&open_set)) {
        Node current = pop(&open_set);

        if (current.x == goal_x && current.y == goal_y) {
            print_path(came_from, start_node, current);
            break;
        }

        if (!closed_set[current.x][current.y]) {
            closed_set[current.x][current.y] = true;

            for (int i = 0; i < 8; i++) {
                int new_x = current.x + moves[i][0];
                int new_y = current.y + moves[i][1];

                if (is_valid_move(new_x, new_y, grid, grid_rows, grid_cols)) {
                    double new_cost = current.cost + heuristic(current.x, current.y, new_x, new_y);
                    if (!closed_set[new_x][new_y]) {
                        double new_priority = new_cost + heuristic(new_x, new_y, goal_x, goal_y);
                        Node neighbor = (Node){new_x, new_y, new_cost, new_priority};
                        push(&open_set, neighbor);
                        came_from[new_x][new_y] = current;
                    }
                }
            }
        }
    }

    // Clean up
    free(open_set.nodes);
    for (int i = 0; i < grid_rows; i++) {
        free(closed_set[i]);
        free(came_from[i]);
    }
    free(closed_set);
    free(came_from);
}



int main() {
    int grid_rows = 500;
    int grid_cols = 500;
    int start_x = 0;
    int start_y = 0;
    int goal_x = 320;
    int goal_y = 222;

    int** grid = (int**) malloc(grid_rows * sizeof(int*));
    for (int i = 0; i < grid_rows; i++) {
        grid[i] = (int*) calloc(grid_cols, sizeof(int));
    }

    // Seed the random number generator
    srand(time(NULL));

    // Add random obstacles
    float obstacle_probability = 0.3; // Adjust this value to control the density of obstacles (0.0 to 1.0)
    for (int i = 0; i < grid_rows; i++) {
        for (int j = 0; j < grid_cols; j++) {
            if (i == start_x && j == start_y) continue;
            if (i == goal_x && j == goal_y) continue;
            float random_value = (float)rand() / (float)RAND_MAX;
            if (random_value < obstacle_probability) {
                grid[i][j] = -1;
            }
        }
    }

    clock_t start_time = clock();
    serial_a_star(grid, grid_rows, grid_cols, start_x, start_y, goal_x, goal_y);
    clock_t end_time = clock();
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", execution_time);

    for (int i = 0; i < grid_rows; i++) {
        free(grid[i]);
    }
    free(grid);

    return 0;
}

