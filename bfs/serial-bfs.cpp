/*
Serial BFS for roadNet-CA
How to run:
   Compile: g++ -O2 -std=c++23 -o bfs_serial serial-bfs.cpp
   Run:     ./bfs_serial ../datasets/roadNet-CA.txt <source_node>
       ex:  ./bfs_serial ../datasets/roadNet-CA.txt 0
*/

#include "graph_utils.h"
#include <chrono>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

// BFS for single connected component
std::vector<int> bfs(const Graph &g, int src) {
    std::vector<bool> visited(g.num_nodes, false);
    std::vector<int> distances(g.num_nodes, -1);
    std::queue<int> queue;

    visited[src] = true;
    queue.push(src);
    distances[src] = 0;

    while (!queue.empty()) {
        int curr = queue.front();
        queue.pop();

        // visit all the unvisited neighbours of curr node
        for (long index_for_col_ind = g.row_ptr[curr];
             index_for_col_ind < g.row_ptr[curr + 1]; index_for_col_ind++) {
            int neighbor = g.col_ind[index_for_col_ind];
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                distances[neighbor] = distances[curr] + 1;
                queue.push(neighbor);
            }
        }
    }

    return distances;
}

std::vector<int> parallel_bfs(const Graph &g, int src) {
    // TODO: write parallel version
    std::vector<int> res;
    return res;
}

int main(int argc, char *argv[]) { // argv looks like {./bfs_serial,
                                   // ../datasets/roadNet-CA.txt, 0}
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <source_node>"
                  << std::endl;
        return 1;
    }

    const std::string dataset_file_name = argv[1];
    int source = std::stoi(argv[2]);

    // load da graph
    std::cout << "Loading graph from: " << dataset_file_name << std::endl;
    Graph g;
    try {
        g = load_graph(dataset_file_name);
    } catch (const std::exception &e) {
        std::cerr << "Error loading the graph??" << std::endl;
        return 1;
    }

    if (source < 0 || source >= g.num_nodes) {
        std::cerr << "Source node " << source << " out of range [0, "
                  << g.num_nodes << ")\n";
        return 1;
    }

    std::cout << "Running serial BFS from source node: " << source << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    std::vector<int> distances = bfs(g, source);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n----- Serial BFS Results -----\n";
    std::cout << "  Source node      : " << source << std::endl;
    std::cout << "  Nodes visited    : " << nodes_visited(distances) << std::endl;
    std::cout << "  Node ID space    : " << g.num_nodes << " (max_node_id + 1)"
              << std::endl;
    std::cout << "  Elapsed time     : " << elapsed << " seconds\n";

    return 0;
}
