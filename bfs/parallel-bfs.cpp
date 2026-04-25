/*
In progress...

Parallel BFS for roadNet-CA
How to run:
   Compile: g++ -O2 -std=c++23 -o bfs_parallel parallel-bfs.cpp
   Run:     ./bfs_parallel ../datasets/roadNet-CA.txt <source_node>
       ex:  ./bfs_parallel ../datasets/roadNet-CA.txt 0
*/

#include "graph_utils.h"
#include <chrono>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

std::vector<int> parallel_bfs(const Graph &g, int src) {
    // TODO: write parallel version
    std::vector<int> res;
    return res;
}

int main(int argc, char *argv[]) {
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

    std::vector<int> res = parallel_bfs(g, source);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n----- Serial BFS Results -----\n";
    std::cout << "  Source node      : " << source << std::endl;
    std::cout << "  Nodes visited    : " << res.size() << std::endl;
    std::cout << "  Node ID space    : " << g.num_nodes << " (max_node_id + 1)"
              << std::endl;
    std::cout << "  Elapsed time     : " << elapsed << " seconds\n";

    return 0;
}
