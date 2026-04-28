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
// #include <mpi.h>
#include <queue>
#include <string>
#include <vector>

std::vector<int> parallel_bfs(const Graph &g, int src) {
    std::vector<int> dists;
    std::vector<bool> frontier{}; // Current frontier nodes
    std::vector<bool> new_frontier{};
    dists.reserve(g.num_nodes);
    frontier.reserve(g.num_nodes);
    new_frontier.reserve(g.num_nodes);

    for (int i = 0; i < g.num_nodes; ++i) {
        dists.push_back(-1); // Signifies a node hasn't been visited
        frontier.push_back(false);
        new_frontier.push_back(false);
    }
    frontier[src] = true;
    dists[src] = 0;

    int distance = 1;
    while (true) {
        bool has_frontier = false;
        // Vector-vector multiplication of [each node's outgoing edges] x
        // [current frontier]
        for (int row = 0; row < g.num_nodes; ++row) {
            if (dists[row] >= 0) { // Skip rows for nodes that've been visited
                continue;
            }
            for (int index = g.row_ptr[row]; index < g.row_ptr[row + 1];
                 ++index) {
                if (frontier[g.col_ind[index]] && dists[row] == -1) {
                    new_frontier[row] = true;
                    dists[row] = distance;
                    has_frontier = true;
                }
            }
        }

        // If the new frontier contains any nodes...
        if (has_frontier) {
            // Update the current frontier by swapping
            frontier.swap(new_frontier);
            // Finished processing this layer of nodes, so increment distance
            distance++;
        } else {
            break;
        }
    }

    return dists;
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
        g = load_graph(dataset_file_name, true);
    } catch (const std::exception &e) {
        std::cerr << "Error loading the graph??" << std::endl;
        return 1;
    }

    if (source < 0 || source >= g.num_nodes) {
        std::cerr << "Source node " << source << " out of range [0, "
                  << g.num_nodes << ")\n";
        return 1;
    }

    // MPI_Init(&argc, &argv);
    // MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "Running parallel BFS from source node: " << source
              << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    std::vector<int> res = parallel_bfs(g, source);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n----- Parallel BFS Results -----\n";
    std::cout << "  Source node      : " << source << std::endl;
    std::cout << "  Nodes visited    : " << nodes_visited(res) << std::endl;
    std::cout << "  Node ID space    : " << g.num_nodes << " (max_node_id + 1)"
              << std::endl;
    std::cout << "  Elapsed time     : " << elapsed << " seconds\n";

    print_distances(res, 50);

    // MPI_Finalize();

    return 0;
}
