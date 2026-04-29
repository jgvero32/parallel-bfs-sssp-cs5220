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
#include <mpi.h>
#include <omp.h>
#include <queue>
#include <string>
#include <vector>

// Stores the distances from source node
static std::vector<int> dists;
// This rank's local frontier
static std::vector<bool> frontier;
static int start_row;
static int end_row; // non-inclusive
static std::vector<int> displacements;
static std::vector<int> rows_per_proc;
static std::vector<int> result;

/**
 *
 */
void initialize(const Graph &graph, int source, int rank, int num_procs) {
    if (rank >= graph.num_nodes) return;

    dists.reserve(graph.num_nodes);
    frontier.reserve(graph.num_nodes);
    for (int i = 0; i < graph.num_nodes; ++i) {
        dists.push_back(-1); // Signifies a node hasn't been visited
        frontier.push_back(false);
    }

    // TODO: load-balancing nodes based on number of outgoing edges
    // Divide nodes evenly between processors
    int base_rows = graph.num_nodes / num_procs;
    int remainder_rows = graph.num_nodes % num_procs; // Number of ranks with extra row
    displacements.reserve(num_procs);
    rows_per_proc.reserve(num_procs);

    int disp = 0;
    for (int i = 0; i < num_procs; ++i) {
        displacements.push_back(disp);
        if (i < remainder_rows) {
            disp += base_rows + 1;
            rows_per_proc.push_back(base_rows + 1);
        } else {
            disp += base_rows;
            rows_per_proc.push_back(base_rows);
        }
    }

    start_row = displacements[rank];
    if (displacements.size() - 1 == rank) {
        end_row = graph.num_nodes;
    } else {
        end_row = displacements[rank+1];
    }

    // Create the first frontier
    if (start_row <= source && source < end_row) {
        frontier[source] = true;
        dists[source] = 0;
    }
}

void parallel_bfs(const Graph &g, int rank, int num_procs) {
    /*
    As long as frontier isn't 0, continue loop
    Construct new frontier and then send nodes to respective owners
    Communication is n(p-1) per step.
    Creates new_frontier of size avg_size * num_procs for later all-to-all
    */
    if (rank >= g.num_nodes) return;

    std::vector<int> new_frontier;
    new_frontier.resize(g.num_nodes);
    new_frontier.assign(g.num_nodes, 0);
    std::vector<int> recv_vals;
    recv_vals.resize((end_row-start_row)*num_procs);
    int displacements_rank[num_procs];
    int row_ct_per_rank[num_procs];
    for (int i = 0; i < num_procs; ++i) {
        displacements_rank[i] = (end_row-start_row)*i;
        row_ct_per_rank[i] = end_row-start_row;
    }


    int distance = 1;
    while (true) {
        for (int row = start_row; row < end_row; ++row) {
            if (frontier[row]) {
                for (int index = g.row_ptr[row]; index < g.row_ptr[row + 1];
                 ++index) {
                    // Add all the neighbors of this node to the new frontier
                    new_frontier[g.col_ind[index]] = 1;
                }
            }
        }

        MPI_Alltoallv(new_frontier.data(), rows_per_proc.data(), displacements.data(),
            MPI_INT, recv_vals.data(), row_ct_per_rank, displacements_rank,
            MPI_INT, MPI_COMM_WORLD);

        int has_frontier = 0;
        for (int row = start_row; row < end_row; ++row) {
            frontier[row] = false;
            for (int i = 0; i < num_procs; ++i) {
                int offset = row - start_row;
                // Found unvisited node --> add to local frontier
                if (recv_vals[i * (end_row - start_row) + offset] == 1 && dists[row] == -1) {
                    frontier[row] = true;
                    dists[row] = distance;
                    has_frontier = 1;
                    break;
                }
            }
        }

        // Check if any new frontier is non-zero
        int frontier_nodes;
        MPI_Allreduce(&has_frontier, &frontier_nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (frontier_nodes > 0) {
            // Finished processing one layer of nodes, clear the "next" frontier
            std::fill(new_frontier.begin(), new_frontier.end(), 0);
            distance++;
        } else {
            break;
        }
    }
}

/**
 * Gathers the parts of the dists vector onto rank 0
 */
void gather_result(int num_nodes, int rank, int num_procs) {
    if (rank >= num_nodes) return;
    // TODO: can I just use a pointer to start index instead of making a new
    // vector?
    std::vector<int> rank_dists(dists.begin() + start_row,
                                dists.begin() + end_row);

    if (rank == 0) {
        result.resize(num_nodes);
        MPI_Gatherv(rank_dists.data(), rank_dists.size(), MPI_INT, result.data(),
                    rows_per_proc.data(), displacements.data(), MPI_INT, 0,
                    MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(rank_dists.data(), rank_dists.size(), MPI_INT, NULL, NULL,
                    NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <source_node>"
                  << std::endl;
        return 1;
    }

    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::string dataset_file_name = argv[1];
    int source = std::stoi(argv[2]);

    // load da graph
    if (rank == 0) {
        std::cout << "Loading graph from: " << dataset_file_name << std::endl;
    }
    Graph g;
    try {
        g = load_graph(dataset_file_name,
                       false); // Row i contains outgoing edges for node i
    } catch (const std::exception &e) {
        std::cerr << "Error loading the graph??" << std::endl;
        return 1;
    }

    if (source < 0 || source >= g.num_nodes) {
        std::cerr << "Source node " << source << " out of range [0, "
                  << g.num_nodes << ")\n";
        return 1;
    }

    if (rank == 0) {
        std::cout << "Running parallel BFS from source node: " << source
            << std::endl;
    }

    auto t0 = std::chrono::steady_clock::now();

    initialize(g, source, rank, num_procs);

    parallel_bfs(g, rank, num_procs);

    gather_result(g.num_nodes, rank, num_procs);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    if (rank == 0) {
        std::cout << "\n----- Parallel BFS Results -----\n";
        std::cout << "  Source node      : " << source << std::endl;
        std::cout << "  Nodes visited    : " << nodes_visited(result) << std::endl;
        std::cout << "  Node ID space    : " << g.num_nodes << " (max_node_id + 1)"
                << std::endl;
        std::cout << "  Elapsed time     : " << elapsed << " seconds\n";

        print_distances(result, 50);
    }

    MPI_Finalize();

    return 0;
}
